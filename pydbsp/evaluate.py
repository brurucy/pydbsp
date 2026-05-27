"""Evaluator. A steppable view over (Circuit, Storage).

Given a :class:`Circuit` (progress + compute rules) and a
:class:`Storage` backend, the :class:`Evaluator` exposes:

* :meth:`push`. Append a value to an input node at the next axis-0
  tick. Inputs are 1-D streams. Events arrive in order. Each push
  advances the input's frontier by one. No explicit timestamps, no
  out-of-order arrivals, no multi-axis pushes. All out of scope.
* :meth:`read`. Return the value at any node at any timestamp.
  Routes input nodes to storage (defaulting to ``group.identity()``
  when absent), derived nodes to their compute rule (memoised in
  storage on first compute).
* :meth:`compact`. GC pass: run :func:`propagate_backward` from
  caller-supplied cursors, then ``evict_dominated`` per node.
* :meth:`frontiers`. Current per-node settled frontier via
  :func:`propagate_forward`.

**Feedback wiring.** When a node's progress rule is
:class:`progress.Feedback`, the evaluator binds an extra reader
pointing back to this node. The self-edge. The recursion terminates
because :class:`compute.Prev` returns ``group.identity()`` at the
axis bottom. The memo cache (via :class:`Storage`) prevents the
recurrence from re-expanding exponentially.

**Group assumption.** This evaluator takes a single group for all
input nodes (homogeneous-``V`` inputs). The compute layer supports
heterogeneous types (``Map[V_in, V_out]``, etc.), but the evaluator
needs *some* group to define the "no-push-at-t returns identity" rule
for input nodes. Heterogeneous-input circuits need a richer evaluator
that knows the group per input node. Out of scope here.

**Reader contract.** :meth:`compact` is sound only under
*monotone-forward* reads: callers must read each cell at-or-after the
most recent compaction frontier. Reading at an evicted timestamp
returns ``group.identity()`` regardless of what was there before.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

from pydbsp.core import OMEGA, AbelianGroupOperation, Antichain

from pydbsp import compute as cpt
from pydbsp import progress as prg
from pydbsp.circuit import Circuit
from pydbsp.progress import NodeId, Time, propagate_backward, propagate_forward
from pydbsp.storage import Storage

# Sentinel for the "not in storage" case in :meth:`Evaluator.read`.
# Distinct from :data:`pydbsp.storage._MISSING`, which storage uses
# internally to mean "no default supplied — raise on absence".
_CACHE_MISS: Any = object()


@dataclass
class Evaluator[T: Time]:
    """A steppable view over a circuit and a storage backend."""

    circuit: Circuit[T]
    storage: Storage
    ctx: cpt.ComputeCtx
    group: AbelianGroupOperation[Any]

    # ---- Push: input writes -----------------------------------------------

    def push(
        self,
        input_node: NodeId,
        value: Any,
        t: Time | None = None,
    ) -> None:
        """Write ``value`` at ``t`` on an input node, advancing the
        input's frontier to include it.

        If ``t`` is omitted, default to **next axis-0 tick on a 1-D
        Input** . ``t = (max_so_far + 1,)`` (or ``(0,)`` for the first
        push). Higher-arity Inputs must pass ``t`` explicitly.

        DBSP assumption: monotone, in-order events.
        """
        rule = self.circuit.progress_rules[input_node]
        if not isinstance(rule, prg.Input):
            raise ValueError(f"node {input_node} is not an input (got {type(rule).__name__})")
        input_rule = cast(prg.Input[T], rule)
        if t is None:
            if input_rule.frontier.elements:
                next_t0 = max(e[0] for e in input_rule.frontier.elements) + 1
            else:
                next_t0 = 0
            t = (next_t0,)
        self.storage.write(input_node, t, value)
        input_rule.frontier.insert(cast(T, t))

    # ---- Read: dispatch storage / compute ---------------------------------

    def read(self, node: NodeId, t: Time) -> Any:
        """Return the value at ``(node, t)``.

        * **Input nodes**: return the stored value, or
          ``group.identity()`` if no push has happened at ``t``.
        * **Derived nodes**: cache hit returns memoised value. Otherwise invokes the compute rule with readers bound for
          each progress input (plus a self-reader for Feedback shapes),
          and memoises the result.

        Recursion through Feedback's self-edge bottoms out because
        :class:`compute.Prev` returns identity at the axis floor.
        """
        val = self.storage.read(node, t, default=_CACHE_MISS)
        if val is not _CACHE_MISS:
            return val

        rule = self.circuit.progress_rules[node]

        if isinstance(rule, prg.Input):
            return self.storage.read(node, t, default=self.group.identity())

        compute_rule = self.circuit.compute_rules[node]
        reads = tuple(partial(self.read, b) for b in rule.inputs)
        if isinstance(rule, prg.Feedback):
            reads = (*reads, partial(self.read, node))

        value = compute_rule.compute(t, reads, self.ctx)
        self.storage.write(node, t, value)
        return value

    # ---- Convenience: read at the latest frontier tick --------------------

    def latest(self, node: NodeId) -> Any:
        """Return the node's value at the **lattice-maximal element
        of its settled frontier**, with any ``OMEGA`` coordinates
        resolved to ``0``. "The freshest known tick". Arity-agnostic.

        A coordinate in ``max(frontier)`` is ``OMEGA`` when the axis
        is trivially settled (upstream ``AxisIntroduction``) and
        nothing concrete is wired in via ``Meet`` to constrain it.
        ``0`` is the unique concrete tick guaranteed to be in the
        down-set of any ω-containing element, and on an
        ``AxisIntroduction``-lifted axis the output at coord 0
        equals the unlifted input. Which is exactly what "freshest"
        means on an axis that does not iterate.

        For pipelines that *do* iterate on an axis (Datalog,
        reachability, fixpoints), the state input's concrete
        antichain meets the ω-fill out of the node's frontier via
        :class:`progress.Meet`, so the ω-substitution is a no-op
        there: ``max(frontier)`` is already concrete on the
        iteration axis.

        Returns ``group.identity()`` when the frontier is empty (no
        pushes yet). For nodes whose frontier has multiple
        incomparable maximal elements, picks the lexicographic
        max."""
        fr = self.frontiers()[node]
        if not fr.elements:
            return self.group.identity()
        max_elem = max(fr.elements)
        concrete = cast(T, tuple(0 if c == OMEGA else c for c in max_elem))
        return self.read(node, concrete)

    def saturate_inner(
        self,
        body: NodeId,
        outer_tick: int,
        is_empty: Callable[[Any], bool],
        *,
        min_inner: int = -1,
    ) -> Iterator[tuple[int, Any]]:
        """Drive the inner fixpoint of a 2-D body at ``outer_tick``.

        Reads ``body`` at ``(outer_tick, 0), (outer_tick, 1), …`` and
        yields ``(k, diff)`` at each step. Stops (returns) the first
        iteration where both ``is_empty(diff)`` holds **and**
        ``k > min_inner``. Callers run state pushes inside the loop
        body using the yielded ``k``.

        ``min_inner`` matters in multi-outer-tick streaming: at
        ``outer_tick > 0``, state from prior outer ticks reaches the
        body's compute via ``Iᵒ`` only at inner depths ≥ those at
        which the prior state was last pushed. Setting ``min_inner =
        max_inner_tick_at_any_outer < outer_tick`` ensures we do not
        converge before that state has had a chance to fully
        propagate. Single-outer callers leave it at the default
        ``-1`` (k > -1 is trivially true)."""
        k = 0
        while True:
            diff = self.read(body, (outer_tick, k))
            if is_empty(diff) and k > min_inner:
                return
            yield k, diff
            k += 1

    # ---- Frontiers / GC ---------------------------------------------------

    def frontiers(self) -> list[Antichain[T]]:
        """Per-node current settled frontier (one entry per node).
        Recomputed each call. For steady-state checks, cache the
        result."""
        return propagate_forward(self.circuit.progress_rules)

    def compact(self, cursors: dict[NodeId, Antichain[T]] | None = None) -> int:
        """GC pass: seed :func:`propagate_backward` with ``cursors``,
        meet each node's backward contributions, evict every storage
        entry dominated by the resulting per-node dead antichain.
        Returns the total number of entries evicted.

        **No-arg form** (``cursors is None``, the recommended default):
        each node's cursor is its current settled frontier (from
        :func:`propagate_forward`). This is the *monotone-forward*
        contract. The caller is reading each cell at-or-after the
        most-recent compaction frontier and will not go back.

        Why the walk-based meet rather than a direct frontier-as-dead
        rule. A "trust the cache" rule. Dead = frontier for stateless
        nodes, dead = ``retreat_omega_fill`` for Feedback. Looks
        tighter on paper because of the strict-feedback theorem, but
        breaks 2-D streaming patterns. Concretely, in streaming reach /
        Datalog, the body's saturate-inner loop may reach a deeper
        ``k`` at batch ``N+1`` than at batch ``N``. Computing
        ``int_a_oi(N+1, k > batch_N_depth)`` then triggers a recompute
        of ``int_a_oi(N, k)``, whose chain walks back through
        ``int_a_i(N, batch_N_depth-1)``. The frontier cell of an
        axis=1 Feedback at the prior outer. An aggressive axis=1
        ``retreat_omega_fill`` evicts that cell across batches and the
        recompute returns identity. The conservative walk-based meet
        retains the boundary slices those recomputes need.

        For patterns where ``saturate_inner`` is never used (kafi-style
        outer-only streaming, or any pipeline where the inner axis
        stays at zero), the walk and a direct frontier-as-dead rule
        converge. But the walk stays safe under the wider class of
        callers, so it is the default.

        **Explicit cursors**: pass when you need to be more
        conservative than the frontier. E.g. you have pushed values you
        have not yet read and want to hold compaction back. Caller is
        responsible for only reading at-or-after the supplied cursors
        after this call. Reads at evicted timestamps return
        ``group.identity()``."""
        if cursors is None:
            fr = self.frontiers()
            cursors = {nid: fr[nid] for nid in range(len(fr))}
        deads = propagate_backward(self.circuit.progress_rules, cursors)
        total = 0
        for node_id, dead in deads.items():
            total += self.storage.evict_dominated(node_id, cast(Antichain[Time], dead))
        return total
