"""Progress. The antichain algebra of frontier propagation.

The hard part of DBSP, isolated. No value rules, no evaluation
strategies, no operators-as-classes. Just:

* the **antichain lattice** (re-exported from ``pydbsp.core``).
* per-axis **primitives** specifically on ``DBSPTime`` antichains
  (the product-of-``NaturalChain`` structure is what gives axes
  meaning. Generic ``BoundedBelowLattice`` antichains do not have
  them).
* a small library of **progress shapes**. Value objects describing a
  node's algebraic rule in both directions. And
* graph-level **propagation**. Forward (input nodes → roots) and backward
  (roots → input nodes) walks.

Every shape, the DAG, and the propagation functions are generic in
``T: Time`` (the antichain element type). Use ``T = tuple[int, ...]``
for DAGs whose nodes change arity (``AxisIntroduction`` /
``AxisElimination``). A specific ``tuple[int]`` or ``tuple[int, int]``
also works for arity-preserving DAGs.

Forward maps input frontiers to mine; backward maps my cursor to
"what can be freed" on each input. Backward rules use
``retreat_omega_fill`` to claim ``ω`` on non-relevant axes — a tighter
eviction than the per-axis predecessor would produce, justified
case-by-case (strict-feedback theorem for ``Feedback``, etc.).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, cast, runtime_checkable

from pydbsp.core import (
    OMEGA,
    Antichain,
    DBSPTime,
)

# ---- Identifiers -----------------------------------------------------------

NodeId: TypeAlias = int
"""Arena index into a :class:`Progress`'s ``nodes`` list. Stable for
the lifetime of the DAG."""

type Time = tuple[int, ...]
"""Bound for the antichain element type. ``DBSPTime`` antichains hold
``N``-tuples of natural numbers (with ω for top); the algebra below is
parametric in the specific tuple shape."""


# ---- DBSPTime antichain primitives -----------------------------------------
#
# These are *not* generic antichain operations: they require the
# product-of-``NaturalChain`` structure of ``DBSPTime`` to make sense of
# "axis", "successor", and "predecessor". A generic
# ``BoundedBelowLattice`` antichain has no axes. Each primitive reads
# the underlying lattice from ``antichain.lattice`` and asserts it's a
# ``DBSPTime``; passing one of these an antichain over any other
# lattice raises ``TypeError``.


def _dbsp_lattice[T: Time](antichain: Antichain[T]) -> DBSPTime[T]:
    lat = antichain.lattice
    if not isinstance(lat, DBSPTime):
        raise TypeError(f"axis operations require a DBSPTime antichain; got lattice of type {type(lat).__name__}")
    return DBSPTime[T](lat.nestedness)


def shift[T: Time](antichain: Antichain[T], axis: int) -> Antichain[T]:
    """Per-element successor on ``axis``. Empty stays empty. Universal
    is a fixed point."""
    if antichain.is_universal:
        return antichain
    lattice = _dbsp_lattice(antichain)
    if not antichain.elements:
        return Antichain(lattice)
    out: Antichain[T] = Antichain(lattice)
    chain = lattice.factors[axis]
    n = lattice.nestedness
    for e in antichain.elements:
        s = chain.successor(e[axis])
        out.insert(cast(T, tuple(s if i == axis else e[i] for i in range(n))))
    return out


def retreat[T: Time](antichain: Antichain[T], axis: int) -> Antichain[T]:
    """Per-element predecessor on ``axis``. Drops elements at axis
    bottom (no predecessor). Universal is a fixed point."""
    if antichain.is_universal:
        return antichain
    lattice = _dbsp_lattice(antichain)
    out: Antichain[T] = Antichain(lattice)
    chain = lattice.factors[axis]
    n = lattice.nestedness
    for e in antichain.elements:
        p = chain.predecessor(e[axis])
        if p is None:
            continue
        out.insert(cast(T, tuple(p if i == axis else e[i] for i in range(n))))
    return out


def retreat_omega_fill[T: Time](antichain: Antichain[T], axis: int) -> Antichain[T]:
    """Like :func:`retreat`, but fills every non-``axis`` coordinate
    with ``OMEGA``. The tight backward contribution for an edge that
    "reads its source only at the predecessor on ``axis``": future
    reads at the cursor's other-axis values are not constrained by this
    edge, so the dead can claim ``OMEGA`` on those axes.

    Concrete example. ``Delay(axis=0)`` at cursor ``(1, 29)`` reads its
    input at ``(0, 29)``. Future Delay reads at ``(2, k)`` (for any
    ``k``) will read input at ``(1, k)``. Never at outer = 0. So from
    this edge, every input cell with outer ≤ 0 is evictable. Expressible as the antichain ``{(0, OMEGA)}``, not
    ``retreat((1, 29), 0) = {(0, 29)}`` (which leaves
    ``(0, 30..OMEGA)`` cells in storage). The OMEGA-fill makes the
    bound tight.

    At the axis bottom (no predecessor on ``axis``) the result is the
    empty antichain. Meet-with-empty in ``propagate_backward`` then
    correctly preserves the cursor cell on the downstream node
    (annihilator of meet)."""
    if antichain.is_universal:
        return antichain
    lattice = _dbsp_lattice(antichain)
    out: Antichain[T] = Antichain(lattice)
    chain = lattice.factors[axis]
    n = lattice.nestedness
    for e in antichain.elements:
        p = chain.predecessor(e[axis])
        if p is None:
            continue
        out.insert(cast(T, tuple(p if i == axis else OMEGA for i in range(n))))
    return out


def drop_axis[T: Time](antichain: Antichain[T], axis: int) -> Antichain[Time]:
    """Project an antichain onto a lattice with one fewer axis by
    removing ``axis``. The output lattice has arity
    ``antichain.lattice.nestedness - 1``. The output element type
    widens to ``Time`` since the static type cannot track arity."""
    in_lat = _dbsp_lattice(antichain)
    out_lat: DBSPTime[Time] = DBSPTime(nestedness=in_lat.nestedness - 1)
    if antichain.is_universal:
        return Antichain.universal(out_lat)
    out: Antichain[Time] = Antichain(out_lat)
    for e in antichain.elements:
        out.insert(tuple(x for i, x in enumerate(e) if i != axis))
    return out


def insert_axis[T: Time](antichain: Antichain[T], axis: int, value: int | float) -> Antichain[Time]:
    """Lift an antichain onto a lattice with one more axis by inserting
    ``value`` at ``axis``. The output lattice has arity
    ``antichain.lattice.nestedness + 1``. The output element type
    widens to ``Time``. Use ``OMEGA`` (``math.inf``, typed ``float``)
    for "settled everywhere on this axis"."""
    in_lat = _dbsp_lattice(antichain)
    out_lat: DBSPTime[Time] = DBSPTime(nestedness=in_lat.nestedness + 1)
    if antichain.is_universal:
        return Antichain.universal(out_lat)
    out: Antichain[Time] = Antichain(out_lat)
    # OMEGA is typed ``float`` but lives alongside ints in DBSPTime
    # tuples (see ``NaturalChain.top``); the cast follows the same
    # whole-codebase convention.
    int_value = cast(int, value)
    for e in antichain.elements:
        lifted = list(e)
        lifted.insert(axis, int_value)
        out.insert(tuple(lifted))
    return out


# ---- Progress shapes -------------------------------------------------------
#
# Each shape is a value (frozen dataclass) describing one node's
# algebraic rule. Operators in the streams layer attach an instance of
# one of these to themselves; the propagation walks below interpret it.


@runtime_checkable
class ProgressRule[T: Time](Protocol):
    """The algebraic rule for one node in a progress DAG."""

    @property
    def inputs(self) -> tuple[NodeId, ...]:
        """Upstream nodes this one reads from."""
        ...

    def forward(self, input_frontiers: dict[NodeId, Antichain[T]]) -> Antichain[T]:
        """My frontier given my inputs' frontiers."""
        ...

    def backward(self, my_cursor: Antichain[T]) -> dict[NodeId, Antichain[T]]:
        """For each input (and possibly self, for feedback shapes), the
        antichain of cells safe to free given my cursor."""
        ...


@dataclass(frozen=True)
class Input[T: Time]:
    """A source node. No upstream. Carries its settled frontier as
    data. ``forward`` returns it directly. For ``Input`` (whose
    frontier mutates as values are pushed: rebuild the DAG) and
    ``StreamIntroduction`` / δ₀ (universal frontier)."""

    frontier: Antichain[T]

    @property
    def inputs(self) -> tuple[NodeId, ...]:
        return ()

    def forward(self, input_frontiers: dict[NodeId, Antichain[T]]) -> Antichain[T]:
        return self.frontier

    def backward(self, my_cursor: Antichain[T]) -> dict[NodeId, Antichain[T]]:
        return {}


@dataclass(frozen=True)
class Identity[T: Time]:
    """forward = my single input's frontier. Backward = same on that input.

    For ``Lift1``, ``Project``, ``Select``, and (on the input-edge of
    ``Integrate``) the strict-feedback's external input."""

    input: NodeId

    @property
    def inputs(self) -> tuple[NodeId, ...]:
        return (self.input,)

    def forward(self, input_frontiers: dict[NodeId, Antichain[T]]) -> Antichain[T]:
        return input_frontiers[self.input]

    def backward(self, my_cursor: Antichain[T]) -> dict[NodeId, Antichain[T]]:
        return {self.input: my_cursor}


@dataclass(frozen=True)
class Meet[T: Time]:
    """forward = meet of all inputs' frontiers. Backward = passthrough
    to each input. For ``Lift2`` and any N-ary pointwise op."""

    _inputs: tuple[NodeId, ...]

    @property
    def inputs(self) -> tuple[NodeId, ...]:
        return self._inputs

    def forward(self, input_frontiers: dict[NodeId, Antichain[T]]) -> Antichain[T]:
        fs = [input_frontiers[b] for b in self._inputs]
        if not fs:
            raise ValueError("Meet requires at least one input")
        result = fs[0]
        for f in fs[1:]:
            result = result.meet(f)
        return result

    def backward(self, my_cursor: Antichain[T]) -> dict[NodeId, Antichain[T]]:
        return {b: my_cursor for b in self._inputs}


@dataclass(frozen=True)
class AxisShift[T: Time]:
    """forward = shift input on axis (plus ⊥ seed for empty input). Backward = retreat my cursor on the same axis.

    For ``Delay(axis)``. Strict. By ``delay_strict`` in the theory.
    """

    input: NodeId
    axis: int

    @property
    def inputs(self) -> tuple[NodeId, ...]:
        return (self.input,)

    def forward(self, input_frontiers: dict[NodeId, Antichain[T]]) -> Antichain[T]:
        input_f = input_frontiers[self.input]
        shifted = shift(input_f, self.axis)
        if shifted.is_universal:
            return shifted
        if not shifted.elements:
            seeded: Antichain[T] = Antichain(input_f.lattice)
            seeded.insert(input_f.lattice.bottom())
            return seeded
        return shifted

    def backward(self, my_cursor: Antichain[T]) -> dict[NodeId, Antichain[T]]:
        return {self.input: retreat_omega_fill(my_cursor, self.axis)}


@dataclass(frozen=True)
class AxisIntroduction[T: Time]:
    """forward = insert ω at ``axis`` (lift to higher-arity lattice). Backward = drop that axis. For ``TimeAxisIntroduction``.

    Arity changes between input and output. The DAG's ``T`` must be the
    broad ``tuple[int, ...]`` bound to allow heterogeneous arities."""

    input: NodeId
    axis: int

    @property
    def inputs(self) -> tuple[NodeId, ...]:
        return (self.input,)

    def forward(self, input_frontiers: dict[NodeId, Antichain[T]]) -> Antichain[T]:
        return cast(Antichain[T], insert_axis(input_frontiers[self.input], self.axis, OMEGA))

    def backward(self, my_cursor: Antichain[T]) -> dict[NodeId, Antichain[T]]:
        return {self.input: cast(Antichain[T], drop_axis(my_cursor, self.axis))}


@dataclass(frozen=True)
class Feedback[T: Time]:
    """A node defined by a strict feedback equation
    ``self(t) = body(input, delay(self))(t)``.

    At fixpoint, ``self.forward = input.forward`` because the strict
    feedback body's value at ``t`` is fully determined by input at
    ``t`` and self at ``t-1``. Both of which are already settled.

    backward includes both edges: the external input gets ``my_cursor``
    pointwise. ``self_id`` gets the strict-feedback self bound. For
    each cursor element, predecessor on ``self.axis`` with ``OMEGA`` on
    every other axis. That OMEGA-fill expresses the theorem fully:
    "for any inner / cross-axis position, only the cell at the current
    outer is needed". A plain ``retreat(my_cursor, axis)`` (single-axis
    only) leaves cells at the current outer but earlier non-self axes
    in storage, which is sound but conservative. The OMEGA-fill makes
    the bound tight.

    At the axis bottom (no predecessor on ``self.axis``), the bound is
    the empty antichain. Meet-with-empty in ``propagate_backward`` then
    *correctly* annihilates the dead at this Feedback node so the
    cursor-cell itself stays. There is nothing below to evict.

    Used to express ``Integrate(s)`` compositionally as ``fix(λα. s +
    delay α)``."""

    input: NodeId
    self_id: NodeId
    axis: int

    @property
    def inputs(self) -> tuple[NodeId, ...]:
        return (self.input,)

    def forward(self, input_frontiers: dict[NodeId, Antichain[T]]) -> Antichain[T]:
        return input_frontiers[self.input]

    def backward(self, my_cursor: Antichain[T]) -> dict[NodeId, Antichain[T]]:
        return {
            self.input: my_cursor,
            self.self_id: retreat_omega_fill(my_cursor, self.axis),
        }


# ---- Propagation ----------------------------------------------------------


def propagate_forward[T: Time](
    nodes: Sequence[ProgressRule[T]],
) -> list[Antichain[T]]:
    """Compute every node's settled frontier from input nodes to roots.
    ``nodes`` is the arena (e.g. ``circuit.progress_rules``). Returns
    a list parallel to it."""
    frontiers: list[Antichain[T]] = []
    for prog in nodes:
        input_frontiers = {b: frontiers[b] for b in prog.inputs}
        frontiers.append(prog.forward(input_frontiers))
    return frontiers


def propagate_backward[T: Time](
    nodes: Sequence[ProgressRule[T]],
    cursors: dict[NodeId, Antichain[T]],
) -> dict[NodeId, Antichain[T]]:
    """Compute every node's dead antichain from roots to input nodes.

    ``nodes`` is the arena (e.g. ``circuit.progress_rules``). ``cursors``
    seeds the walk. Typically the root(s) the caller is observing,
    but cursors at intermediate nodes work too. Multiple seeds are
    supported. Their contributions meet at fan-in.

    At every node (in reverse arena order):

    * if it has consumers, their backward contributions have already
      been met into ``deads[node]`` by the time we visit it.
    * we then call ``shape.backward(deads[node])`` and meet each
      returned entry into the target's accumulator.

    For ``Feedback`` shapes, one of the returned targets is ``self_id``
    (= the node's own index) carrying ``retreat(F, axis)``. The meet
    refines ``deads[node]`` *after* the call has already produced
    contributions for other targets using the pre-meet ``F``. This
    one-shot semantics . *not* fixpoint iteration. Is correct under
    the strict-feedback theorem: only the most-recent self-slot is
    needed (which is exactly what ``retreat(F)``'s down-set encodes).
    Iterating to fixpoint would converge to ``⊥`` (GC nothing). Conservative but wasteful.

    Returns ``{i: dead_antichain}`` for every reached node. Nodes not
    reachable from any cursor are absent.
    """
    deads: dict[NodeId, Antichain[T]] = dict(cursors)
    for i in range(len(nodes) - 1, -1, -1):
        if i not in deads:
            continue
        prog = nodes[i]
        for target, ach in prog.backward(deads[i]).items():
            if target in deads:
                deads[target] = deads[target].meet(ach)
            else:
                deads[target] = ach
    return deads
