"""Circuit. The builder that pairs progress + compute per node.

A :class:`Circuit` holds two parallel lists indexed by ``NodeId``:

* ``progress_rules``. One :class:`ProgressRule` per node (frontier
  algebra).
* ``compute_rules``. One :class:`ComputeRule` per node (value
  algebra).

The two layers share their ``NodeId``s but live in separate modules.
This class is the only place where they are attached, via
:meth:`Circuit.add`.

User code does **not** wire raw progress/compute shapes through this
class. It goes through :mod:`pydbsp.operator`, which exposes DBSP
operators that materialise themselves into a circuit by calling
``add`` with the appropriate pair.

``add`` validates input references on every call: progress inputs
must be lower indices, and ``Feedback.self_id`` must equal the node's
own arena index. Catching wiring errors at the call site beats
deferring them to a separate validation pass.

Propagation operates on a plain :class:`Sequence` of progress rules,
so ``circuit.progress_rules`` feeds straight into
``propagate_forward`` / ``propagate_backward``. No snapshot or
wrapper type needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydbsp.progress import (
    Feedback,
    NodeId,
    ProgressRule,
    Time,
)


@dataclass
class Circuit[T: Time]:
    """Arena-style builder: parallel lists of progress + compute
    rules, one entry per node. Use :mod:`pydbsp.operator`'s DBSP
    operators (``Input``, ``Lift1``, ``Integrate``, ...) to populate
    a Circuit. This class's :meth:`add` is the primitive each
    operator's ``materialize`` calls."""

    progress_rules: list[ProgressRule[T]] = field(default_factory=list)
    # `Any` rather than ``list[ComputeRule]``: concrete compute shapes
    # use fixed-length tuple types for ``reads`` (Map → 1-tuple, ZipWith
    # → 2-tuple, etc.), which contravariantly fail the variadic ``reads``
    # in ``ComputeRule``'s Protocol. The list stores them uniformly;
    # the evaluator (separate layer) dispatches on shape type.
    compute_rules: list[Any] = field(default_factory=list)

    def add(
        self,
        progress: ProgressRule[T],
        compute: Any,
    ) -> NodeId:
        """Append a (progress, compute) pair as one new node. Validates
        progress references against arena-style invariants:

        * Every ``progress.inputs`` element must be a strictly lower
          index (acyclic, no forward references).
        * If ``progress`` is :class:`Feedback`, its ``self_id`` must
          equal the new node's own arena index.

        Returns the new node's :data:`NodeId`.
        """
        i: NodeId = len(self.progress_rules)
        for b in progress.inputs:
            if not (0 <= b < i):
                raise ValueError(
                    f"node {i} ({type(progress).__name__}) references input "
                    f"{b}; inputs must be strictly lower indices in arena "
                    f"order"
                )
        if isinstance(progress, Feedback) and progress.self_id != i:
            raise ValueError(
                f"Feedback at index {i} has self_id={progress.self_id}; must equal own index in arena style"
            )
        self.progress_rules.append(progress)
        self.compute_rules.append(compute)
        return i

    def next_id(self) -> NodeId:
        """Peek the next ``NodeId`` without appending. Used by
        ``Feedback``-shaped operators (``Integrate``) to know their
        own arena index at progress-construction time, so they can
        set ``Feedback.self_id``."""
        return len(self.progress_rules)

    # ---- Inspection helpers ------------------------------------------------

    def consumers(self) -> list[list[NodeId]]:
        """Parallel list: ``consumers()[i]`` is every node with ``i``
        as a direct progress input. Self-edges not included."""
        out: list[list[NodeId]] = [[] for _ in self.progress_rules]
        for i, p in enumerate(self.progress_rules):
            for b in p.inputs:
                out[b].append(i)
        return out

    def roots(self) -> list[NodeId]:
        """Indices that no one consumes. The DAG's outputs."""
        cons = self.consumers()
        return [i for i, c in enumerate(cons) if not c]

    def input_nodes(self) -> list[NodeId]:
        """Indices of source nodes (no progress inputs)."""
        return [i for i, p in enumerate(self.progress_rules) if not p.inputs]
