"""Operator. DBSP operators as the single layer that pairs progress
and compute.

Each operator implements :class:`Operator` by connect\1 itself
into a :class:`Circuit`: it appends one or more nodes whose progress
shape + compute shape together describe the operator's frontier
algebra and value algebra.

The operators here are the user-facing API. The two algebra layers
(:mod:`pydbsp.progress`, :mod:`pydbsp.compute`) provide the
primitives. This module wires them.

**Primitive operators** connect as a single node:

* :class:`Input`                    → ``progress.Input`` + ``compute.Get``
* :class:`Lift1`                    → ``progress.Identity`` + ``compute.Map``
* :class:`Lift2`                    → ``progress.Meet`` + ``compute.ZipWith``
* :class:`CoreDelay`                → ``progress.AxisShift`` + ``compute.Prev``
* :class:`CoreIntegrate`            → ``progress.Feedback`` + ``compute.Sum``
* :class:`CoreTimeAxisIntroduction` → ``progress.AxisIntroduction`` + ``compute.Constant``

**Composite operators** connect as multiple nodes:

* :class:`CoreDifferentiate` = ``Lift2(add)(input, Lift1(neg) ∘ Delay)``. 3 nodes.

**DBSP-convention overloads** fix the axis on the ``Core*`` operators
to match canonical outer / inner naming:

* :class:`Delay` / :class:`LiftDelay` . ``z⁻¹`` on outer / inner.
* :class:`Integrate` / :class:`LiftIntegrate` . ``I`` on outer / inner.
* :class:`Differentiate` / :class:`LiftDifferentiate` . ``D`` on outer / inner.
* :class:`StreamIntroduction` / :class:`LiftStreamIntroduction` .
  ``δ₀`` 0-D → 1-D / ↑δ₀ 1-D → 2-D.
* :class:`StreamElimination` / :class:`LiftStreamElimination` .
  ``Σ₀`` 1-D → 0-D / ↑Σ₀ 2-D → 1-D.

For non-canonical axes, use the ``Core*`` versions explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydbsp.core import AbelianGroupOperation, Antichain

from pydbsp import compute
from pydbsp import progress
from pydbsp.circuit import Circuit
from pydbsp.progress import NodeId, Time


@runtime_checkable
class Operator(Protocol):
    """A DBSP operator. Each concrete implementation knows how to
    wire itself into a :class:`Circuit` by adding one or more
    (progress, compute) nodes via :meth:`Circuit.add`."""

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        """Add this operator's node(s) to ``circuit``. Returns the
        ``NodeId`` of the operator's output (the final node added)."""
        ...


# ---- Primitive operators ---------------------------------------------------


@dataclass(frozen=True)
class Input[V](Operator):
    """A source. ``inputs`` must be empty. The progress side carries
    the initial settled frontier. The compute side is :class:`compute.Get`
    (storage-routed by the evaluator)."""

    frontier: Antichain[Time]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        if inputs:
            raise ValueError(f"Input takes no inputs; got {len(inputs)}")
        return circuit.add(
            progress.Input[Time](frontier=self.frontier),
            compute.Get[V](),
        )


@dataclass(frozen=True)
class Lift1[V_in, V_out](Operator):
    """``Lift1(s, f)``. Pointwise unary. ``f: V_in → V_out`` may be
    non-closed. Single node: ``progress.Identity`` + ``compute.Map(f)``."""

    f: Callable[[V_in], V_out]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        (s,) = inputs
        return circuit.add(
            progress.Identity[Time](input=s),
            compute.Map[V_in, V_out](f=self.f),
        )


@dataclass(frozen=True)
class Lift2[V_l, V_r, V_out](Operator):
    """``Lift2(l, r, op)``. Pointwise binary. ``op: V_l × V_r →
    V_out`` may be non-closed. Single node: ``progress.Meet`` +
    ``compute.ZipWith(op)``."""

    op: Callable[[V_l, V_r], V_out]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        if len(inputs) != 2:
            raise ValueError(f"Lift2 takes two inputs; got {len(inputs)}")
        return circuit.add(
            progress.Meet[Time](_inputs=inputs),
            compute.ZipWith[V_l, V_r, V_out](op=self.op),
        )


@dataclass(frozen=True)
class CoreDelay[V](Operator):
    """``Delay(s, axis=i)``. Strict shift on ``axis``. Single node:
    ``progress.AxisShift`` + ``compute.Prev`` (group identity at the
    axis bottom)."""

    axis: int
    group: AbelianGroupOperation[V]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        (s,) = inputs
        return circuit.add(
            progress.AxisShift[Time](input=s, axis=self.axis),
            compute.Prev[V](axis=self.axis, group=self.group),
        )


@dataclass(frozen=True)
class CoreIntegrate[V](Operator):
    """``Integrate(s, axis=i)``. Running sum on ``axis``. Single node
    pairing a strict-feedback progress shape with the running-sum
    compute: ``progress.Feedback`` + ``compute.Sum``. The ``self_id``
    on the feedback shape is the node's own arena index, set via
    :meth:`Circuit.next_id`."""

    axis: int
    group: AbelianGroupOperation[V]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        (s,) = inputs
        own_id = circuit.next_id()
        return circuit.add(
            progress.Feedback[Time](input=s, self_id=own_id, axis=self.axis),
            compute.Sum[V](axis=self.axis, group=self.group),
        )


@dataclass(frozen=True)
class CoreTimeAxisIntroduction[V](Operator):
    """``TimeAxisIntroduction(s, axis=k)``. Δ_k impulse. Lifts the
    input to a higher-arity lattice with the value placed at ``t_k =
    0`` and identity elsewhere. Single node:
    ``progress.AxisIntroduction`` + ``compute.Constant``."""

    axis: int
    group: AbelianGroupOperation[V]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        (s,) = inputs
        return circuit.add(
            progress.AxisIntroduction[Time](input=s, axis=self.axis),
            compute.Constant[V](axis=self.axis, group=self.group),
        )


# ---- Composite operators ---------------------------------------------------


@dataclass(frozen=True)
class CoreDifferentiate[V](Operator):
    """``Differentiate(s, axis=i)(t) = s(t) - s(predecessor(t, i))``. Not a primitive: composes as ``Lift2(group.add)`` over the input
    read and ``Lift1(group.neg) ∘ Delay(input, axis)``. Connects
    as **three** nodes in the circuit:

    1. ``Delay(input, axis)``
    2. ``Lift1(group.neg)`` over (1)
    3. ``Lift2(group.add)`` over (input, (2))

    Returns the ``Lift2``'s ``NodeId``. The operator's output."""

    axis: int
    group: AbelianGroupOperation[V]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        (s,) = inputs
        delayed = CoreDelay[V](axis=self.axis, group=self.group).connect(circuit, (s,))
        neg_delayed = Lift1[V, V](f=self.group.neg).connect(circuit, (delayed,))
        return Lift2[V, V, V](op=self.group.add).connect(circuit, (s, neg_delayed))


# ---- DBSP-convention overloads ---------------------------------------------
#
# These thin wrappers fix the axis of the corresponding ``Core*`` operator
# to match DBSP's outer/inner convention. ``F`` operates on the *outer*
# axis (axis 0); ``LiftF`` operates on the *inner* axis (axis 1) —
# corresponding to DBSP's ↑F (lift F to a stream-of-streams).
#
# Time-axis introduction / elimination get the ``Stream`` naming since
# their canonical 0-D → 1-D form is "wrap into a stream", and the
# ``Lift`` form is "wrap a stream into a stream-of-streams".


@dataclass(frozen=True)
class Delay[V](Operator):
    """``z⁻¹`` on the **outer** axis. Shorthand for
    ``CoreDelay(axis=0)``."""

    group: AbelianGroupOperation[V]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        return CoreDelay[V](axis=0, group=self.group).connect(circuit, inputs)


@dataclass(frozen=True)
class LiftDelay[V](Operator):
    """``↑z⁻¹``. Delay on the **inner** axis (axis 1)."""

    group: AbelianGroupOperation[V]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        return CoreDelay[V](axis=1, group=self.group).connect(circuit, inputs)


@dataclass(frozen=True)
class Integrate[V](Operator):
    """``I`` on the **outer** axis. Shorthand for
    ``CoreIntegrate(axis=0)``."""

    group: AbelianGroupOperation[V]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        return CoreIntegrate[V](axis=0, group=self.group).connect(circuit, inputs)


@dataclass(frozen=True)
class LiftIntegrate[V](Operator):
    """``↑I``. Integrate on the **inner** axis (axis 1)."""

    group: AbelianGroupOperation[V]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        return CoreIntegrate[V](axis=1, group=self.group).connect(circuit, inputs)


@dataclass(frozen=True)
class Differentiate[V](Operator):
    """``D`` on the **outer** axis. Shorthand for
    ``CoreDifferentiate(axis=0)``."""

    group: AbelianGroupOperation[V]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        return CoreDifferentiate[V](axis=0, group=self.group).connect(circuit, inputs)


@dataclass(frozen=True)
class LiftDifferentiate[V](Operator):
    """``↑D``. Differentiate on the **inner** axis (axis 1)."""

    group: AbelianGroupOperation[V]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        return CoreDifferentiate[V](axis=1, group=self.group).connect(circuit, inputs)


@dataclass(frozen=True)
class StreamIntroduction[V](Operator):
    """``δ₀``. Wraps a 0-dimensional value into a 1-dimensional stream
    by introducing axis 0. Shorthand for
    ``CoreTimeAxisIntroduction(axis=0)``."""

    group: AbelianGroupOperation[V]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        return CoreTimeAxisIntroduction[V](axis=0, group=self.group).connect(circuit, inputs)


@dataclass(frozen=True)
class LiftStreamIntroduction[V](Operator):
    """``↑δ₀``. Wraps a 1-D stream into a 2-D stream-of-streams by
    introducing the inner axis (axis 1). Shorthand for
    ``CoreTimeAxisIntroduction(axis=1)``."""

    group: AbelianGroupOperation[V]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        return CoreTimeAxisIntroduction[V](axis=1, group=self.group).connect(circuit, inputs)
