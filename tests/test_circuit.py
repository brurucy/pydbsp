"""Tests for ``pydbsp.circuit`` — the (progress, compute) builder.

Circuit's surface is small: ``add``, ``next_id``, and the three
inspection helpers. We exercise each plus the add-time validation."""

from __future__ import annotations

import pytest

from pydbsp.circuit import Circuit
from pydbsp.compute import Get, Map, Sum
from pydbsp.progress import (
    AxisShift,
    Feedback,
    Identity,
    Input,
    Meet,
    Time,
)
from pydbsp.core import AbelianGroupOperation, Antichain, Time1D


class IntGroup(AbelianGroupOperation[int]):
    def add(self, a: int, b: int) -> int:
        return a + b

    def neg(self, a: int) -> int:
        return -a

    def identity(self) -> int:
        return 0


INT_GROUP = IntGroup()


# ---- add: appends and returns sequential ids -------------------------------


def test_add_appends_and_returns_sequential_ids() -> None:
    b: Circuit[tuple[int]] = Circuit()
    i0 = b.add(Input[tuple[int]](frontier=Antichain(Time1D)), Get[int]())
    i1 = b.add(Identity[tuple[int]](input=i0), Map[int, int](f=lambda x: x))
    i2 = b.add(Identity[tuple[int]](input=i1), Map[int, int](f=lambda x: x))
    assert (i0, i1, i2) == (0, 1, 2)
    assert len(b.progress_rules) == 3
    assert len(b.compute_rules) == 3


# ---- next_id: peeks without committing -------------------------------------


def test_next_id_peeks_arena_index_without_appending() -> None:
    b: Circuit[tuple[int]] = Circuit()
    b.add(Input[tuple[int]](frontier=Antichain(Time1D)), Get[int]())
    assert b.next_id() == 1  # would-be index of next add
    assert len(b.progress_rules) == 1  # not yet appended


def test_next_id_enables_feedback_self_id_binding() -> None:
    """Integrate-style: Feedback's self_id must equal its own arena
    index. ``next_id`` is how the operator learns that index up front."""
    b: Circuit[tuple[int]] = Circuit()
    x = b.add(Input[tuple[int]](frontier=Antichain(Time1D)), Get[int]())
    own = b.next_id()
    integrate = b.add(
        Feedback[tuple[int]](input=x, self_id=own, axis=0),
        Sum[int](axis=0, group=INT_GROUP),
    )
    assert integrate == own == 1


# ---- add-time validation ---------------------------------------------------


def test_add_rejects_forward_input_reference() -> None:
    b: Circuit[tuple[int]] = Circuit()
    with pytest.raises(ValueError, match="lower indices"):
        b.add(Identity[tuple[int]](input=5), Map[int, int](f=lambda x: x))


def test_add_rejects_self_reference_as_input() -> None:
    """A node can't reference itself as an input — only Feedback's
    separate self-channel is allowed."""
    b: Circuit[tuple[int]] = Circuit()
    with pytest.raises(ValueError, match="lower indices"):
        b.add(Identity[tuple[int]](input=0), Map[int, int](f=lambda x: x))


def test_add_rejects_feedback_with_wrong_self_id() -> None:
    b: Circuit[tuple[int]] = Circuit()
    x = b.add(Input[tuple[int]](frontier=Antichain(Time1D)), Get[int]())
    with pytest.raises(ValueError, match="self_id"):
        b.add(
            Feedback[tuple[int]](input=x, self_id=99, axis=0),
            Sum[int](axis=0, group=INT_GROUP),
        )


# ---- Inspection helpers ----------------------------------------------------


def test_inspection_helpers() -> None:
    b: Circuit[tuple[int]] = Circuit()
    b.add(Input[tuple[int]](frontier=Antichain(Time1D)), Get[int]())  # 0
    b.add(Identity[tuple[int]](input=0), Map[int, int](f=lambda x: x))  # 1
    b.add(Identity[tuple[int]](input=0), Map[int, int](f=lambda x: x))  # 2
    b.add(Meet[tuple[int]](_inputs=(1, 2)), Map[int, int](f=lambda x: x))  # 3
    assert b.consumers()[0] == [1, 2]
    assert b.consumers()[3] == []
    assert b.roots() == [3]
    assert b.input_nodes() == [0]


# ---- Arity-changing shapes wire in without "SubCircuit" boundaries ----------


def test_axis_shape_inside_one_circuit_at_broad_T() -> None:
    """At ``Circuit[Time]`` the arity-changing shapes
    (AxisIntroduction, AxisElimination) wire as ordinary nodes — no
    boundary; the broad ``T`` accommodates every arity."""
    from typing import cast

    F: Antichain[Time] = cast(Antichain[Time], Antichain(Time1D, [(3,)]))
    b: Circuit[Time] = Circuit()
    b.add(Input[Time](frontier=F), Get[int]())
    b.add(AxisShift[Time](input=0, axis=0), Map[int, int](f=lambda x: x))
    assert len(b.progress_rules) == 2
