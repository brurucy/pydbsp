"""Tests for ``pydbsp.compute`` — the value-algebra of each compute
shape.

Per shape we test:

1. Basic positive behaviour (canonical input → canonical output).
2. Edge cases (axis bottom, no-push fallbacks, identity returns).
3. For ``Apply`` variants — heterogeneous input/output types.
4. Compositional behaviour where applicable (Differentiate ∘ Integrate
   recovers the input, etc.).
"""

from __future__ import annotations

from collections.abc import Callable

from pydbsp.compute import (
    Apply,
    ComputeCtx,
    Constant,
    Diff,
    Foldl,
    Get,
    Map,
    Prev,
    Sum,
    ZipWith,
)
from pydbsp.progress import Time
from pydbsp.core import AbelianGroupOperation, dbsp_time


# ---- Test fixtures ---------------------------------------------------------


class IntGroup(AbelianGroupOperation[int]):
    """Integers under addition — the standard testing group."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def neg(self, a: int) -> int:
        return -a

    def identity(self) -> int:
        return 0


INT_GROUP = IntGroup()
CTX_1D = ComputeCtx(lattice=dbsp_time(1))
CTX_2D = ComputeCtx(lattice=dbsp_time(2))


def reader_from(storage: dict[Time, int]) -> Callable[[Time], int]:
    """Build a reader that returns ``storage[t]`` or 0 if absent."""
    return lambda t: storage.get(t, 0)


# ============================================================================
# Get
# ============================================================================


def test_input_compute_returns_the_single_read() -> None:
    storage: dict[Time, int] = {(5,): 42, (7,): 99}
    read = reader_from(storage)
    op = Get[int]()
    assert op.compute((5,), (read,), CTX_1D) == 42
    assert op.compute((7,), (read,), CTX_1D) == 99
    assert op.compute((3,), (read,), CTX_1D) == 0  # absent → 0 (storage default)


# ============================================================================
# Apply / Map / ZipWith
# ============================================================================


def test_lift1_applies_unary_function() -> None:
    op = Map[int, int](f=lambda x: x * 2)
    read = reader_from({(0,): 21})
    assert op.compute((0,), (read,), CTX_1D) == 42


def test_lift1_supports_heterogeneous_types() -> None:
    """Map from int to str — non-closed."""
    op = Map[int, str](f=lambda x: f"value={x}")
    read = reader_from({(0,): 7})
    assert op.compute((0,), (read,), CTX_1D) == "value=7"


def test_lift2_applies_binary_operator() -> None:
    op = ZipWith[int, int, int](op=lambda a, b: a + b)
    read_l = reader_from({(0,): 10})
    read_r = reader_from({(0,): 32})
    assert op.compute((0,), (read_l, read_r), CTX_1D) == 42


def test_lift2_supports_heterogeneous_types() -> None:
    """ZipWith from (int, str) → str — non-closed in three different
    types."""
    op = ZipWith[int, str, str](op=lambda i, s: f"{s}-{i}")
    read_i: Callable[[Time], int] = lambda t: 7
    read_s: Callable[[Time], str] = lambda t: "x"
    assert op.compute((0,), (read_i, read_s), CTX_1D) == "x-7"


def test_lift_variadic_combines_n_readers() -> None:
    op = Apply[int](f=lambda *xs: sum(xs))
    readers: tuple[Callable[[Time], int], ...] = (
        lambda t: 10,
        lambda t: 20,
        lambda t: 12,
    )
    assert op.compute((0,), readers, CTX_1D) == 42


def test_lift_zero_readers_calls_nullary_function() -> None:
    op = Apply[int](f=lambda: 7)
    assert op.compute((0,), (), CTX_1D) == 7


# ============================================================================
# Prev
# ============================================================================


def test_delay_reads_predecessor_on_axis() -> None:
    op = Prev[int](axis=0, group=INT_GROUP)
    storage: dict[Time, int] = {(5,): 100, (6,): 200}
    read = reader_from(storage)
    # At t = (6,), predecessor on axis 0 is (5,) → read returns 100.
    assert op.compute((6,), (read,), CTX_1D) == 100


def test_delay_returns_identity_at_axis_bottom() -> None:
    op = Prev[int](axis=0, group=INT_GROUP)
    read = reader_from({(0,): 999})  # would be returned, but bottom skips
    assert op.compute((0,), (read,), CTX_1D) == 0  # group identity


def test_delay_2d_only_shifts_chosen_axis() -> None:
    """Shifting axis 1 leaves axis 0 untouched."""
    op = Prev[int](axis=1, group=INT_GROUP)
    storage: dict[Time, int] = {(3, 5): 42}
    read = reader_from(storage)
    # At t = (3, 6), predecessor on axis 1 is (3, 5).
    assert op.compute((3, 6), (read,), CTX_2D) == 42


# ============================================================================
# Constant
# ============================================================================


def test_axis_introduction_reads_input_at_t_axis_zero() -> None:
    op = Constant[int](axis=1, group=INT_GROUP)
    # Input is 1D; output is 2D.
    storage: dict[Time, int] = {(5,): 42}
    read = reader_from(storage)
    # At t = (5, 0) we're on the impulse — read input at (5,).
    assert op.compute((5, 0), (read,), CTX_2D) == 42


def test_axis_introduction_returns_identity_off_impulse() -> None:
    op = Constant[int](axis=1, group=INT_GROUP)
    read = reader_from({(5,): 999})  # would be returned, but off-impulse skips
    # At t = (5, 3), axis-1 != 0 → identity.
    assert op.compute((5, 3), (read,), CTX_2D) == 0


def test_axis_introduction_works_at_any_axis_position() -> None:
    """Introduce axis at position 0 (the new outer axis)."""
    op = Constant[int](axis=0, group=INT_GROUP)
    storage: dict[Time, int] = {(5,): 42}
    read = reader_from(storage)
    # At t = (0, 5), axis 0 is 0 → read input at (5,).
    assert op.compute((0, 5), (read,), CTX_2D) == 42
    # At t = (3, 5), axis 0 != 0 → identity.
    assert op.compute((3, 5), (read,), CTX_2D) == 0


# ============================================================================
# Foldl
# ============================================================================


def test_causal_aggregate_running_sum() -> None:
    """``self(t) = input(t) + self(t-1)``; at axis bottom self collapses
    to identity, so self(0) = input(0)."""
    op = Foldl[int](axis=0, op=lambda a, b: a + b, group=INT_GROUP)
    input_values: dict[Time, int] = {(0,): 1, (1,): 2, (2,): 3, (3,): 4}
    read_input = reader_from(input_values)
    self_values: dict[Time, int] = {}
    read_self = reader_from(self_values)

    for tick in range(4):
        t: Time = (tick,)
        self_values[t] = op.compute(t, (read_input, read_self), CTX_1D)

    assert self_values == {(0,): 1, (1,): 3, (2,): 6, (3,): 10}


def test_causal_aggregate_with_non_add_op() -> None:
    """Op can be anything closed in V — try max over the axis."""
    op = Foldl[int](axis=0, op=max, group=INT_GROUP)
    input_values: dict[Time, int] = {(0,): 5, (1,): 2, (2,): 8, (3,): 3}
    read_input = reader_from(input_values)
    self_values: dict[Time, int] = {}
    read_self = reader_from(self_values)

    for tick in range(4):
        t: Time = (tick,)
        self_values[t] = op.compute(t, (read_input, read_self), CTX_1D)

    # Running max: 5, 5, 8, 8.
    assert self_values == {(0,): 5, (1,): 5, (2,): 8, (3,): 8}


# ============================================================================
# Sum
# ============================================================================


def _running_sum_via(
    op: Sum[int],
    input_values: dict[Time, int],
    ticks: int,
) -> dict[Time, int]:
    """Helper: step a causal-aggregate-shaped op through ``ticks``."""
    read_input = reader_from(input_values)
    self_values: dict[Time, int] = {}
    read_self = reader_from(self_values)
    for tick in range(ticks):
        t: Time = (tick,)
        self_values[t] = op.compute(t, (read_input, read_self), CTX_1D)
    return self_values


def test_integrate_compute_running_sum() -> None:
    op = Sum[int](axis=0, group=INT_GROUP)
    out = _running_sum_via(op, {(0,): 1, (1,): 2, (2,): 3}, ticks=3)
    assert out == {(0,): 1, (1,): 3, (2,): 6}


# ============================================================================
# Diff
# ============================================================================


def test_differentiate_compute_is_input_minus_delay() -> None:
    op = Diff[int](axis=0, group=INT_GROUP)
    # input(0..3) = 1, 3, 6, 10 — running sums
    read = reader_from({(0,): 1, (1,): 3, (2,): 6, (3,): 10})
    # Differences: 1-0, 3-1, 6-3, 10-6.
    assert op.compute((0,), (read,), CTX_1D) == 1
    assert op.compute((1,), (read,), CTX_1D) == 2
    assert op.compute((2,), (read,), CTX_1D) == 3
    assert op.compute((3,), (read,), CTX_1D) == 4


def test_differentiate_after_integrate_recovers_input() -> None:
    """``∂ ∘ ∫ = id``: differentiating a running sum recovers the
    original deltas."""
    integrate = Sum[int](axis=0, group=INT_GROUP)
    diff = Diff[int](axis=0, group=INT_GROUP)

    inputs: dict[Time, int] = {(0,): 1, (1,): 2, (2,): 3, (3,): 4}
    integrated = _running_sum_via(integrate, inputs, ticks=4)
    read_integrated = reader_from(integrated)

    for tick in range(4):
        t: Time = (tick,)
        assert diff.compute(t, (read_integrated,), CTX_1D) == inputs[t]


# ============================================================================
# Cross-shape composition (manual wiring)
# ============================================================================


def test_lift1_neg_after_input_pipeline() -> None:
    """Compose: read an input, then Map(neg) on it."""
    storage: dict[Time, int] = {(0,): 5, (1,): 10}
    read = reader_from(storage)
    op = Map[int, int](f=INT_GROUP.neg)
    assert op.compute((0,), (read,), CTX_1D) == -5
    assert op.compute((1,), (read,), CTX_1D) == -10


def test_lift2_add_combines_two_inputs() -> None:
    """Two independent inputs feeding a ZipWith(add)."""
    s_l: dict[Time, int] = {(0,): 5, (1,): 10}
    s_r: dict[Time, int] = {(0,): 7, (1,): 3}
    op = ZipWith[int, int, int](op=INT_GROUP.add)
    assert op.compute((0,), (reader_from(s_l), reader_from(s_r)), CTX_1D) == 12
    assert op.compute((1,), (reader_from(s_l), reader_from(s_r)), CTX_1D) == 13
