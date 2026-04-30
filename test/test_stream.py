"""Tests for the stream primitives: Antichain, δ₀, Delay,
Integrate, Differentiate, Lift1/Lift2, StreamAddition.
Operators step along an axis of a ``DBSPTime`` lattice; every
timestamp is a tuple, even in 1D.
"""

from pydbsp.core import Antichain, DBSPTime, Time1D, Time2D
from pydbsp.stream import Lift1, Lift2
from pydbsp.stream.functions.linear import (
    StreamElimination,
    StreamIntroduction,
)
from pydbsp.stream.operators.linear import (
    Delay,
    Differentiate,
    Integrate,
    StreamAddition,
)

from test.test_algebra import IntegerAddition


# ---- Antichain ----


def test_antichain_normalises_redundant_elements() -> None:
    frontier: Antichain[tuple[int]] = Antichain(Time1D)
    for value in [(1,), (2,), (3,), (2,)]:
        frontier.insert(value)

    assert frontier.elements == [(3,)]  # dominated values drop out


def test_antichain_empty_covers_nothing() -> None:
    frontier: Antichain[tuple[int]] = Antichain(Time1D)
    assert frontier.covers((0,)) is False
    assert frontier.covers((5,)) is False


def test_antichain_universal_covers_everything() -> None:
    universal: Antichain[tuple[int]] = Antichain.universal(Time1D)
    assert universal.is_universal is True
    assert universal.covers((0,)) is True
    assert universal.covers((9999,)) is True


def test_antichain_join_absorbs_universal() -> None:
    a: Antichain[tuple[int]] = Antichain(Time1D, [(3,)])
    universal: Antichain[tuple[int]] = Antichain.universal(Time1D)
    joined = a.join(universal)
    assert joined.is_universal is True


def test_antichain_meet_with_universal_is_identity() -> None:
    a: Antichain[tuple[int]] = Antichain(Time1D, [(5,)])
    universal: Antichain[tuple[int]] = Antichain.universal(Time1D)
    met = a.meet(universal)
    assert met.is_universal is False
    assert met.elements == [(5,)]


def test_product_lattice_meet_join() -> None:
    t2 = Time2D
    assert t2.bottom() == (0, 0)
    assert t2.join((1, 3), (2, 1)) == (2, 3)
    assert t2.meet((1, 3), (2, 1)) == (1, 1)
    assert t2.leq((1, 2), (1, 3)) is True
    assert t2.leq((1, 3), (2, 1)) is False


# ---- DBSPTime.advance_antichain ----


def test_advance_antichain_empty_seeds_at_bottom() -> None:
    empty: Antichain[tuple[int]] = Antichain(Time1D)
    stepped = Time1D.advance_antichain(empty, axis=0)
    assert stepped.elements == [(0,)]


def test_advance_antichain_shifts_along_axis() -> None:
    f: Antichain[tuple[int, int]] = Antichain(Time2D, [(1, 0), (0, 1)])
    stepped = Time2D.advance_antichain(f, axis=0)
    # {(1,0),(0,1)} shifted on axis 0 → {(2,0),(1,1)}; (1,0) absorbed by (1,1)
    assert set(stepped.elements) == {(2, 0), (1, 1)}


def test_advance_antichain_universal_is_fixed_point() -> None:
    universal = Antichain.universal(Time1D)
    assert Time1D.advance_antichain(universal, axis=0).is_universal is True


# ---- δ₀ ----


def test_delta0_value_at_bottom_and_identity_elsewhere() -> None:
    group = IntegerAddition()
    s = StreamIntroduction(42, group, Time1D)

    assert s.at((0,)) == 42
    assert s.at((1,)) == 0
    assert s.at((100,)) == 0
    assert s.settled_frontier.is_universal is True


# ---- Delay ----


def test_delay_shifts_by_one() -> None:
    group = IntegerAddition()
    s = StreamIntroduction(7, group, Time1D)
    delayed = Delay(s, Time1D)

    assert delayed.at((0,)) == 0
    assert delayed.at((1,)) == 7
    assert delayed.at((2,)) == 0


def test_delay_of_universal_is_universal() -> None:
    group = IntegerAddition()
    s = StreamIntroduction(1, group, Time1D)
    assert s.settled_frontier.is_universal is True
    assert Delay(s, Time1D).settled_frontier.is_universal is True


def test_delay_on_2d_shifts_first_axis() -> None:
    group = IntegerAddition()
    s = StreamIntroduction(5, group, Time2D)
    # Delay always shifts the first axis.
    d = Delay(s, Time2D)
    assert d.at((0, 0)) == 0
    assert d.at((1, 0)) == 5
    assert d.at((1, 1)) == 0


# ---- Integrate / Differentiate ----


def test_integrate_accumulates() -> None:
    group = IntegerAddition()
    s = StreamIntroduction(5, group, Time1D)
    # s = [5, 0, 0, 0, …]
    integrated = Integrate(s, Time1D)
    # integrated = [5, 5, 5, 5, …]
    assert integrated.at((0,)) == 5
    assert integrated.at((1,)) == 5
    assert integrated.at((3,)) == 5


def test_differentiate_then_integrate_is_identity() -> None:
    group = IntegerAddition()
    s = StreamIntroduction(9, group, Time1D)
    back = Integrate(Differentiate(s, Time1D), Time1D)

    for t in range(4):
        assert back.at((t,)) == s.at((t,))


# ---- Lift1 / Lift2 ----


def test_lift1_applies_function_pointwise() -> None:
    group = IntegerAddition()
    s = StreamIntroduction(4, group, Time1D)
    doubled = Lift1(s, lambda x: x * 2, group)

    assert doubled.at((0,)) == 8
    assert doubled.at((1,)) == 0


def test_lift2_applies_binary_pointwise() -> None:
    group = IntegerAddition()
    a = StreamIntroduction(3, group, Time1D)
    b = StreamIntroduction(4, group, Time1D)
    summed = Lift2(a, b, lambda x, y: x + y, group)

    assert summed.at((0,)) == 7
    assert summed.at((1,)) == 0


# ---- StreamAddition ----


def test_stream_addition_identity_is_universal_zero() -> None:
    group = IntegerAddition()
    stream_group = StreamAddition(group, Time1D)
    zero = stream_group.identity()

    assert zero.settled_frontier.is_universal is True
    assert zero.at((0,)) == 0
    assert zero.at((17,)) == 0


def test_stream_addition_add_is_pointwise() -> None:
    group = IntegerAddition()
    stream_group = StreamAddition(group, Time1D)
    a = StreamIntroduction(2, group, Time1D)
    b = StreamIntroduction(5, group, Time1D)
    summed = stream_group.add(a, b)

    assert summed.at((0,)) == 7
    assert summed.at((3,)) == 0


# ---- StreamElimination (∫) ----


def test_stream_elim_sums_until_zero() -> None:
    group = IntegerAddition()
    # Build a stream that goes [3, 2, 1, 0, 0, …] via δ₀ + Delay.
    s0 = StreamIntroduction(3, group, Time1D)
    s1 = Delay(StreamIntroduction(2, group, Time1D), Time1D)
    s2 = Delay(Delay(StreamIntroduction(1, group, Time1D), Time1D), Time1D)
    stream_group = StreamAddition(group, Time1D)
    s = stream_group.add(stream_group.add(s0, s1), s2)

    assert StreamElimination(s, Time1D) == 6


def test_dbsp_time_3d_advances_per_axis() -> None:
    t3 = DBSPTime(nestedness=3)
    empty: Antichain[tuple[int, int, int]] = Antichain(t3)
    seeded = t3.advance_antichain(empty, axis=0)
    assert seeded.elements == [(0, 0, 0)]
    stepped1 = t3.advance_antichain(seeded, axis=1)
    assert stepped1.elements == [(0, 1, 0)]
    stepped2 = t3.advance_antichain(stepped1, axis=2)
    assert stepped2.elements == [(0, 1, 1)]
