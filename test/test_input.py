"""Tests for ``Input`` — the externally-driven source stream."""

import pytest

from pydbsp.core import Time1D, Time2D
from pydbsp.history import History
from pydbsp.stream import Input, Lift2
from pydbsp.stream.operators.linear import Integrate

from test.test_algebra import IntegerAddition


def test_input_starts_with_empty_frontier() -> None:
    inp: Input[int, tuple[int]] = Input(IntegerAddition(), Time1D)
    assert inp.settled_frontier.elements == []
    assert inp.settled_frontier.is_universal is False


def test_input_push_extends_frontier() -> None:
    inp: Input[int, tuple[int]] = Input(IntegerAddition(), Time1D)
    inp.push((0,), 42)
    assert inp.settled_frontier.elements == [(0,)]
    assert inp.at((0,)) == 42
    # Unsettled read returns identity (the frontier check is the observer's job).
    assert inp.at((1,)) == 0


def test_input_push_changes_antichain_identity() -> None:
    """The source-tracking cache depends on this: every push installs
    a fresh ``Antichain`` object. Identity is the progress signal."""
    inp: Input[int, tuple[int]] = Input(IntegerAddition(), Time1D)
    before = inp.settled_frontier
    inp.push((0,), 1)
    after = inp.settled_frontier
    assert before is not after


def test_input_rejects_redundant_push() -> None:
    inp: Input[int, tuple[int]] = Input(IntegerAddition(), Time1D)
    inp.push((0,), 1)
    with pytest.raises(ValueError):
        inp.push((0,), 2)


def test_history_observes_input_over_time() -> None:
    inp: Input[int, tuple[int]] = Input(IntegerAddition(), Time1D)
    # Trivial circuit: just the input.
    h: History[int, tuple[int]] = History(inp)

    # No pushes yet — can't step.
    assert h.try_step() is False

    inp.push((0,), 5)
    assert h.try_step() is True
    assert h.at((0,)) == 5

    inp.push((1,), 7)
    assert h.try_step() is True
    assert h.at((1,)) == 7


def test_history_over_integrated_input() -> None:
    """Build a small circuit over an Input and drive it. Integrate
    accumulates pushed values along the single axis."""
    inp: Input[int, tuple[int]] = Input(IntegerAddition(), Time1D)
    integrated = Integrate(inp, Time1D)
    h: History[int, tuple[int]] = History(integrated)

    inp.push((0,), 3)
    h.try_step()
    assert h.at((0,)) == 3

    inp.push((1,), 4)
    h.try_step()
    assert h.at((1,)) == 7  # 3 + 4

    inp.push((2,), -2)
    h.try_step()
    assert h.at((2,)) == 5  # 7 + (-2)


def test_history_over_two_inputs() -> None:
    """Two independent Inputs — output's frontier is the meet of both,
    so stepping requires both to have advanced."""
    a: Input[int, tuple[int]] = Input(IntegerAddition(), Time1D)
    b: Input[int, tuple[int]] = Input(IntegerAddition(), Time1D)
    summed = Lift2(a, b, lambda x, y: x + y, IntegerAddition())
    h: History[int, tuple[int]] = History(summed)

    # Only a has been pushed — b's frontier is empty → meet is empty → can't step.
    a.push((0,), 10)
    assert h.try_step() is False

    # Push b too; now meet covers (0,).
    b.push((0,), 32)
    assert h.try_step() is True
    assert h.at((0,)) == 42


def test_2d_input() -> None:
    inp: Input[int, tuple[int, int]] = Input(IntegerAddition(), Time2D)
    inp.push((0, 0), 1)
    inp.push((0, 1), 2)
    inp.push((1, 0), 3)

    assert inp.at((0, 0)) == 1
    assert inp.at((0, 1)) == 2
    assert inp.at((1, 0)) == 3
    # Frontier contains max antichain: (0, 1) and (1, 0) are
    # incomparable; (0, 0) is dominated by both.
    assert set(inp.settled_frontier.elements) == {(0, 1), (1, 0)}
