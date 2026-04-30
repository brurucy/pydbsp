"""Tests for ``History`` — the execution driver.

History owns a cursor antichain on a circuit's ``DBSPTime`` lattice
and advances it with ``try_step(axis)``. The circuit itself is just
a ``Stream`` graph; History provides the only notion of "now".
"""

import pytest

from pydbsp.core import Time1D, Time2D
from pydbsp.evaluator import Evaluator
from pydbsp.history import History
from pydbsp.stream.functions.linear import StreamIntroduction
from pydbsp.stream.operators.linear import Delay, StreamAddition

from test.test_algebra import IntegerAddition


def test_history_starts_with_empty_cursor() -> None:
    group = IntegerAddition()
    output = StreamIntroduction(1, group, Time1D)
    h: History[int, tuple[int]] = History(output)

    assert h.frontier.elements == []
    assert h.frontier.is_universal is False
    # No reads are legal before the first step.
    with pytest.raises(IndexError):
        h.at((0,))


def test_history_first_step_seeds_at_bottom() -> None:
    group = IntegerAddition()
    output = StreamIntroduction(7, group, Time1D)
    h: History[int, tuple[int]] = History(output)

    assert h.try_step() is True
    assert h.frontier.elements == [(0,)]
    assert h.at((0,)) == 7


def test_history_step_advances_along_chain() -> None:
    group = IntegerAddition()
    # δ₀(9) has value 9 at bottom, 0 elsewhere — universal frontier.
    output = StreamIntroduction(9, group, Time1D)
    h: History[int, tuple[int]] = History(output)

    assert h.try_step() is True
    assert h.at((0,)) == 9
    assert h.try_step() is True
    assert h.frontier.elements == [(1,)]
    assert h.at((0,)) == 9
    assert h.at((1,)) == 0
    assert h.try_step() is True
    assert h.at((2,)) == 0


def test_history_rejects_reads_outside_cursor() -> None:
    group = IntegerAddition()
    output = StreamIntroduction(1, group, Time1D)
    h: History[int, tuple[int]] = History(output)

    h.try_step()  # cursor now covers (0,)
    h.at((0,))  # ok
    with pytest.raises(IndexError):
        h.at((1,))


def test_history_step_on_non_universal_output_is_bounded_by_output_frontier() -> None:
    """A ``Delay`` of δ₀ is universal, but a non-universal base
    caps the output's settled frontier — History can only step while
    the output admits the proposed point.
    """
    group = IntegerAddition()
    s = StreamIntroduction(3, group, Time1D)
    delayed = Delay(s, Time1D)
    h: History[int, tuple[int]] = History(delayed)

    # δ₀-derived, so universal — steps always succeed.
    assert h.try_step() is True
    assert h.at((0,)) == 0
    assert h.try_step() is True
    assert h.at((1,)) == 3


def test_history_step_along_2d_axis() -> None:
    group = IntegerAddition()
    output = StreamIntroduction(5, group, Time2D)
    h: History[int, tuple[int, int]] = History(output)

    # Step the "inner" axis (axis=1) three times without advancing outer.
    assert h.try_step(axis=0) is True
    assert h.frontier.elements == [(0, 0)]
    assert h.try_step(axis=1) is True
    assert h.frontier.elements == [(0, 1)]
    assert h.try_step(axis=1) is True
    assert h.frontier.elements == [(0, 2)]
    # Step outer now.
    assert h.try_step(axis=0) is True
    assert h.frontier.elements == [(1, 2)]


def test_history_rejects_out_of_range_axis() -> None:
    group = IntegerAddition()
    output = StreamIntroduction(1, group, Time1D)
    h: History[int, tuple[int]] = History(output)
    with pytest.raises(IndexError):
        h.try_step(axis=1)  # only axis 0 exists for Time1D


def test_history_observes_stream_addition_output() -> None:
    group = IntegerAddition()
    stream_group = StreamAddition(group, Time1D)
    a = StreamIntroduction(2, group, Time1D)
    b = StreamIntroduction(3, group, Time1D)
    h: History[int, tuple[int]] = History(stream_group.add(a, b))

    h.try_step()
    assert h.at((0,)) == 5
    h.try_step()
    assert h.at((1,)) == 0


def test_history_reuses_attached_evaluator() -> None:
    group = IntegerAddition()
    output = StreamIntroduction(11, group, Time1D)
    ev = Evaluator(output)
    setattr(output, "_evaluator", ev)

    h: History[int, tuple[int]] = History(output)
    assert h._eval is ev
    h.try_step()
    assert h.at((0,)) == 11
    assert h._eval is ev
