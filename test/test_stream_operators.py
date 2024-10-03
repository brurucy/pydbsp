from collections import deque
from typing import List

from pydbsp.stream import (
    LiftedGroupAdd,
    Stream,
    StreamAddition,
    StreamHandle,
    step_until_fixpoint_and_return,
)
from pydbsp.stream.functions.linear import stream_elimination, stream_introduction
from pydbsp.stream.operators.linear import (
    Delay,
    Differentiate,
    Integrate,
    LiftedDelay,
    LiftedDifferentiate,
    LiftedGroupNegate,
    LiftedIntegrate,
    LiftedStreamElimination,
    LiftedStreamIntroduction,
)

from test.test_algebra import IntegerAddition


def create_integer_stream_up_to(n: int) -> Stream[int]:
    s = Stream(IntegerAddition())
    for i in range(n):
        s.send(i)

    return s


def test_delay() -> None:
    operator: Delay[int] = Delay(None)
    s = create_integer_stream_up_to(10)

    operator.set_input(StreamHandle(lambda: s), None)

    delayed_s = step_until_fixpoint_and_return(operator)

    delayed_list = deque(s.inner)
    delayed_list.appendleft(s.group().identity())

    assert delayed_s.inner == list(delayed_list)
    assert delayed_s.current_time() == s.current_time() + 1


def test_lifted_group_negate() -> None:
    s = create_integer_stream_up_to(10)
    s_handle = StreamHandle(lambda: s)
    operator: LiftedGroupNegate[int] = LiftedGroupNegate(s_handle)

    negated_s = step_until_fixpoint_and_return(operator)

    negated_list = [s.group().neg(i) for i in s.inner]

    assert negated_s.inner == negated_list
    assert negated_s.current_time() == s.current_time()


def test_lifted_group_add() -> None:
    s = create_integer_stream_up_to(10)
    s_handle = StreamHandle(lambda: s)
    operator: LiftedGroupAdd[int] = LiftedGroupAdd(s_handle, s_handle)

    doubled_s = step_until_fixpoint_and_return(operator)

    doubled_list = [s.group().add(i, i) for i in s.inner]

    assert doubled_s.inner == doubled_list  # type: ignore
    assert doubled_s.current_time() == s.current_time()  # type: ignore


def test_differentiate() -> None:
    s = create_integer_stream_up_to(10)
    s_handle = StreamHandle(lambda: s)
    operator: Differentiate[int] = Differentiate(s_handle)

    diffed_s = step_until_fixpoint_and_return(operator)

    diffed_list = [
        s.group().add(i, s.group().neg(s.inner[idx - 1])) if idx > 0 else 0 for (idx, i) in enumerate(s.inner)
    ]

    assert diffed_s.inner == diffed_list
    assert diffed_s.current_time() == s.current_time()


def test_integrate() -> None:
    s = create_integer_stream_up_to(10)
    s_handle = StreamHandle(lambda: s)
    operator: Integrate[int] = Integrate(s_handle)

    integrated_s = step_until_fixpoint_and_return(operator)

    integrated_list = [sum(s.inner[0 : idx + 1]) for (idx, _) in enumerate(s)]

    assert integrated_s.inner == integrated_list
    assert integrated_s.current_time() == s.current_time()


def create_stream_of_streams(n: int) -> Stream[Stream[int]]:
    s = Stream(StreamAddition(IntegerAddition()))
    for t1 in range(n):
        row = Stream(IntegerAddition())

        for t0 in range(n):
            row.send(t0 + (2 * t1))

        s.send(row)

    return s


def test_abelian_property_of_stream_addition():
    s = create_stream_of_streams(10)
    integer_group = s.group()
    a = s[0]
    b = s[1]
    c = integer_group.add(a, b)

    assert integer_group.has_identity(a)
    assert integer_group.has_inverse(a)
    assert integer_group.is_associative(a, b, c)
    assert integer_group.is_commutative(a, b)


def from_stream_into_list[T](s: Stream[T]) -> List[T]:
    return s.inner


def from_stream_of_streams_into_list_of_lists[T](xs: Stream[Stream[T]]) -> List[List[T]]:
    return [from_stream_into_list(s) for s in xs.inner]


def test_delay_stream_of_streams() -> None:
    n = 4
    s = create_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator: Delay[Stream[int]] = Delay(s_handle)

    delayed_s = step_until_fixpoint_and_return(operator)
    delayed_list = [[], [0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]]

    assert from_stream_of_streams_into_list_of_lists(delayed_s) == delayed_list
    assert delayed_s.current_time() == s.current_time() + 1


def test_lifted_delay() -> None:
    n = 4
    s = create_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator: LiftedDelay[int] = LiftedDelay(s_handle)

    delayed_s = step_until_fixpoint_and_return(operator)
    delayed_list = [[0, 0, 1, 2, 3], [0, 2, 3, 4, 5], [0, 4, 5, 6, 7], [0, 6, 7, 8, 9]]

    assert from_stream_of_streams_into_list_of_lists(delayed_s) == delayed_list
    assert delayed_s.current_time() == s.current_time()


def test_integrate_stream_of_streams() -> None:
    n = 4
    s = create_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator: Integrate[Stream[int]] = Integrate(s_handle)

    integrated_s = step_until_fixpoint_and_return(operator)
    integrated_list = [[0, 1, 2, 3], [2, 4, 6, 8], [6, 9, 12, 15], [12, 16, 20, 24]]
    assert from_stream_of_streams_into_list_of_lists(integrated_s) == integrated_list
    assert integrated_s.current_time() == s.current_time()


def test_lifted_integrate() -> None:
    n = 4
    s = create_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator: LiftedIntegrate[int] = LiftedIntegrate(s_handle)

    integrated_s = step_until_fixpoint_and_return(operator)
    integrated_list = [
        [
            0,
            1,
            3,
            6,
        ],
        [2, 5, 9, 14],
        [4, 9, 15, 22],
        [6, 13, 21, 30],
    ]
    assert from_stream_of_streams_into_list_of_lists(integrated_s) == integrated_list
    assert integrated_s.current_time() == s.current_time()


def test_differentiate_stream_of_streams() -> None:
    n = 4
    s = create_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator: Differentiate[Stream[int]] = Differentiate(s_handle)

    diffed_s = step_until_fixpoint_and_return(operator)
    diffed_list = [[0, 1, 2, 3], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
    assert from_stream_of_streams_into_list_of_lists(diffed_s) == diffed_list
    assert diffed_s.current_time() == s.current_time()


def test_lifted_differentiate() -> None:
    n = 4
    s = create_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator: LiftedDifferentiate[int] = LiftedDifferentiate(s_handle)

    diffed_s = step_until_fixpoint_and_return(operator)
    diffed_list = [[0, 1, 1, 1], [2, 1, 1, 1], [4, 1, 1, 1], [6, 1, 1, 1]]
    assert from_stream_of_streams_into_list_of_lists(diffed_s) == diffed_list
    assert diffed_s.current_time() == s.current_time()


def test_stream_introduction() -> None:
    i = 0
    s = create_integer_stream_up_to(1)
    s_prime = stream_introduction(i, s.group())

    assert s == s_prime
    assert s.group() == s_prime.group()


def test_stream_elimination() -> None:
    i = 0
    s = create_integer_stream_up_to(1)

    i_prime = stream_elimination(s)

    assert i == i_prime


def test_lifted_stream_introduction_and_elimination() -> None:
    n = 10
    s = create_integer_stream_up_to(n)
    i = sum(s.inner)

    s_handle = StreamHandle(lambda: s)
    intro_operator: LiftedStreamIntroduction[int] = LiftedStreamIntroduction(s_handle)
    step_until_fixpoint_and_return(intro_operator)

    elim_operator: LiftedStreamElimination[int] = LiftedStreamElimination(intro_operator.output_handle())
    step_until_fixpoint_and_return(elim_operator)

    assert s == elim_operator.output_handle().get()
    assert i == sum(elim_operator.output_handle().get().inner)
    assert i == stream_elimination(stream_elimination(intro_operator.output_handle().get()))
    assert stream_elimination(intro_operator.output_handle().get()).group() == s.group()
