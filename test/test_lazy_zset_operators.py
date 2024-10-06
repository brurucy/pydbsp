from typing import List, Set, Tuple

from pydbsp.algorithms.graph_reachability import Edge, LazyIncrementalGraphReachability
from pydbsp.lazy_zset import LazyZSet, LazyZSetAddition
from pydbsp.lazy_zset.operators.bilinear import DeltaLiftedDeltaLiftedJoin, LiftedJoin, LiftedLiftedJoin
from pydbsp.lazy_zset.operators.linear import LiftedLiftedProject, LiftedLiftedSelect, LiftedProject, LiftedSelect
from pydbsp.stream import (
    Stream,
    StreamAddition,
    StreamHandle,
    step_until_fixpoint,
    step_until_fixpoint_and_return,
)
from pydbsp.stream.functions.linear import stream_introduction
from pydbsp.stream.operators.linear import (
    stream_elimination,
)
from pydbsp.zset import ZSet, ZSetAddition

from test.test_lazy_zset import LazyGraphZSet, create_test_lazy_zset_graph
from test.test_stream_operators import from_stream_into_list, from_stream_of_streams_into_list_of_lists


def regular_join[K, V1, V2](left: Set[Tuple[K, V1]], right: Set[Tuple[K, V2]]) -> List[Tuple[K, V1, V2]]:
    output: List[Tuple[K, V1, V2]] = []

    for left_key, left_value in left:
        for right_key, right_value in right:
            if left_key == right_key:
                output.append((left_key, left_value, right_value))

    return output


def test_zset_addition() -> None:
    n = 4
    test_graph = create_test_lazy_zset_graph(n)
    group: LazyZSetAddition[Edge] = LazyZSetAddition()

    assert group.is_associative(test_graph, test_graph, test_graph)
    assert group.is_commutative(test_graph, test_graph)
    assert group.has_identity(test_graph)
    assert group.has_inverse(test_graph)


def create_lazy_zset_graph_stream(n: int) -> Stream[LazyGraphZSet]:
    group: LazyZSetAddition[Edge] = LazyZSetAddition()

    return stream_introduction(create_test_lazy_zset_graph(n), group)


def create_lazy_zset_graph_stream_of_streams(n: int) -> Stream[Stream[LazyGraphZSet]]:
    inner_group: LazyZSetAddition[Edge] = LazyZSetAddition()
    group: StreamAddition[LazyGraphZSet] = StreamAddition(inner_group)

    return stream_introduction(create_lazy_zset_graph_stream(n), group)


def is_fst_even(edge: Edge) -> bool:
    return edge[0] % 2 == 0


def test_lifted_select() -> None:
    n = 4
    s = create_lazy_zset_graph_stream(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedSelect(s_handle, is_fst_even)

    selected_s = step_until_fixpoint_and_return(operator)
    empty_zset = s.group().identity()
    assert from_stream_into_list(selected_s) == [empty_zset, LazyZSet([ZSet({(0, 1): 1, (2, 3): 1})])]


def test_lifted_lifted_select() -> None:
    n = 4
    s = create_lazy_zset_graph_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedLiftedSelect(s_handle, is_fst_even)

    selected_s = step_until_fixpoint_and_return(operator)

    empty_zset = s.group().identity().group().identity()
    assert from_stream_of_streams_into_list_of_lists(selected_s) == [
        [empty_zset],
        [empty_zset, LazyZSet([ZSet({(0, 1): 1, (2, 3): 1})])],
    ]


def mod_2(edge: Edge) -> int:
    return edge[0] % 2


def test_lifted_project() -> None:
    n = 4
    s = create_lazy_zset_graph_stream(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedProject(s_handle, mod_2)

    projected_s = step_until_fixpoint_and_return(operator)

    empty_zset = s.group().identity()
    assert from_stream_into_list(projected_s) == [empty_zset, LazyZSet([ZSet({0: 2, 1: 2})])]


def test_lifted_lifted_project() -> None:
    n = 4
    s = create_lazy_zset_graph_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedLiftedProject(s_handle, mod_2)

    projected_s = step_until_fixpoint_and_return(operator)

    empty_zset = s.group().identity().group().identity()
    assert from_stream_of_streams_into_list_of_lists(projected_s) == [
        [empty_zset],
        [empty_zset, LazyZSet([ZSet({0: 2, 1: 2})])],
    ]


def test_lifted_join() -> None:
    n = 4
    s = create_lazy_zset_graph_stream(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedJoin(
        s_handle, s_handle, lambda left, right: left[1] == right[0], lambda left, right: (left[0], right[1])
    )

    joined_s = step_until_fixpoint_and_return(operator)

    assert from_stream_into_list(joined_s) == [
        s.group().identity(),
        LazyZSet([ZSet({(0, 2): 1, (1, 3): 1, (2, 4): 1})]),
    ]


def test_lifted_lifted_join() -> None:
    n = 4
    s = create_lazy_zset_graph_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedLiftedJoin(
        s_handle, s_handle, lambda left, right: left[1] == right[0], lambda left, right: (left[0], right[1])
    )

    joined_s = step_until_fixpoint_and_return(operator)

    empty_zset = s.group().identity().group().identity()
    assert from_stream_of_streams_into_list_of_lists(joined_s) == [
        [empty_zset],
        [empty_zset, LazyZSet([ZSet({(0, 2): 1, (1, 3): 1, (2, 4): 1})])],
    ]


def from_edges_into_zset(edges: List[tuple[int, int]]) -> LazyGraphZSet:
    return LazyZSet([ZSet({edge: 1 for edge in edges})])


def from_edges_into_lazy_zset_stream(edges: List[tuple[int, int]]) -> Stream[LazyGraphZSet]:
    inner_group: LazyZSetAddition[Edge] = LazyZSetAddition()
    s: Stream[LazyGraphZSet] = Stream(inner_group)
    for edge in edges:
        s.send(from_edges_into_zset([edge]))

    return s


def test_lifted_lifted_delta_join() -> None:
    inner_group: LazyZSetAddition[Edge] = LazyZSetAddition()
    s_inner: List[Stream[LazyGraphZSet]] = [
        stream_introduction(LazyGraphZSet([ZSet({edge: 1})]), inner_group) for edge in [(0, 1), (1, 2), (2, 3), (3, 4)]
    ]
    s_a_inner: List[Stream[LazyGraphZSet]] = s_inner[1:]
    s_b_inner = [s_inner[0], from_edges_into_lazy_zset_stream([(0, 2)]), from_edges_into_lazy_zset_stream([(0, 3)])]

    s_a: Stream[Stream[LazyGraphZSet]] = Stream(StreamAddition(inner_group))
    s_b: Stream[Stream[LazyGraphZSet]] = Stream(StreamAddition(inner_group))

    for s in s_a_inner:
        s_a.send(s)

    for s in s_b_inner:
        s_b.send(s)

    s_a_handle = StreamHandle(lambda: s_a)
    s_b_handle = StreamHandle(lambda: s_b)

    operator = DeltaLiftedDeltaLiftedJoin(
        s_a_handle, s_b_handle, lambda left, right: left[0] == right[1], lambda left, right: (right[0], left[1])
    )

    joined_s = step_until_fixpoint_and_return(operator)

    s1 = from_edges_into_lazy_zset_stream([(0, 2)])
    s2 = from_edges_into_lazy_zset_stream([(0, 3)])
    s3 = from_edges_into_lazy_zset_stream([(0, 4)])

    empty_zset = inner_group.identity()
    assert from_stream_of_streams_into_list_of_lists(joined_s) == [
        [empty_zset],
        s1.inner + [empty_zset],
        s2.inner + [empty_zset],
        s3.inner + [empty_zset],
    ]


def create_lazy_zset_from_edges(edges: List[Edge]) -> LazyGraphZSet:
    group: ZSetAddition[Edge] = ZSetAddition()

    output = group.identity()
    for edge in edges:
        output.inner[edge] = 1

    return LazyZSet([output])


def test_incremental_transitive_closure() -> None:
    n = 1
    s = create_lazy_zset_graph_stream(n)
    s_h = StreamHandle(lambda: s)

    op = LazyIncrementalGraphReachability(s_h)
    op.step()

    s.send(create_lazy_zset_from_edges([(1, 2)]))
    step_until_fixpoint(op)
    expected_integrated_state = create_lazy_zset_from_edges([(0, 1), (1, 2), (0, 2)])
    actual_integrated_state = stream_elimination(op.output())
    assert actual_integrated_state == expected_integrated_state

    s.send(LazyZSet([ZSet({(0, 1): -1})]))
    step_until_fixpoint(op)
    expected_integrated_state = create_lazy_zset_from_edges([(1, 2)])
    actual_integrated_state = stream_elimination(op.output())
    assert actual_integrated_state.coalesce() == expected_integrated_state.coalesce()

    s.send(LazyZSet([ZSet({(2, 3): 1})]))
    step_until_fixpoint(op)
    expected_integrated_state = create_lazy_zset_from_edges([(1, 2), (2, 3), (1, 3)])
    actual_integrated_state = stream_elimination(op.output())
    assert actual_integrated_state == expected_integrated_state

    s.send(LazyZSet([ZSet({(3, 4): 1})]))
    step_until_fixpoint(op)
    expected_integrated_state = create_lazy_zset_from_edges([(1, 2), (2, 3), (1, 3), (3, 4), (1, 4), (2, 4)])
    actual_integrated_state = stream_elimination(op.output())
    assert actual_integrated_state == expected_integrated_state

    s.send(LazyZSet([ZSet({(0, 1): 1})]))
    step_until_fixpoint(op)
    expected_integrated_state = create_lazy_zset_from_edges(
        [(0, 1), (1, 2), (2, 3), (1, 3), (3, 4), (1, 4), (2, 4), (0, 2), (0, 3), (0, 4)]
    )
    actual_integrated_state = stream_elimination(op.output())
    assert actual_integrated_state == expected_integrated_state

    s.send(LazyZSet([ZSet({(0, 1): -1})]))
    step_until_fixpoint(op)
    expected_integrated_state = create_lazy_zset_from_edges(
        [
            (1, 2),
            (2, 3),
            (1, 3),
            (3, 4),
            (1, 4),
            (2, 4),
        ]
    )
    actual_integrated_state = stream_elimination(op.output())
    assert actual_integrated_state == expected_integrated_state

    s.send(LazyZSet([ZSet({(0, 1): 1})]))
    step_until_fixpoint(op)
    expected_integrated_state = create_lazy_zset_from_edges(
        [(0, 1), (1, 2), (2, 3), (1, 3), (3, 4), (1, 4), (2, 4), (0, 2), (0, 3), (0, 4)]
    )
    actual_integrated_state = stream_elimination(op.output())
    assert actual_integrated_state == expected_integrated_state

    s.send(LazyZSet([ZSet({(0, 1): -1})]))
    step_until_fixpoint(op)
    expected_integrated_state = create_lazy_zset_from_edges([(1, 2), (2, 3), (1, 3), (3, 4), (1, 4), (2, 4)])
    actual_integrated_state = stream_elimination(op.output())
    assert actual_integrated_state == expected_integrated_state
