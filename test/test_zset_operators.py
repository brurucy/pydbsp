from typing import List, Set, Tuple

from pydbsp.stream import (
    LiftedGroupAdd,
    Stream,
    StreamAddition,
    StreamHandle,
    UnaryOperator,
    step_until_timestamp_and_return,
)
from pydbsp.stream.functions.linear import stream_introduction
from pydbsp.stream.operators.linear import LiftedDelay, LiftedStreamElimination, LiftedStreamIntroduction
from pydbsp.zset import ZSet, ZSetAddition
from pydbsp.zset.operators.bilinear import DeltaLiftedDeltaLiftedJoin, LiftedJoin, LiftedLiftedJoin
from pydbsp.zset.operators.linear import LiftedLiftedProject, LiftedLiftedSelect, LiftedProject, LiftedSelect
from pydbsp.zset.operators.unary import DeltaLiftedDeltaLiftedDistinct

from test.test_stream_operators import from_stream_into_list, from_stream_of_streams_into_list_of_lists
from test.test_zset import Edge, GraphZSet, create_test_zset_graph


def regular_join[K, V1, V2](left: Set[Tuple[K, V1]], right: Set[Tuple[K, V2]]) -> List[Tuple[K, V1, V2]]:
    output: List[Tuple[K, V1, V2]] = []

    for left_key, left_value in left:
        for right_key, right_value in right:
            if left_key == right_key:
                output.append((left_key, left_value, right_value))

    return output


def test_zset_addition() -> None:
    n = 4
    test_graph = create_test_zset_graph(n)
    group: ZSetAddition[Edge] = ZSetAddition()

    assert group.is_associative(test_graph, test_graph, test_graph)
    assert group.is_commutative(test_graph, test_graph)
    assert group.has_identity(test_graph)
    assert group.has_inverse(test_graph)


def create_zset_graph_stream(n: int) -> Stream[GraphZSet]:
    group: ZSetAddition[Edge] = ZSetAddition()

    return stream_introduction(create_test_zset_graph(n), group)


def create_zset_graph_stream_of_streams(n: int) -> Stream[Stream[GraphZSet]]:
    inner_group: ZSetAddition[Edge] = ZSetAddition()
    group: StreamAddition[GraphZSet] = StreamAddition(inner_group)

    return stream_introduction(create_zset_graph_stream(n), group)


def is_fst_even(edge: Edge) -> bool:
    return edge[0] % 2 == 0


def test_lifted_select() -> None:
    n = 4
    s = create_zset_graph_stream(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedSelect(s_handle, is_fst_even)

    selected_s = step_until_timestamp_and_return(operator, s.current_time())

    assert from_stream_into_list(selected_s) == [ZSet({(0, 1): 1, (2, 3): 1})]


def test_lifted_lifted_select() -> None:
    n = 4
    s = create_zset_graph_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedLiftedSelect(s_handle, is_fst_even)

    selected_s = step_until_timestamp_and_return(operator, s.current_time())

    assert from_stream_of_streams_into_list_of_lists(selected_s) == [[ZSet({(0, 1): 1, (2, 3): 1})]]


def mod_2(edge: Edge) -> int:
    return edge[0] % 2


def test_lifted_project() -> None:
    n = 4
    s = create_zset_graph_stream(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedProject(s_handle, mod_2)

    projected_s = step_until_timestamp_and_return(operator, s.current_time())

    assert from_stream_into_list(projected_s) == [ZSet({0: 2, 1: 2})]


def test_lifted_lifted_project() -> None:
    n = 4
    s = create_zset_graph_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedLiftedProject(s_handle, mod_2)

    projected_s = step_until_timestamp_and_return(operator, s.current_time())

    assert from_stream_of_streams_into_list_of_lists(projected_s) == [[ZSet({0: 2, 1: 2})]]


def test_lifted_join() -> None:
    n = 4
    s = create_zset_graph_stream(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedJoin(
        s_handle, s_handle, lambda left, right: left[1] == right[0], lambda left, right: (left[0], right[1])
    )

    joined_s = step_until_timestamp_and_return(operator, s.current_time())

    assert from_stream_into_list(joined_s) == [ZSet({(0, 2): 1, (1, 3): 1, (2, 4): 1})]


def test_lifted_lifted_join() -> None:
    n = 4
    s = create_zset_graph_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedLiftedJoin(
        s_handle, s_handle, lambda left, right: left[1] == right[0], lambda left, right: (left[0], right[1])
    )

    joined_s = step_until_timestamp_and_return(operator, s.current_time())

    assert from_stream_of_streams_into_list_of_lists(joined_s) == [[ZSet({(0, 2): 1, (1, 3): 1, (2, 4): 1})]]


def from_edges_into_zset(edges: List[tuple[int, int]]) -> GraphZSet:
    return ZSet({edge: 1 for edge in edges})


def from_edges_into_zset_stream(edges: List[tuple[int, int]]) -> Stream[GraphZSet]:
    inner_group: ZSetAddition[Edge] = ZSetAddition()
    s: Stream[GraphZSet] = Stream(inner_group)
    for edge in edges:
        s.send(from_edges_into_zset([edge]))

    return s


def test_lifted_lifted_delta_join() -> None:
    inner_group: ZSetAddition[Edge] = ZSetAddition()
    s_inner: List[Stream[GraphZSet]] = [
        stream_introduction(GraphZSet({edge: 1}), inner_group) for edge in [(0, 1), (1, 2), (2, 3), (3, 4)]
    ]
    # (1, 2), (2, 3), (3, 4)
    s_a_inner: List[Stream[GraphZSet]] = s_inner[1:]
    # (0, 1)
    s_b_inner = [s_inner[0], from_edges_into_zset_stream([(0, 2)]), from_edges_into_zset_stream([(0, 3)])]

    s_a: Stream[Stream[GraphZSet]] = Stream(StreamAddition(inner_group))
    s_b: Stream[Stream[GraphZSet]] = Stream(StreamAddition(inner_group))

    for s in s_a_inner:
        s_a.send(s)

    for s in s_b_inner:
        s_b.send(s)

    s_a_handle = StreamHandle(lambda: s_a)
    s_b_handle = StreamHandle(lambda: s_b)

    operator = DeltaLiftedDeltaLiftedJoin(
        s_a_handle, s_b_handle, lambda left, right: left[0] == right[1], lambda left, right: (right[0], left[1])
    )

    joined_s = step_until_timestamp_and_return(operator, len(s_inner) - 1)

    s1 = from_edges_into_zset_stream([(0, 2)])
    s2 = from_edges_into_zset_stream([(0, 3)])
    s3 = from_edges_into_zset_stream([(0, 4)])

    assert from_stream_of_streams_into_list_of_lists(joined_s) == [
        s1.inner,
        s2.inner,
        s3.inner,
        [inner_group.identity()],
    ]


class IncrementalGraphReachability(UnaryOperator[GraphZSet, GraphZSet]):
    delta_input: LiftedStreamIntroduction[GraphZSet]
    join: DeltaLiftedDeltaLiftedJoin[Edge, Edge, Edge]
    delta_input_join_sum: LiftedGroupAdd[Stream[GraphZSet]]
    distinct: DeltaLiftedDeltaLiftedDistinct[Edge]
    lift_delayed_distinct: LiftedDelay[GraphZSet]
    flattened_output: LiftedStreamElimination[GraphZSet]

    def __init__(self, stream: StreamHandle[GraphZSet]):
        self._input_stream = stream

        self.delta_input = LiftedStreamIntroduction(self._input_stream)

        self.join = DeltaLiftedDeltaLiftedJoin(
            None,
            None,
            lambda left, right: left[1] == right[0],
            lambda left, right: (left[0], right[1]),
        )
        self.delta_input_join_sum = LiftedGroupAdd(self.delta_input.output_handle(), self.join.output_handle())
        self.distinct = DeltaLiftedDeltaLiftedDistinct(self.delta_input_join_sum.output_handle())
        self.lift_delayed_distinct = LiftedDelay(self.distinct.output_handle())
        self.join.set_input_a(self.lift_delayed_distinct.output_handle())
        self.join.set_input_b(self.delta_input.output_handle())

        self.flattened_output = LiftedStreamElimination(self.distinct.output_handle())
        self.output_stream_handle = self.flattened_output.output_handle()

    def step(self) -> bool:
        self.delta_input.step()
        self.delta_input_join_sum.step()
        self.distinct.step()
        self.lift_delayed_distinct.step()
        self.join.step()
        self.flattened_output.step()

        return True


def create_zset_from_edges(edges: List[Edge]) -> GraphZSet:
    group: ZSetAddition[Edge] = ZSetAddition()

    output = group.identity()
    for edge in edges:
        output.inner[edge] = 1

    return output


def test_incremental_transitive_closure() -> None:
    n = 1
    s = create_zset_graph_stream(n)
    s_h = StreamHandle(lambda: s)

    op = IncrementalGraphReachability(s_h)
    op.step()
    assert op.output().latest() == create_zset_from_edges([(0, 1)])

    s.send(create_zset_from_edges([(0, 2)]))
    op.step()
    assert op.output().latest() == create_zset_from_edges([(0, 2)])
