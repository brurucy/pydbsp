from random import randrange
from statistics import mean, stdev
from time import time
from typing import List, Set, Tuple

from stream import Stream, StreamHandle
from stream_operator import UnaryOperator
from stream_operators import (
    Incremental2,
    Integrate,
    LiftedDelay,
    LiftedGroupAdd,
    LiftedStreamElimination,
    LiftedStreamIntroduction,
    StreamAddition,
    step_n_times_and_return,
    stream_introduction,
)
from test_stream_operators import from_stream_into_list, from_stream_of_streams_into_list_of_lists
from test_zset import Edge, GraphZSet, create_test_zset_graph
from zset import ZSet, join
from zset_operators import (
    DeltaLiftedDeltaLiftedDistinct,
    LiftedJoin,
    LiftedLiftedDeltaJoin,
    LiftedLiftedJoin,
    LiftedLiftedProject,
    LiftedLiftedSelect,
    LiftedProject,
    LiftedSelect,
    ZSetAddition,
)


def regular_join[K, V1, V2](left: Set[Tuple[K, V1]], right: Set[Tuple[K, V2]]) -> List[Tuple[K, V1, V2]]:
    output: List[Tuple[K, V1, V2]] = []

    for left_key, left_value in left:
        for right_key, right_value in right:
            if left_key == right_key:
                output.append((left_key, left_value, right_value))

    return output


def test_example() -> None:
    employees = {(0, "kristjan"), (1, "mark"), (2, "mike")}
    salaries = {(2, "40000"), (0, "38750"), (1, "50000")}
    employees_salaries = regular_join(employees, salaries)
    print(f"Regular join: {employees_salaries}")

    employees_zset = ZSet({k: 1 for k in employees})
    salaries_zset = ZSet({k: 1 for k in salaries})
    employees_salaries_zset = join(
        employees_zset,
        salaries_zset,
        lambda left, right: left[0] == right[0],
        lambda left, right: (left[0], left[1], right[1]),
    )
    print(f"ZSet join: {employees_salaries_zset}")

    group = ZSetAddition()
    employees_stream = Stream(group)
    employees_stream_handle = StreamHandle(lambda: employees_stream)
    employees_stream.send(employees_zset)

    salaries_stream = Stream(group)
    salaries_stream_handle = StreamHandle(lambda: salaries_stream)
    salaries_stream.send(salaries_zset)

    join_cmp = lambda left, right: left[0] == right[0]
    join_projection = lambda left, right: (left[0], left[1], right[1])

    integrated_employees = Integrate(employees_stream_handle)
    integrated_salaries = Integrate(salaries_stream_handle)
    stream_join = LiftedJoin(
        integrated_employees.output_handle(),
        integrated_salaries.output_handle(),
        join_cmp,
        join_projection,
    )
    integrated_employees.step()
    integrated_salaries.step()
    stream_join.step()
    print(f"ZSet stream join: {stream_join.output().latest()}")

    incremental_stream_join = Incremental2(
        employees_stream_handle,
        salaries_stream_handle,
        lambda left, right: join(left, right, join_cmp, join_projection),
        group,
    )
    incremental_stream_join.step()
    print(f"Incremental ZSet stream join: {incremental_stream_join.output().latest()}")

    employees_stream.send(ZSet({(2, "mike"): -1}))
    incremental_stream_join.step()
    print(f"Incremental ZSet stream join update: {incremental_stream_join.output().latest()}")

    names = ("kristjan", "mark", "mike")
    max_pay = 100000
    fake_data = [((i, names[randrange(len(names))] + str(i)), (i, randrange(max_pay))) for i in range(3, 10003)]
    batch_size = 500
    fake_data_batches = [fake_data[i : i + batch_size] for i in range(0, len(fake_data), batch_size)]
    for batch in fake_data_batches:
        employees_stream.send(ZSet({employee: 1 for employee, _ in batch}))
        salaries_stream.send(ZSet({salary: 1 for _, salary in batch}))

    steps_to_take = len(fake_data_batches)
    time_start = time()
    incremental_measurements = []
    for _ in range(steps_to_take):
        local_time = time()
        incremental_stream_join.step()
        incremental_measurements.append(time() - local_time)
    print(f"Time taken - incremental: {time() - time_start}s")
    print(f"Per step - mean: {mean(incremental_measurements)}, std: {stdev(incremental_measurements)}")

    time_start = time()
    measurements = []
    for _ in range(steps_to_take):
        local_time = time()
        integrated_employees.step()
        integrated_salaries.step()
        stream_join.step()
        measurements.append(time() - local_time)
    print(f"Time taken - on demand: {time() - time_start}s")
    print(f"Per step - mean: {mean(measurements)}, std: {stdev(measurements)}")

    assert True == False


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

    selected_s = step_n_times_and_return(operator, s.current_time() + 1)

    assert from_stream_into_list(selected_s) == [ZSet({(0, 1): 1, (2, 3): 1})]


def test_lifted_lifted_select() -> None:
    n = 4
    s = create_zset_graph_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedLiftedSelect(s_handle, is_fst_even)

    selected_s = step_n_times_and_return(operator, s.current_time() + 1)

    assert from_stream_of_streams_into_list_of_lists(selected_s) == [[ZSet({(0, 1): 1, (2, 3): 1})]]


def mod_2(edge: Edge) -> int:
    return edge[0] % 2


def test_lifted_project() -> None:
    n = 4
    s = create_zset_graph_stream(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedProject(s_handle, mod_2)

    projected_s = step_n_times_and_return(operator, s.current_time() + 1)

    assert from_stream_into_list(projected_s) == [ZSet({0: 2, 1: 2})]


def test_lifted_lifted_project() -> None:
    n = 4
    s = create_zset_graph_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedLiftedProject(s_handle, mod_2)

    projected_s = step_n_times_and_return(operator, s.current_time() + 1)

    assert from_stream_of_streams_into_list_of_lists(projected_s) == [[ZSet({0: 2, 1: 2})]]


def test_lifted_join() -> None:
    n = 4
    s = create_zset_graph_stream(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedJoin(
        s_handle, s_handle, lambda left, right: left[1] == right[0], lambda left, right: (left[0], right[1])
    )

    joined_s = step_n_times_and_return(operator, s.current_time() + 1)

    assert from_stream_into_list(joined_s) == [ZSet({(0, 2): 1, (1, 3): 1, (2, 4): 1})]


def test_lifted_lifted_join() -> None:
    n = 4
    s = create_zset_graph_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedLiftedJoin(
        s_handle, s_handle, lambda left, right: left[1] == right[0], lambda left, right: (left[0], right[1])
    )

    joined_s = step_n_times_and_return(operator, s.current_time() + 1)

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

    operator = LiftedLiftedDeltaJoin(
        s_a_handle, s_b_handle, lambda left, right: left[0] == right[1], lambda left, right: (right[0], left[1])
    )

    joined_s = step_n_times_and_return(operator, len(s_inner))

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
    join: LiftedLiftedDeltaJoin[Edge, Edge, Edge]
    delta_input_join_sum: LiftedGroupAdd[Stream[GraphZSet]]
    distinct: DeltaLiftedDeltaLiftedDistinct[Edge]
    lift_delayed_distinct: LiftedDelay[GraphZSet]
    flattened_output: LiftedStreamElimination[GraphZSet]

    def __init__(self, stream: StreamHandle[GraphZSet]):
        self._input_stream = stream

        self.delta_input = LiftedStreamIntroduction(self._input_stream)

        self.join = LiftedLiftedDeltaJoin(
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
