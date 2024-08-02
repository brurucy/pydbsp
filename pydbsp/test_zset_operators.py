from typing import Callable, List
from zset import ZSet
from stream import Stream, StreamHandle
from stream_operators import StreamAddition, step_n_times_and_return, stream_introduction
from test_stream_operators import from_stream_into_list, from_stream_of_streams_into_list_of_lists
from test_zset import GraphZSet, create_test_zset_graph
from zset_operators import LiftedJoin, LiftedLiftedDeltaJoin, LiftedLiftedJoin, LiftedLiftedProject, LiftedLiftedSelect, LiftedProject, LiftedSelect, ZSetAddition


def test_zset_addition() -> None:
    n = 4
    test_graph = create_test_zset_graph(n)
    group: ZSetAddition[tuple[int, int]] = ZSetAddition()
    
    assert group.is_associative(test_graph, test_graph, test_graph)
    assert group.is_commutative(test_graph, test_graph)
    assert group.has_identity(test_graph)
    assert group.has_inverse(test_graph)


def create_zset_graph_stream(n: int) -> Stream[GraphZSet]:
    return stream_introduction(create_test_zset_graph(n), ZSetAddition())

def create_zset_graph_stream_of_streams(n: int) -> Stream[Stream[GraphZSet]]:
    return stream_introduction(create_zset_graph_stream(n), StreamAddition(ZSetAddition()))

is_fst_even: Callable[[tuple[int, int]], bool] = lambda i : i[0] % 2 == 0

def test_lifted_select() -> None:
    n = 4
    s = create_zset_graph_stream(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedSelect(s_handle, is_fst_even)

    selected_s = step_n_times_and_return(operator, s.current_time() + 1)

    assert from_stream_into_list(selected_s) == [ ZSet({ (0, 1): 1, (2, 3): 1 }) ]

def test_lifted_lifted_select() -> None:
    n = 4
    s = create_zset_graph_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedLiftedSelect(s_handle, is_fst_even)

    selected_s = step_n_times_and_return(operator, s.current_time() + 1)

    assert from_stream_of_streams_into_list_of_lists(selected_s) == [ [ ZSet({ (0, 1): 1, (2, 3): 1 }) ] ]

mod_2: Callable[[tuple[int, int]], int] = lambda i: i[0] % 2
    
def test_lifted_project() -> None:
    n = 4
    s = create_zset_graph_stream(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedProject(s_handle, mod_2)

    projected_s = step_n_times_and_return(operator, s.current_time() + 1)

    assert from_stream_into_list(projected_s) == [ ZSet({ 0: 2, 1: 2 }) ]

def test_lifted_lifted_project() -> None:
    n = 4
    s = create_zset_graph_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedLiftedProject(s_handle, mod_2)

    projected_s = step_n_times_and_return(operator, s.current_time() + 1)

    assert from_stream_of_streams_into_list_of_lists(projected_s) == [ [ ZSet({ 0: 2, 1: 2}) ] ]

def test_lifted_join() -> None:
    n = 4
    s = create_zset_graph_stream(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedJoin(s_handle, s_handle, lambda l, r: l[1] == r[0], lambda l, r: (l[0], r[1]))

    joined_s = step_n_times_and_return(operator, s.current_time() + 1)

    assert from_stream_into_list(joined_s) == [ ZSet({ (0, 2): 1, (1, 3): 1, (2, 4): 1 }) ]

def test_lifted_lifted_join() -> None:
    n = 4
    s = create_zset_graph_stream_of_streams(n)
    s_handle = StreamHandle(lambda: s)
    operator = LiftedLiftedJoin(s_handle, s_handle, lambda l, r: l[1] == r[0], lambda l, r: (l[0], r[1]))

    joined_s = step_n_times_and_return(operator, s.current_time() + 1)

    assert from_stream_of_streams_into_list_of_lists(joined_s) == [ [ ZSet({ (0, 2): 1, (1, 3): 1, (2, 4): 1 }) ] ]

def from_edges_into_zset(edges: List[tuple[int, int]]) -> GraphZSet:
    return ZSet( { edge: 1 for edge in edges })

def from_edges_into_zset_stream(edges: List[tuple[int, int]]) -> Stream[GraphZSet]:
    s: Stream[GraphZSet] = Stream(ZSetAddition())
    for edge in edges:
        s.send(from_edges_into_zset([edge]))

    return s

def test_lifted_lifted_delta_join() -> None:
    s_inner: List[Stream[GraphZSet]] = [ stream_introduction(GraphZSet({ edge: 1 }), ZSetAddition()) for edge in [ (0, 1), (1, 2), (2, 3), (3, 4) ]]
    # (1, 2), (2, 3), (3, 4)
    s_a_inner: List[Stream[GraphZSet]] = s_inner[1:]
    # (0, 1)
    s_b_inner = [
        s_inner[0],
        from_edges_into_zset_stream([ (0, 2) ]),
        from_edges_into_zset_stream([ (0, 3) ])
    ]

    s_a: Stream[Stream[GraphZSet]] = Stream(StreamAddition(ZSetAddition()))
    s_b: Stream[Stream[GraphZSet]] = Stream(StreamAddition(ZSetAddition()))
    
    for s in s_a_inner:
        s_a.send(s)

    for s in s_b_inner:
        s_b.send(s)
    
    s_a_handle = StreamHandle(lambda: s_a)
    s_b_handle = StreamHandle(lambda: s_b)
    
    operator = LiftedLiftedDeltaJoin(s_a_handle, s_b_handle, lambda l, r: l[0] == r[1], lambda l, r: (r[0], l[1]))

    joined_s = step_n_times_and_return(operator, len(s_inner))

    s1 = from_edges_into_zset_stream([(0, 2)])
    s1.send(GraphZSet({}))
    s2 = from_edges_into_zset_stream([(0, 3)])
    s2.send(GraphZSet({}))
    s3 = from_edges_into_zset_stream([(0, 4)])
    s3.send(GraphZSet({}))

    assert from_stream_of_streams_into_list_of_lists(joined_s) == [
        s1.inner,
        s2.inner,
        s3.inner,
        [s_a_inner[0].group().identity(), s_a_inner[0].group().identity()]
    ]
    
