from typing import List, Tuple

from pydbsp.indexed_zset import IndexedZSet, IndexedZSetAddition, Indexer
from pydbsp.indexed_zset.operators.bilinear import (
    DeltaLiftedDeltaLiftedSortMergeJoin,
    LiftedLiftedSortMergeJoin,
    LiftedSortMergeJoin,
)
from pydbsp.indexed_zset.operators.linear import LiftedLiftedIndex
from pydbsp.stream import Stream, StreamAddition, StreamHandle, step_until_fixpoint_and_return
from pydbsp.stream.functions.linear import stream_introduction
from pydbsp.zset import ZSet, ZSetAddition

from test.test_stream_operators import from_stream_into_list, from_stream_of_streams_into_list_of_lists
from test.test_zset_operators import from_edges_into_zset_stream

Edge = Tuple[int, int]
IndexedGraphZSet = IndexedZSet[int, Edge]


def from_edges_into_indexed_zset(edges: List[Edge], indexer: Indexer[Edge, int]) -> IndexedGraphZSet:
    return IndexedZSet(values={edge: 1 for edge in edges}, indexer=indexer)


def from_edges_into_indexed_zset_stream_with_indexer(
    edges: List[Edge], indexer: Indexer[Edge, int]
) -> Stream[IndexedGraphZSet]:
    innermost_group: ZSetAddition[Edge] = ZSetAddition()
    inner_group: IndexedZSetAddition[int, Edge] = IndexedZSetAddition(innermost_group, indexer)
    s: Stream[IndexedZSet[int, Edge]] = Stream(inner_group)
    for edge in edges:
        s.send(from_edges_into_indexed_zset([edge], indexer))

    return s


def create_test_zset_graph(n: int, indexer: Indexer[Edge, int]) -> IndexedGraphZSet:
    return IndexedZSet({(k, k + 1): 1 for k in range(n)}, indexer)


def create_zset_graph_stream(n: int, indexer: Indexer[Edge, int]) -> Stream[IndexedGraphZSet]:
    inner_group: ZSetAddition[Edge] = ZSetAddition()
    group: IndexedZSetAddition[int, Edge] = IndexedZSetAddition(inner_group, indexer)

    return stream_introduction(create_test_zset_graph(n, indexer), group)


def create_zset_graph_stream_of_streams(n: int, indexer: Indexer[Edge, int]) -> Stream[Stream[IndexedGraphZSet]]:
    innermost_group: ZSetAddition[Edge] = ZSetAddition()
    inner_group: IndexedZSetAddition[int, Edge] = IndexedZSetAddition(innermost_group, indexer)
    group: StreamAddition[IndexedGraphZSet] = StreamAddition(inner_group)

    return stream_introduction(create_zset_graph_stream(n, indexer), group)


def index_by_fst(edge: Edge) -> int:
    return edge[0]


def index_by_snd(edge: Edge) -> int:
    return edge[1]


def test_lifted_join() -> None:
    n = 4
    s = create_zset_graph_stream(n, index_by_fst)
    s_handle = StreamHandle(lambda: s)
    s_prime = create_zset_graph_stream(n, index_by_snd)
    s_prime_handle = StreamHandle(lambda: s_prime)
    operator = LiftedSortMergeJoin(s_handle, s_prime_handle, lambda key, left, right: (right[0], left[1]))

    joined_s = step_until_fixpoint_and_return(operator)

    assert from_stream_into_list(joined_s) == [
        s.group().inner_group.identity(),  # type: ignore
        ZSet({(0, 2): 1, (1, 3): 1, (2, 4): 1}),
    ]


def test_lifted_lifted_join() -> None:
    n = 4
    s = create_zset_graph_stream_of_streams(n, index_by_fst)
    s_handle = StreamHandle(lambda: s)
    s_prime = create_zset_graph_stream_of_streams(n, index_by_snd)
    s_prime_handle = StreamHandle(lambda: s_prime)

    operator = LiftedLiftedSortMergeJoin(s_handle, s_prime_handle, lambda key, left, right: (right[0], left[1]))

    joined_s = step_until_fixpoint_and_return(operator)

    empty_zset = s.group().identity().group().inner_group.identity()  # type: ignore
    assert from_stream_of_streams_into_list_of_lists(joined_s) == [
        [empty_zset],
        [empty_zset, ZSet({(0, 2): 1, (1, 3): 1, (2, 4): 1})],
    ]


def test_lifted_lifted_delta_join() -> None:
    innermost_group: ZSetAddition[Edge] = ZSetAddition()
    s_inner: List[Stream[ZSet[Edge]]] = [
        stream_introduction(ZSet({edge: 1}), innermost_group) for edge in [(0, 1), (1, 2), (2, 3), (3, 4)]
    ]
    s_a_inner: List[Stream[ZSet[Edge]]] = s_inner[1:]
    s_b_inner = [s_inner[0], from_edges_into_zset_stream([(0, 2)]), from_edges_into_zset_stream([(0, 3)])]

    s_a: Stream[Stream[ZSet[Edge]]] = Stream(StreamAddition(innermost_group))
    s_b: Stream[Stream[ZSet[Edge]]] = Stream(StreamAddition(innermost_group))

    for s in s_a_inner:
        s_a.send(s)

    for s in s_b_inner:
        s_b.send(s)

    s_a_handle = StreamHandle(
        lambda: step_until_fixpoint_and_return(LiftedLiftedIndex(StreamHandle(lambda: s_a), index_by_fst))
    )
    s_b_handle = StreamHandle(
        lambda: step_until_fixpoint_and_return(LiftedLiftedIndex(StreamHandle(lambda: s_b), index_by_snd))
    )

    operator = DeltaLiftedDeltaLiftedSortMergeJoin(s_a_handle, s_b_handle, lambda key, left, right: (right[0], left[1]))
    joined_s = step_until_fixpoint_and_return(operator)

    s1 = from_edges_into_zset_stream([(0, 2)])
    s2 = from_edges_into_zset_stream([(0, 3)])
    s3 = from_edges_into_zset_stream([(0, 4)])

    empty_zset = innermost_group.identity()
    assert from_stream_of_streams_into_list_of_lists(joined_s) == [
        [empty_zset],
        s1.to_list(),
        s2.to_list(),
        s3.to_list(),
    ]
