from pydbsp.algorithms.graph_reachability import Edge
from pydbsp.lazy_zset import LazyZSet, LazyZSetAddition
from pydbsp.lazy_zset.functions.bilinear import join
from pydbsp.lazy_zset.functions.binary import H
from pydbsp.lazy_zset.functions.linear import project, select
from pydbsp.zset import ZSet

LazyGraphZSet = LazyZSet[Edge]


def create_test_lazy_zset_graph(n: int) -> LazyGraphZSet:
    return LazyZSet([ZSet({(k, k + 1): 1 for k in range(n)})])


def test_lazy_zset() -> None:
    n = 4
    test_graph = create_test_lazy_zset_graph(n)

    assert (0, 1) in test_graph
    assert (1, 2) in test_graph
    assert (2, 3) in test_graph
    assert (3, 4) in test_graph
    assert (1, 3) not in test_graph

    assert test_graph[(1, 3)] == 0
    assert test_graph[(0, 1)] == 1
    assert test_graph[(1, 2)] == 1
    assert test_graph[(2, 3)] == 1
    assert test_graph[(3, 4)] == 1


def test_select() -> None:
    n = 4
    test_graph = create_test_lazy_zset_graph(n)

    test_graph_prime = select(test_graph, lambda i: i[0] % 2 == 0)

    assert LazyZSet([ZSet({(0, 1): 1, (2, 3): 1})]) == test_graph_prime


def test_project() -> None:
    n = 4
    test_graph = create_test_lazy_zset_graph(n)

    test_graph_prime = project(test_graph, lambda i: i[0] % 2)

    assert LazyZSet([ZSet({0: 2, 1: 2})]) == test_graph_prime


def test_join() -> None:
    n = 4
    test_graph = create_test_lazy_zset_graph(n)

    one_hop = join(
        test_graph, test_graph, lambda left, right: left[1] == right[0], lambda left, right: (left[0], right[1])
    )

    assert one_hop == LazyZSet([ZSet({(0, 2): 1, (1, 3): 1, (2, 4): 1})])


def test_H() -> None:
    n = 4
    group: LazyZSetAddition[Edge] = LazyZSetAddition()
    test_graph: LazyGraphZSet = create_test_lazy_zset_graph(n)

    id = H(test_graph, test_graph)

    # No change
    assert id == group.identity()
    # New
    diff_neg = group.neg(create_test_lazy_zset_graph(1))
    diff_neg_times_two = group.add(diff_neg, diff_neg)

    assert diff_neg == H(diff_neg_times_two, test_graph)

    diff_pos = LazyGraphZSet([ZSet({(2, 4): 1})])
    diff_pos_times_two = group.add(diff_pos, diff_pos)

    assert diff_pos == H(diff_pos_times_two, test_graph)
