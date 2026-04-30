"""Reachability on graph1000 via the flat-lattice circuit.

Both the plain and indexed variants converge to the same transitive
closure (|reach| = 11532) at diameter ≈ 30.
"""

from pathlib import Path

from pydbsp.algorithms.reachability import IncrementalReachability
from pydbsp.algorithms.reachability_indexed import (
    IncrementalReachabilityWithIndexing,
)
from pydbsp.zset import ZSet


Edge = tuple[int, int]


def _load_graph1000() -> ZSet[Edge]:
    edges: dict[Edge, int] = {}
    for line in Path("notebooks/data/graph1000.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        edges[(int(parts[0]), int(parts[1]))] = 1
    return ZSet(edges)


def test_flat_reachability_plain_converges_to_11532():
    all_edges = _load_graph1000()
    c = IncrementalReachability()
    # ``edges`` is a 1-D Input lifted into the 2-D body lattice via
    # ``TimeAxisIntroduction``, so its inner-axis frontier is universal by
    # construction — no per-inner-tick padding needed.
    c.edges.push((0,), all_edges)
    max_inner = 1 << 16
    for k in range(max_inner):
        diff = c.body_at((0, k))
        c.state.push((0, k), diff)
        if diff.inner == {} and k > 0:
            break
    diameter = k
    assert 25 <= diameter <= 35, diameter
    reach = c.observable_at((0,))
    assert len(reach.inner) == 11532, len(reach.inner)


def test_flat_reachability_indexed_matches_plain():
    all_edges = _load_graph1000()
    c = IncrementalReachabilityWithIndexing()
    # Indexed circuit exposes ``edges`` as a 1-D Input — no inner-axis padding
    # needed; TimeAxisIntroduction lifts its frontier across the inner axis.
    c.edges.push((0,), all_edges)
    max_inner = 1 << 16
    for k in range(max_inner):
        diff = c.body_at((0, k))
        c.state.push((0, k), diff)
        if diff.inner == {} and k > 0:
            break
    diameter = k
    reach = c.observable_at((0,))
    assert len(reach.inner) == 11532, len(reach.inner)
    assert 25 <= diameter <= 35, diameter


if __name__ == "__main__":
    test_flat_reachability_plain_converges_to_11532()
    test_flat_reachability_indexed_matches_plain()
    print("OK")
