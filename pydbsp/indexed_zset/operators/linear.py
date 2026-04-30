"""Linear operators over indexed-zset streams."""

from collections.abc import Callable

from pydbsp.indexed_zset import IndexedZSet, IndexedZSetAddition
from pydbsp.stream import Lift1, Stream
from pydbsp.zset import ZSet, ZSetAddition


def Index[T, I, Time: tuple[int, ...]](
    s: Stream[ZSet[T], Time],
    indexer: Callable[[T], I],
    inner_group: ZSetAddition[T],
) -> Stream[IndexedZSet[I, T], Time]:
    """Wrap each ZSet in an ``IndexedZSet`` keyed by ``indexer``.
    Pointwise at every lattice point.

    Empty ZSets share one identity IndexedZSet — avoids the O(1)
    ``__init__`` dance for the many empty-delta cells in a DLD
    expansion (e.g. edges at inner>0 in reachability).
    """
    out_group: IndexedZSetAddition[I, T] = IndexedZSetAddition(inner_group, indexer)
    empty = out_group.identity()

    def lift(z: ZSet[T]) -> IndexedZSet[I, T]:
        if not z.inner:
            return empty
        return IndexedZSet(z.inner, indexer)

    return Lift1(s, lift, out_group)
