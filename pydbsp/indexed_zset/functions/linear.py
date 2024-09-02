from pydbsp.indexed_zset import IndexedZSet, Indexer
from pydbsp.zset import ZSet


def index_zset[T, I](zset: ZSet[T], indexer: Indexer[T, I]) -> IndexedZSet[T, I]:
    """Indexes a Z-Set according to some indexer function."""
    return IndexedZSet(zset.inner, indexer)
