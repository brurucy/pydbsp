from typing import Optional

from pydbsp.indexed_zset import IndexedZSet, IndexedZSetAddition, Indexer
from pydbsp.indexed_zset.functions.linear import index_zset
from pydbsp.stream import Lift1, Stream, StreamAddition, StreamHandle, step_until_fixpoint_and_return
from pydbsp.zset import ZSet, ZSetAddition


class LiftedIndex[I, T](Lift1[ZSet[T], IndexedZSet[I, T]]):
    """Creates a stream where the output at each timestamp is the input ZSet indexed according to some indexing function"""

    indexer: Indexer[T, I]

    def __init__(self, stream: Optional[StreamHandle[ZSet[T]]], indexer: Indexer[T, I]):
        self.indexer = indexer
        inner_group: ZSetAddition[T] = ZSetAddition()
        group = IndexedZSetAddition(inner_group, self.indexer)

        super().__init__(stream, lambda z: index_zset(z, self.indexer), group)


class LiftedLiftedIndex[I, T](Lift1[Stream[ZSet[T]], Stream[IndexedZSet[I, T]]]):
    """Creates a stream where the output at each timestamp is the input ZSet indexed according to some indexing function"""

    indexer: Indexer[T, I]

    def __init__(self, stream: Optional[StreamHandle[Stream[ZSet[T]]]], indexer: Indexer[T, I]):
        self.indexer = indexer
        innermost_group: ZSetAddition[T] = ZSetAddition()
        inner_group = IndexedZSetAddition(innermost_group, self.indexer)
        outer_group = StreamAddition(inner_group)

        super().__init__(
            stream,
            lambda sp: step_until_fixpoint_and_return(LiftedIndex(StreamHandle(lambda: sp), indexer)),
            outer_group,
        )
