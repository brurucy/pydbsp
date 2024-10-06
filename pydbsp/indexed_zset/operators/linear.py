from typing import Optional

from pydbsp.core import AbelianGroupOperation
from pydbsp.indexed_zset import IndexedZSet, IndexedZSetAddition, Indexer
from pydbsp.indexed_zset.functions.linear import index_zset
from pydbsp.stream import Lift1, Stream, StreamHandle, step_until_fixpoint_and_return
from pydbsp.zset import ZSet, ZSetAddition


class LiftedIndex[I, T](Lift1[ZSet[T], IndexedZSet[I, T]]):
    """Creates a stream where the output at each timestamp is the input ZSet indexed according to some indexing function"""

    indexer: Indexer[T, I]

    def __init__(self, stream: Optional[StreamHandle[ZSet[T]]], indexer: Indexer[T, I]):
        self.indexer = indexer
        super().__init__(stream, lambda z: index_zset(z, indexer), None)

    def set_input(
        self,
        stream_handle: StreamHandle[ZSet[T]],
        output_stream_group: Optional[AbelianGroupOperation[IndexedZSet[I, T]]],
    ) -> None:
        self.input_stream_handle = stream_handle
        inner_group: ZSetAddition[T] = ZSetAddition()
        group = IndexedZSetAddition(inner_group, self.indexer)
        output = Stream(group)

        self.output_stream_handle = StreamHandle(lambda: output)


class LiftedLiftedIndex[I, T](Lift1[Stream[ZSet[T]], Stream[IndexedZSet[I, T]]]):
    """Creates a stream where the output at each timestamp is the input ZSet indexed according to some indexing function"""

    def __init__(self, stream: Optional[StreamHandle[Stream[ZSet[T]]]], indexer: Indexer[T, I]):
        super().__init__(
            stream, lambda sp: step_until_fixpoint_and_return(LiftedIndex(StreamHandle(lambda: sp), indexer)), None
        )
