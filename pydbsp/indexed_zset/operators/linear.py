from typing import Optional, TypeVar

from pydbsp.core import AbelianGroupOperation
from pydbsp.indexed_zset import IndexedZSet, IndexedZSetAddition, Indexer
from pydbsp.indexed_zset.functions.linear import index_zset
from pydbsp.stream import Lift1, Stream, StreamHandle
from pydbsp.zset import ZSet, ZSetAddition

T = TypeVar("T")
I = TypeVar("I")


class LiftedIndex(Lift1[ZSet[T], IndexedZSet[T, I]]):
    indexer: Indexer[T, I]

    def __init__(self, stream: Optional[StreamHandle[ZSet[T]]], indexer: Indexer[T, I]):
        self.indexer = indexer
        super().__init__(stream, lambda z: index_zset(z, indexer), None)

    def set_input(
        self,
        stream_handle: StreamHandle[ZSet[T]],
        output_stream_group: Optional[AbelianGroupOperation[IndexedZSet[T, I]]],
    ) -> None:
        self.input_stream_handle = stream_handle
        inner_group: ZSetAddition[T] = ZSetAddition()
        group = IndexedZSetAddition(inner_group, self.indexer)
        output = Stream(group)

        self.output_stream_handle = StreamHandle(lambda: output)
