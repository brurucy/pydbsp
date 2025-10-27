from typing import Dict

from pydbsp.stream import Stream, StreamHandle
from pydbsp.zset import ZSet, ZSetAddition


def from_dict_into_singleton_stream[T](d: Dict[T, int]) -> StreamHandle[ZSet[T]]:
    group: ZSetAddition[T] = ZSetAddition()
    stream = Stream(group)
    for k, v in d.items():
        stream.send(ZSet({k: v}))

    return StreamHandle(lambda: stream)
