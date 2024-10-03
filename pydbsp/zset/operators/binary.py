from typing import TypeVar

from pydbsp.stream import (
    Lift2,
    Stream,
    StreamHandle,
    step_until_fixpoint_and_return,
)
from pydbsp.zset import ZSet
from pydbsp.zset.functions.binary import H

T = TypeVar("T")


class LiftedH(Lift2[ZSet[T], ZSet[T], ZSet[T]]):
    def __init__(
        self,
        diff_stream_a: StreamHandle[ZSet[T]],
        integrated_stream_a: StreamHandle[ZSet[T]],
    ):
        super().__init__(diff_stream_a, integrated_stream_a, H, None)


class LiftedLiftedH(Lift2[Stream[ZSet[T]], Stream[ZSet[T]], Stream[ZSet[T]]]):
    def __init__(
        self,
        integrated_diff_stream_a: StreamHandle[Stream[ZSet[T]]],
        lifted_delayed_lifted_integrated_stream_a: StreamHandle[Stream[ZSet[T]]],
    ):
        super().__init__(
            integrated_diff_stream_a,
            lifted_delayed_lifted_integrated_stream_a,
            lambda x, y: step_until_fixpoint_and_return(LiftedH(StreamHandle(lambda: x), StreamHandle(lambda: y))),
            None,
        )
