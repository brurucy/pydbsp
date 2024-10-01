from typing import TypeVar

from pydbsp.stream import (
    Lift2,
    Stream,
    StreamHandle,
    TimeTrackingLift2,
    step_until_false_and_return,
    step_until_timestamp_and_return,
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


class LiftedTimeTrackingH(TimeTrackingLift2[ZSet[T], ZSet[T], ZSet[T]]):
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
            lambda x, y: step_until_timestamp_and_return(
                LiftedH(StreamHandle(lambda: x), StreamHandle(lambda: y)),
                min(x.current_time(), y.current_time()),
            ),
            None,
        )


class LiftedLiftedTimeTrackingH(TimeTrackingLift2[Stream[ZSet[T]], Stream[ZSet[T]], Stream[ZSet[T]]]):
    def __init__(
        self,
        integrated_diff_stream_a: StreamHandle[Stream[ZSet[T]]],
        lifted_delayed_lifted_integrated_stream_a: StreamHandle[Stream[ZSet[T]]],
    ):
        super().__init__(
            integrated_diff_stream_a,
            lifted_delayed_lifted_integrated_stream_a,
            lambda x, y: step_until_false_and_return(
                LiftedTimeTrackingH(StreamHandle(lambda: x), StreamHandle(lambda: y))
            ),
            None,
        )
