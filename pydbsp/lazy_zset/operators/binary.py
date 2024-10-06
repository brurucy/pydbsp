from pydbsp.lazy_zset import LazyZSet
from pydbsp.lazy_zset.functions.binary import H
from pydbsp.stream import Lift2, Stream, StreamHandle, step_until_fixpoint_and_return


class LiftedH[T](Lift2[LazyZSet[T], LazyZSet[T], LazyZSet[T]]):
    def __init__(
        self,
        diff_stream_a: StreamHandle[LazyZSet[T]],
        integrated_stream_a: StreamHandle[LazyZSet[T]],
    ):
        super().__init__(diff_stream_a, integrated_stream_a, H, None)


class LiftedLiftedH[T](Lift2[Stream[LazyZSet[T]], Stream[LazyZSet[T]], Stream[LazyZSet[T]]]):
    def __init__(
        self,
        integrated_diff_stream_a: StreamHandle[Stream[LazyZSet[T]]],
        lifted_delayed_lifted_integrated_stream_a: StreamHandle[Stream[LazyZSet[T]]],
    ):
        super().__init__(
            integrated_diff_stream_a,
            lifted_delayed_lifted_integrated_stream_a,
            lambda x, y: step_until_fixpoint_and_return(LiftedH(StreamHandle(lambda: x), StreamHandle(lambda: y))),
            None,
        )
