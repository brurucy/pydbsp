from typing import Optional, TypeVar

from pydbsp.core import AbelianGroupOperation
from pydbsp.lazy_zset import LazyZSet
from pydbsp.lazy_zset.operators.binary import LiftedLiftedH
from pydbsp.stream import Stream, StreamHandle, UnaryOperator
from pydbsp.stream.operators.linear import Differentiate, Integrate, LiftedDelay, LiftedIntegrate

T = TypeVar("T")


class DeltaLiftedDeltaLiftedDistinct(UnaryOperator[Stream[LazyZSet[T]], Stream[LazyZSet[T]]]):
    """
    An operator for incrementally maintaining distinct elements in a stream of
    streams. See :func:`~pydbsp.zset.functions.binary.H`
    """

    integrated_diff_stream_a: Integrate[Stream[LazyZSet[T]]]
    lift_integrated_diff_stream_a: LiftedIntegrate[LazyZSet[T]]
    lift_delay_lift_integrated_diff_stream_a: LiftedDelay[LazyZSet[T]]
    lift_lift_H: LiftedLiftedH[T]
    diff_lift_lift_H: Differentiate[Stream[LazyZSet[T]]]

    def set_input(
        self,
        stream_handle: StreamHandle[Stream[LazyZSet[T]]],
        output_stream_group: Optional[AbelianGroupOperation[Stream[LazyZSet[T]]]],
    ) -> None:
        self.input_stream_handle = stream_handle
        self.integrated_diff_stream_a = Integrate(self.input_stream_handle)
        self.lift_integrated_diff_stream_a = LiftedIntegrate(self.integrated_diff_stream_a.output_handle())
        self.lift_delay_lift_integrated_diff_stream_a = LiftedDelay(self.lift_integrated_diff_stream_a.output_handle())
        self.lift_lift_H = LiftedLiftedH(
            self.integrated_diff_stream_a.output_handle(),
            self.lift_delay_lift_integrated_diff_stream_a.output_handle(),
        )
        self.diff_lift_lift_H = Differentiate(self.lift_lift_H.output_handle())  # type: ignore
        self.output_stream_handle = self.diff_lift_lift_H.output_handle()

    def __init__(self, diff_stream_a: Optional[StreamHandle[Stream[LazyZSet[T]]]]):
        super().__init__(diff_stream_a, None)

    def step(self) -> bool:
        fixedpoint_integrated_diff_stream_a = self.integrated_diff_stream_a.step()
        fixedpoint_lift_integrated_diff_stream_a = self.lift_integrated_diff_stream_a.step()
        fixedpoint_lift_delay_lift_integrated_diff_stream_a = self.lift_delay_lift_integrated_diff_stream_a.step()
        fixedpoint_lift_lift_H = self.lift_lift_H.step()
        fixedpoint_diff_lift_lift_H = self.diff_lift_lift_H.step()

        return (
            fixedpoint_integrated_diff_stream_a
            and fixedpoint_lift_integrated_diff_stream_a
            and fixedpoint_lift_delay_lift_integrated_diff_stream_a
            and fixedpoint_lift_lift_H
            and fixedpoint_diff_lift_lift_H
        )
