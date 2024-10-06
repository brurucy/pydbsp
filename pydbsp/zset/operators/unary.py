from typing import Optional, TypeVar

from pydbsp.core import AbelianGroupOperation
from pydbsp.stream import Stream, StreamHandle, UnaryOperator
from pydbsp.stream.operators.linear import Differentiate, Integrate, LiftedDelay, LiftedIntegrate
from pydbsp.zset import ZSet
from pydbsp.zset.operators.binary import LiftedLiftedH

T = TypeVar("T")


class DeltaLiftedDeltaLiftedDistinct(UnaryOperator[Stream[ZSet[T]], Stream[ZSet[T]]]):
    """
    An operator for incrementally maintaining distinct elements in a stream of
    streams. See :func:`~pydbsp.zset.functions.binary.H`
    """

    integrated_diff_stream_a: Integrate[Stream[ZSet[T]]]
    lift_integrated_diff_stream_a: LiftedIntegrate[ZSet[T]]
    lift_delay_lift_integrated_diff_stream_a: LiftedDelay[ZSet[T]]
    lift_lift_H: LiftedLiftedH[T]
    diff_lift_lift_H: Differentiate[Stream[ZSet[T]]]

    def set_input(
        self,
        stream_handle: StreamHandle[Stream[ZSet[T]]],
        output_stream_group: Optional[AbelianGroupOperation[Stream[ZSet[T]]]],
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

    def __init__(self, diff_stream_a: Optional[StreamHandle[Stream[ZSet[T]]]]):
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
