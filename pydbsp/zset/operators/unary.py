from typing import Optional, TypeVar

from pydbsp.core import AbelianGroupOperation
from pydbsp.stream import Stream, StreamHandle, UnaryOperator
from pydbsp.stream.operators.linear import Differentiate, Integrate, LiftedDelay, LiftedIntegrate
from pydbsp.zset import ZSet
from pydbsp.zset.operators.binary import LiftedLiftedH

T = TypeVar("T")


class DeltaLiftedDeltaLiftedDistinct(UnaryOperator[Stream[ZSet[T]], Stream[ZSet[T]]]):
    """
    A complex operator for maintaining distinct elements in a stream of streams.
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
        self._input_stream_a = stream_handle
        self.integrated_diff_stream_a = Integrate(self._input_stream_a)
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
        self.integrated_diff_stream_a.step()
        self.lift_integrated_diff_stream_a.step()
        self.lift_delay_lift_integrated_diff_stream_a.step()
        self.lift_lift_H.step()

        return self.diff_lift_lift_H.step()
