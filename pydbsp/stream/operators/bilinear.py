from typing import Optional, TypeVar

from pydbsp.core import AbelianGroupOperation
from pydbsp.stream import F2, BinaryOperator, Lift2, StreamHandle
from pydbsp.stream.operators.linear import Delay, Integrate

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")


class Incrementalize2(BinaryOperator[T, R, S]):
    """
    Given some bilinear function f, it both lifts it AND makes it incremental.

    For instance, if f is the relational join ⨝ over Z-sets **a** and **b**, incrementalizing it
    would create a circuit with three joins, yielding at each timestamp t: a[t] ⨝ b[t] + z^1(I(a))[t] ⨝ b[t] + a[t] ⨝ z^1(I(b))[t]

    Tl;dr this makes the join react to streams of additions and deletions

    """

    integrated_stream_a: Integrate[T]
    delayed_integrated_stream_a: Delay[T]

    integrated_stream_b: Integrate[R]
    delayed_integrated_stream_b: Delay[R]

    f2_a_b: Lift2[T, R, S]
    f2_delayed_integrated_a_b: Lift2[T, R, S]
    f2_a_delayed_integrated_b: Lift2[T, R, S]

    def __init__(
        self,
        stream_a: Optional[StreamHandle[T]],
        stream_b: Optional[StreamHandle[R]],
        f2: F2[T, R, S],
        output_stream_group: Optional[AbelianGroupOperation[S]],
    ) -> None:
        self.f2 = f2
        super().__init__(stream_a, stream_b, output_stream_group)

    def set_input_a(self, stream_handle_a: StreamHandle[T]) -> None:
        self.input_stream_handle_a = stream_handle_a
        self.integrated_stream_a = Integrate(self.input_stream_handle_a)
        self.delayed_integrated_stream_a = Delay(self.integrated_stream_a.output_handle())

    def set_input_b(self, stream_handle_b: StreamHandle[R]) -> None:
        self.input_stream_handle_b = stream_handle_b
        self.integrated_stream_b = Integrate(self.input_stream_handle_b)
        self.delayed_integrated_stream_b = Delay(self.integrated_stream_b.output_handle())

        self.f2_a_b = Lift2(self.input_stream_handle_a, self.input_stream_handle_b, self.f2, None)
        self.f2_a_delayed_integrated_b = Lift2(
            self.input_stream_handle_a, self.delayed_integrated_stream_b.output_handle(), self.f2, None
        )
        self.f2_delayed_integrated_a_b = Lift2(
            self.delayed_integrated_stream_a.output_handle(), self.input_stream_handle_b, self.f2, None
        )

    def step(self) -> bool:
        """Computes a[t] ⨝ b[t] + z^1(I(a))[t] ⨝ b[t] + a[t] ⨝ z^1(I(b))[t]"""
        self.integrated_stream_a.step()
        self.delayed_integrated_stream_a.step()

        self.integrated_stream_b.step()
        self.delayed_integrated_stream_b.step()

        self.f2_a_b.step()
        self.f2_a_delayed_integrated_b.step()
        self.f2_delayed_integrated_a_b.step()

        ab = self.f2_a_b.output().latest()
        adib = self.f2_a_delayed_integrated_b.output().latest()
        diab = self.f2_delayed_integrated_a_b.output().latest()

        group = self.output().group()
        self.output().send(group.add(group.add(ab, adib), diab))

        return True
