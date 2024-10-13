from typing import Optional, TypeVar

from pydbsp.stream import (
    Lift1,
    LiftedGroupAdd,
    LiftedGroupNegate,
    Stream,
    StreamAddition,
    StreamHandle,
    UnaryOperator,
    step_until_fixpoint,
    step_until_fixpoint_and_return,
)
from pydbsp.stream.functions.linear import stream_elimination, stream_introduction

T = TypeVar("T")


class Delay(UnaryOperator[T, T]):
    """
    Delays the input stream by one timestamp.
    """

    def __init__(self, stream: Optional[StreamHandle[T]]) -> None:
        super().__init__(stream, None)

    def step(self) -> bool:
        """
        Outputs the previous value from the input stream.
        """
        output_timestamp = self.output().current_time()
        input_timestamp = self.input_a().current_time()

        if output_timestamp <= input_timestamp:
            self.output().send(self.input_a()[output_timestamp])

            return False

        return True


class Differentiate(UnaryOperator[T, T]):
    """
    Computes the difference between consecutive elements in the input stream.
    """

    delayed_stream: Delay[T]
    delayed_negated_stream: LiftedGroupNegate[T]
    differentiation_stream: LiftedGroupAdd[T]

    def __init__(self, stream: StreamHandle[T]) -> None:
        self.input_stream_handle = stream
        self.delayed_stream = Delay(self.input_stream_handle)
        self.delayed_negated_stream = LiftedGroupNegate(self.delayed_stream.output_handle())
        self.differentiation_stream = LiftedGroupAdd(
            self.input_stream_handle, self.delayed_negated_stream.output_handle()
        )
        self.output_stream_handle = self.differentiation_stream.output_handle()

    def step(self) -> bool:
        """
        Outputs the difference between the latest element from the input stream with the one before
        """
        self.delayed_stream.step()
        self.delayed_negated_stream.step()
        self.differentiation_stream.step()

        return self.output().current_time() == self.input_a().current_time()


class Integrate(UnaryOperator[T, T]):
    """
    Computes the running sum of the input stream.
    """

    delayed_stream: Delay[T]
    integration_stream: LiftedGroupAdd[T]

    def __init__(self, stream: StreamHandle[T]) -> None:
        self.input_stream_handle = stream
        self.integration_stream = LiftedGroupAdd(self.input_stream_handle, None)
        self.delayed_stream = Delay(self.integration_stream.output_handle())
        self.integration_stream.set_input_b(self.delayed_stream.output_handle())

        self.output_stream_handle = self.integration_stream.output_handle()

    def step(self) -> bool:
        """
        Adds the latest element from the input stream to the running sum
        """
        self.delayed_stream.step()
        self.integration_stream.step()

        return self.output().current_time() == self.input_a().current_time()


def step_until_fixpoint_set_new_default_then_return[T](
    operator: Integrate[T] | Delay[T],
) -> Stream[T]:
    step_until_fixpoint(operator)

    out = operator.output_handle().get()
    latest = out.latest()
    out.set_default(latest)

    return out


class LiftedDelay(Lift1[Stream[T], Stream[T]]):
    """
    Lifts the Delay operator to work on streams of streams.
    """

    def __init__(self, stream: StreamHandle[Stream[T]]):
        super().__init__(
            stream,
            lambda s: step_until_fixpoint_set_new_default_then_return(Delay(StreamHandle(lambda: s))),
            None,
        )


class LiftedIntegrate(Lift1[Stream[T], Stream[T]]):
    """
    Lifts the Integrate operator to work on streams of streams.
    """

    def __init__(self, stream: StreamHandle[Stream[T]]):
        super().__init__(
            stream,
            lambda s: step_until_fixpoint_set_new_default_then_return(Integrate(StreamHandle(lambda: s))),
            None,
        )


class LiftedDifferentiate(Lift1[Stream[T], Stream[T]]):
    """
    Lifts the Differentiate operator to work on streams of streams.
    """

    def __init__(self, stream: StreamHandle[Stream[T]]):
        super().__init__(stream, lambda s: step_until_fixpoint_and_return(Differentiate(StreamHandle(lambda: s))), None)


class LiftedStreamIntroduction(Lift1[T, Stream[T]]):
    """
    Lifts the stream_introduction function to work on streams.
    """

    def __init__(self, stream: StreamHandle[T]) -> None:
        group = stream.get().group()

        super().__init__(stream, lambda x: stream_introduction(x, group), StreamAddition(group))


class LiftedStreamElimination(Lift1[Stream[T], T]):
    """
    Lifts the stream_elimination function to work on streams of streams.
    """

    def __init__(self, stream: StreamHandle[Stream[T]]) -> None:
        super().__init__(stream, lambda x: stream_elimination(x), stream.get().group().inner_group())  # type: ignore
