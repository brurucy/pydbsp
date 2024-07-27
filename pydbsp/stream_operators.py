from stream import Stream, StreamHandle
from stream_operator import Operator, UnaryOperator, BinaryOperator
from algebra import AbelianGroupOperation
from typing import Optional, TypeVar, Callable 


T = TypeVar("T")
R = TypeVar("R")

class Delay(UnaryOperator[T, T]):
    def __init__(self, stream: Optional[StreamHandle[T]]) -> None:
        super().__init__(stream, None)

    def step(self) -> bool:
        output_timestamp = self.output().current_time()

        delayed_value = self.input_a()[output_timestamp]

        self.output().send(delayed_value)

        return True


F1 = Callable[[T], R]


class Lifted1(UnaryOperator[T, R]):
    f1: F1[T, R]

    def __init__(
        self,
        stream: Optional[StreamHandle[T]],
        f1: F1[T, R],
        output_stream_group: Optional[AbelianGroupOperation[R]],
    ):
        self.f1 = f1
        super().__init__(stream, output_stream_group)

    def step(self) -> bool:
        output_timestamp = self.output().current_time()

        self.output().send(self.f1(self.input_a()[output_timestamp + 1]))

        return True


class LiftedGroupNegate(Lifted1[T, T]):
    def __init__(self, stream: StreamHandle[T]):
        super().__init__(stream, lambda x: stream.get().group().neg(x), None)


S = TypeVar("S")
F2 = Callable[[T, R], S]

class Lifted2(BinaryOperator[T, R, S]):
    def __init__(
        self,
        stream_a: Optional[StreamHandle[T]],
        stream_b: Optional[StreamHandle[R]],
        f2: F2[T, R, S],
        output_stream_group: Optional[AbelianGroupOperation[S]],
    ) -> None:
        self.f2 = f2
        super().__init__(stream_a, stream_b, output_stream_group)

    def step(self) -> bool:
        output_timestamp = self.output().current_time()

        a = self.input_a()[output_timestamp + 1]
        b = self.input_b()[output_timestamp + 1]

        application = self.f2(a, b)
        self.output().send(application)

        return True


class LiftedGroupAdd(Lifted2[T, T, T]):
    def __init__(self, stream_a: StreamHandle[T], stream_b: Optional[StreamHandle[T]]):
        super().__init__(
            stream_a,
            stream_b,
            lambda x, y: stream_a.get().group().add(x, y),
            None,
        )

class Differentiate(UnaryOperator[T, T]):
    delayed_stream: Delay[T]
    delayed_negated_stream: LiftedGroupNegate[T]
    differentiation_stream: LiftedGroupAdd[T]

    def __init__(self, stream: StreamHandle[T]) -> None:
        self.input_stream_handle = stream
        self.delayed_stream = Delay(self.input_stream_handle)
        self.delayed_negated_stream = LiftedGroupNegate(
            self.delayed_stream.output_handle()
        )
        self.differentiation_stream = LiftedGroupAdd(
            self.input_stream_handle, self.delayed_negated_stream.output_handle()
        )
        self.output_stream_handle = self.differentiation_stream.output_handle()

    def step(self) -> bool:
        self.delayed_stream.step()
        self.delayed_negated_stream.step()
        return self.differentiation_stream.step()


class Integrate(UnaryOperator[T, T]):
    delayed_stream: Delay[T]
    integration_stream: LiftedGroupAdd[T]

    def __init__(self, stream: StreamHandle[T]) -> None:
        self.input_stream_handle = stream
        self.integration_stream = LiftedGroupAdd(self.input_stream_handle, None)
        self.delayed_stream = Delay(self.integration_stream.output_handle())
        self.integration_stream.set_input_b(self.delayed_stream.output_handle())

        self.output_stream_handle = self.integration_stream.output_handle()

    def step(self) -> bool:
        self.delayed_stream.step()
        return self.integration_stream.step()


def step_n_times[T](operator: Operator[T], n: int):
    for _ in range(n):
        operator.step()

def step_n_times_and_return[T](operator: Operator[T], n: int) -> Stream[T]:
    step_n_times(operator, n)

    return operator.output_handle().get()

class LiftedDelay(Lifted1[Stream[T], Stream[T]]):
    def __init__(self, stream: StreamHandle[Stream[T]]):
        super().__init__(
            stream,
            lambda s: step_n_times_and_return(
                Delay(StreamHandle(lambda: s)), s.current_time() + 1
            ),
            None,
        )


class LiftedIntegrate(Lifted1[Stream[T], Stream[T]]):
    def __init__(self, stream: StreamHandle[Stream[T]]):
        super().__init__(
            stream,
            lambda s: step_n_times_and_return(
                Integrate(StreamHandle(lambda: s)), s.current_time() + 1
            ),
            None,
        )


class LiftedDifferentiate(
    Lifted1[Stream[T], Stream[T]] 
):
    def __init__(self, stream: StreamHandle[Stream[T]]):
        super().__init__(
            stream,
            lambda s: step_n_times_and_return(
                Differentiate(StreamHandle(lambda: s)), s.current_time() + 1
            ),
            None,
        )

class StreamAddition(AbelianGroupOperation[Stream[T]]):
    group: AbelianGroupOperation[T]

    def __init__(self, group: AbelianGroupOperation[T]) -> None:
        self.group = group

    def add(self, a: Stream[T], b: Stream[T]) -> Stream[T]:
        handle_a = StreamHandle(lambda: a)
        handle_b = StreamHandle(lambda: b)

        lifted_group_add = LiftedGroupAdd(handle_a, handle_b)
        times = max(a.current_time(), b.current_time()) + 1
        step_n_times(lifted_group_add, times)

        return lifted_group_add.output()

    def neg(self, a: Stream[T]) -> Stream[T]:
        handle_a = StreamHandle(lambda: a)
        lifted_group_neg = LiftedGroupNegate(handle_a)
        times = a.current_time() + 1
        step_n_times(lifted_group_neg, times)

        return lifted_group_neg.output()

    def identity(self) -> Stream[T]:
        identity_stream = Stream(self.group)

        return identity_stream
        
def stream_introduction[T](value: T, group: AbelianGroupOperation[T]) -> Stream[T]:
    output_stream = Stream(group)
    output_stream.send(value)

    return output_stream


def stream_elimination[T](stream: Stream[T]) -> T:
    output_value = stream.group().identity()
    for value in stream:
        output_value = stream.group().add(output_value, value)

    return output_value


class LiftedStreamIntroduction(Lifted1[T, Stream[T]]):
    def __init__(self, stream: StreamHandle[T]) -> None:
        super().__init__(
            stream,
            lambda x: stream_introduction(x, stream.get().group()),
            StreamAddition(stream.get().group()), # type: ignore
        )


class LiftedStreamElimination(Lifted1[Stream[T], T]):
    def __init__(self, stream: StreamHandle[Stream[T]]) -> None:
        super().__init__(stream, lambda x: stream_elimination(x), None)

