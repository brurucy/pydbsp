from abc import abstractmethod
from types import NotImplementedType
from typing import Callable, Generic, Iterator, List, Optional, Protocol, TypeVar, cast

from pydbsp.core import AbelianGroupOperation

T = TypeVar("T")


class Stream(Generic[T]):
    timestamp: int
    inner: List[T]
    group_op: AbelianGroupOperation[T]

    def __init__(self, group_op: AbelianGroupOperation[T]) -> None:
        self.inner = []
        self.group_op = group_op
        self.timestamp = -1

    def send(self, element: T) -> None:
        self.inner.append(element)
        self.timestamp += 1

    def group(self) -> AbelianGroupOperation[T]:
        return self.group_op

    def current_time(self) -> int:
        return self.timestamp

    def __iter__(self) -> Iterator[T]:
        return self.inner.__iter__()

    def __repr__(self) -> str:
        return self.inner.__repr__()

    def __getitem__(self, timestamp: int) -> T:
        if timestamp <= self.current_time() and timestamp >= 0:
            return self.inner.__getitem__(timestamp)

        return self.group().identity()

    def latest(self) -> T:
        return self[self.current_time()]

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, Stream):
            return NotImplemented

        cast(Stream[T], other)

        self_timestamp = self.current_time()
        other_timestamp = other.current_time()

        if self_timestamp != other_timestamp:
            largest = max(self_timestamp, other_timestamp)

            for timestamp in range(largest + 1):
                self_val = self[timestamp]
                other_val = other[timestamp]  # type: ignore

                if self_val != other_val:
                    return False

            return True

        return self.inner == other.inner  # type: ignore


StreamReference = Callable[[], Stream[T]]


class StreamHandle(Generic[T]):
    ref: StreamReference[T]

    def __init__(self, stream_reference: StreamReference[T]) -> None:
        self.ref = stream_reference

    def get(self) -> Stream[T]:
        return self.ref()


class StreamAddition(AbelianGroupOperation[Stream[T]]):
    group: AbelianGroupOperation[T]

    def __init__(self, group: AbelianGroupOperation[T]) -> None:
        self.group = group

    def add(self, a: Stream[T], b: Stream[T]) -> Stream[T]:
        a_timestamp = a.current_time()
        b_timestamp = b.current_time()
        frontier = min(a_timestamp, b_timestamp)
        if a_timestamp == -1 or b_timestamp == -1:
            frontier = max(a_timestamp, b_timestamp)

        output_stream = Stream(self.group)
        for timestamp in range(frontier + 1):
            output_stream.send(self.group.add(a[timestamp], b[timestamp]))

        return output_stream

    def inner_group(self) -> AbelianGroupOperation[T]:
        return self.group

    def neg(self, a: Stream[T]) -> Stream[T]:
        frontier = a.current_time()
        output_stream = Stream(self.group)
        for timestamp in range(frontier + 1):
            output_stream.send(self.group.neg(a[timestamp]))

        return output_stream

    def identity(self) -> Stream[T]:
        identity_stream = Stream(self.group)

        return identity_stream


R = TypeVar("R")


class Operator(Protocol[T]):
    @abstractmethod
    def step(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def output_handle(self) -> StreamHandle[T]:
        raise NotImplementedError


def step_until_timestamp[T](operator: Operator[T], timestamp: int) -> None:
    current_timestamp = operator.output_handle().get().current_time()
    while current_timestamp < timestamp:
        operator.step()
        current_timestamp = operator.output_handle().get().current_time()


def step_until_timestamp_and_return[T](operator: Operator[T], timestamp: int) -> Stream[T]:
    step_until_timestamp(operator, timestamp)

    return operator.output_handle().get()


class UnaryOperator(Operator[R], Protocol[T, R]):
    input_stream_handle: StreamHandle[T]
    output_stream_handle: StreamHandle[R]

    def __init__(
        self,
        stream_handle: Optional[StreamHandle[T]],
        output_stream_group: Optional[AbelianGroupOperation[R]],
    ) -> None:
        if stream_handle is not None:
            self.set_input(stream_handle, output_stream_group)

    def set_input(
        self,
        stream_handle: StreamHandle[T],
        output_stream_group: Optional[AbelianGroupOperation[R]],
    ) -> None:
        self.input_stream_handle = stream_handle
        if output_stream_group is not None:
            output = Stream(output_stream_group)

            self.output_stream_handle = StreamHandle(lambda: output)
        else:
            output = cast(Stream[R], Stream(self.input_a().group()))

            self.output_stream_handle = StreamHandle(lambda: output)

    def output(self) -> Stream[R]:
        return self.output_stream_handle.get()

    def input_a(self) -> Stream[T]:
        return self.input_stream_handle.get()

    def output_handle(self) -> StreamHandle[R]:
        handle = StreamHandle(lambda: self.output())

        return handle


S = TypeVar("S")


class BinaryOperator(Operator[S], Protocol[T, R, S]):
    input_stream_handle_a: StreamHandle[T]
    input_stream_handle_b: StreamHandle[R]
    output_stream_handle: StreamHandle[S]

    def __init__(
        self,
        stream_a: Optional[StreamHandle[T]],
        stream_b: Optional[StreamHandle[R]],
        output_stream_group: Optional[AbelianGroupOperation[S]],
    ) -> None:
        if stream_a is not None:
            self.set_input_a(stream_a)

        if stream_b is not None:
            self.set_input_b(stream_b)

        if output_stream_group is not None:
            output = Stream(output_stream_group)

            self.set_output_stream(StreamHandle(lambda: output))

    def set_input_a(self, stream_handle_a: StreamHandle[T]) -> None:
        self.input_stream_handle_a = stream_handle_a
        output = cast(Stream[S], Stream(self.input_a().group()))

        self.set_output_stream(StreamHandle(lambda: output))

    def set_input_b(self, stream_handle_b: StreamHandle[R]) -> None:
        self.input_stream_handle_b = stream_handle_b

    def set_output_stream(self, output_stream_handle: StreamHandle[S]) -> None:
        self.output_stream_handle = output_stream_handle

    def output(self) -> Stream[S]:
        return self.output_stream_handle.get()

    def input_a(self) -> Stream[T]:
        return self.input_stream_handle_a.get()

    def input_b(self) -> Stream[R]:
        return self.input_stream_handle_b.get()

    def output_handle(self) -> StreamHandle[S]:
        handle = StreamHandle(lambda: self.output())

        return handle


F1 = Callable[[T], R]


class Lift1(UnaryOperator[T, R]):
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


F2 = Callable[[T, R], S]


class Lift2(BinaryOperator[T, R, S]):
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


class LiftedGroupAdd(Lift2[T, T, T]):
    def __init__(self, stream_a: StreamHandle[T], stream_b: Optional[StreamHandle[T]]):
        super().__init__(
            stream_a,
            stream_b,
            lambda x, y: stream_a.get().group().add(x, y),
            None,
        )


class LiftedGroupNegate(Lift1[T, T]):
    def __init__(self, stream: StreamHandle[T]):
        super().__init__(stream, lambda x: stream.get().group().neg(x), None)
