import sys
from abc import abstractmethod
from types import NotImplementedType
from typing import Callable, Generic, Iterator, List, Optional, Protocol, TypeVar, cast

from pydbsp.core import AbelianGroupOperation

T = TypeVar("T")

INFINITY = sys.maxsize


class Stream(Generic[T]):
    """
    Represents a stream of elements from an Abelian group.
    """

    timestamp: int
    inner: List[T]
    group_op: AbelianGroupOperation[T]

    def __init__(self, group_op: AbelianGroupOperation[T]) -> None:
        self.inner = []
        self.group_op = group_op
        self.timestamp = -1

    def send(self, element: T) -> None:
        """Adds an element to the stream and increments the timestamp."""
        self.inner.append(element)
        self.timestamp += 1

    def group(self) -> AbelianGroupOperation[T]:
        """Returns the Abelian group operation associated with this stream."""
        return self.group_op

    def current_time(self) -> int:
        """Returns the timestamp of the most recently arrived element."""
        return self.timestamp

    def __iter__(self) -> Iterator[T]:
        return self.inner.__iter__()

    def __repr__(self) -> str:
        return self.inner.__repr__()

    def __getitem__(self, timestamp: int) -> T:
        """Returns the element at the given timestamp."""
        if timestamp <= self.current_time() and timestamp >= 0:
            return self.inner.__getitem__(timestamp)

        return self.group().identity()

    def latest(self) -> T:
        """Returns the most recent element."""
        return self[self.current_time()]

    def __eq__(self, other: object) -> bool | NotImplementedType:
        """
        Compares this stream with another, considering all timestamps up to the latest.
        """
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
    """A handle to a stream, allowing lazy access."""

    ref: StreamReference[T]

    def __init__(self, stream_reference: StreamReference[T]) -> None:
        self.ref = stream_reference

    def get(self) -> Stream[T]:
        """Returns the referenced stream."""
        return self.ref()


R = TypeVar("R")


class Operator(Protocol[T]):
    @abstractmethod
    def step(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def output_handle(self) -> StreamHandle[T]:
        raise NotImplementedError


def step_until_fixpoint[T](operator: Operator[T]) -> None:
    while not operator.step():
        pass


def step_until_fixpoint_and_return[T](operator: Operator[T]) -> Stream[T]:
    step_until_fixpoint(operator)

    return operator.output_handle().get()


class UnaryOperator(Operator[R], Protocol[T, R]):
    """Base class for stream operators with a single input and output."""

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
        """Sets the input stream and initializes the output stream."""
        self.input_stream_handle = stream_handle
        is_input_identity = isinstance(self.input_stream_handle.get(), IdentityStream)

        if output_stream_group is not None:
            output = Stream(output_stream_group) if not is_input_identity else IdentityStream(output_stream_group)

            self.output_stream_handle = StreamHandle(lambda: output)
        else:
            output = cast(
                Stream[R],
                Stream(self.input_a().group()) if not is_input_identity else IdentityStream(self.input_a().group()),
            )

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
    """Base class for stream operators with two inputs and one output."""

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
        """Sets the first input stream and initializes the output stream."""
        self.input_stream_handle_a = stream_handle_a
        output = cast(Stream[S], Stream(self.input_a().group()))

        self.set_output_stream(StreamHandle(lambda: output))

    def set_input_b(self, stream_handle_b: StreamHandle[R]) -> None:
        """Sets the second input stream."""
        self.input_stream_handle_b = stream_handle_b

    def set_output_stream(self, output_stream_handle: StreamHandle[S]) -> None:
        """Sets the output stream handle."""
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
    """Lifts a unary function to operate on a stream."""

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
        """Applies the lifted function to the next element in the input stream."""
        output_timestamp = self.output().current_time()
        input_timestamp = self.input_a().current_time()
        if output_timestamp < input_timestamp and not isinstance(self.input_a(), IdentityStream):
            self.output().send(self.f1(self.input_a()[output_timestamp + 1]))

            return False

        return True


F2 = Callable[[T, R], S]


class Lift2(BinaryOperator[T, R, S]):
    """Lifts a binary function to operate on two streams where data arrives at
    different times."""

    previous_timestamp_a: int
    prevous_timestamp_b: int

    def __init__(
        self,
        stream_a: Optional[StreamHandle[T]],
        stream_b: Optional[StreamHandle[R]],
        f2: F2[T, R, S],
        output_stream_group: Optional[AbelianGroupOperation[S]],
    ) -> None:
        self.f2 = f2
        self.previous_timestamp_a = -1
        self.previous_timestamp_b = -1

        super().__init__(stream_a, stream_b, output_stream_group)

    def step(self) -> bool:
        """Applies the lifted function to the most recently arrived elements in both input streams."""
        current_timestamp_a = self.input_a().current_time()
        current_timestamp_b = self.input_b().current_time()
        new_a = False
        new_b = False

        a_is_identity = isinstance(self.input_a(), IdentityStream)
        b_is_identity = isinstance(self.input_b(), IdentityStream)
        if a_is_identity and b_is_identity:
            return True

        if current_timestamp_a > self.previous_timestamp_a or a_is_identity:
            new_a = True

        if current_timestamp_b > self.previous_timestamp_b or b_is_identity:
            new_b = True

        if new_a and new_b:
            current_timestamp_a = self.previous_timestamp_a + 1 if not a_is_identity else self.previous_timestamp_a
            a = self.input_a()[current_timestamp_a]
            self.previous_timestamp_a = current_timestamp_a if not a_is_identity else self.previous_timestamp_a

            current_timestamp_b = self.previous_timestamp_b + 1 if not b_is_identity else self.previous_timestamp_b
            b = self.input_b()[current_timestamp_b]
            self.previous_timestamp_b = current_timestamp_b if not b_is_identity else self.previous_timestamp_b

            application = self.f2(a, b)
            self.output().send(application)

            return False

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


class IdentityStream(Stream[T]):
    def __getitem__(self, timestamp: int) -> T:
        return self.group().identity()

    def send(self, element: T) -> None:
        raise ValueError("Cannot send to the identity stream")

    def current_time(self) -> int:
        return INFINITY

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stream):
            return NotImplemented
        if isinstance(other, IdentityStream):
            return True

        return all(self[i] == other[i] for i in range(other.current_time() + 1))  # type: ignore


class StreamAddition(AbelianGroupOperation[Stream[T]]):
    """Defines addition for streams by lifting their underlying group's addition."""

    group: AbelianGroupOperation[T]

    def __init__(self, group: AbelianGroupOperation[T]) -> None:
        self.group = group

    def add(self, a: Stream[T], b: Stream[T]) -> Stream[T]:
        """Adds two streams element-wise."""
        handle_a = StreamHandle(lambda: a)
        handle_b = StreamHandle(lambda: b)

        lifted_group_add = LiftedGroupAdd(handle_a, handle_b)

        return step_until_fixpoint_and_return(lifted_group_add)

    def inner_group(self) -> AbelianGroupOperation[T]:
        """Returns the underlying group operation."""
        return self.group

    def neg(self, a: Stream[T]) -> Stream[T]:
        """Negates a stream element-wise."""
        handle_a = StreamHandle(lambda: a)
        lifted_group_neg = LiftedGroupNegate(handle_a)

        return step_until_fixpoint_and_return(lifted_group_neg)

    def identity(self) -> Stream[T]:
        """
        Returns an identity stream for the addition operation that CANNOT have data streamed into it.

        What you are most likely after is the empty stream, instantiated through the Stream constructor.
        """
        identity_stream = IdentityStream(self.group)

        return identity_stream
