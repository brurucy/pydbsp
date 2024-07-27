from types import NotImplementedType
from typing import TypeVar, Generic, List, Iterator, Callable, cast 
from algebra import AbelianGroupOperation 

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
                other_val = other[timestamp] # type: ignore

                if self_val != other_val:
                    return False

            return True

        return self.inner == other.inner # type: ignore


StreamReference = Callable[[], Stream[T]]


class StreamHandle(Generic[T]):
    ref: StreamReference[T]

    def __init__(self, stream_reference: StreamReference[T]) -> None:
        self.ref = stream_reference

    def get(self) -> Stream[T]:
        return self.ref()

