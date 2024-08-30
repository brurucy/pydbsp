from pydbsp.core import AbelianGroupOperation
from pydbsp.stream import Stream


def stream_introduction[T](value: T, group: AbelianGroupOperation[T]) -> Stream[T]:
    output_stream = Stream(group)
    output_stream.send(value)

    return output_stream


def stream_elimination[T](stream: Stream[T]) -> T:
    output_value = stream.group().identity()
    for value in stream:
        output_value = stream.group().add(output_value, value)

    return output_value
