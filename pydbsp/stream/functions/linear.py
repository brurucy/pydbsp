from pydbsp.core import AbelianGroupOperation
from pydbsp.stream import Stream


def stream_introduction[T](value: T, group: AbelianGroupOperation[T]) -> Stream[T]:
    """
    Creates a stream out of a single element and an associated abelian group.

    Args:
        value (T): The value to be introduced into the stream.
        group (AbelianGroupOperation[T]): The group operation for the stream.

    Returns:
        Stream[T]: A new stream containing the single value.
    """
    output_stream = Stream(group)
    output_stream.send(value)

    return output_stream


def stream_elimination[T](stream: Stream[T]) -> T:
    """
    Sums all elements in a stream using the stream's group operation.

    Args:
        stream (Stream[T]): The input stream to be "squashed".

    Returns:
        T: The result of combining all elements in the stream.
    """
    output_value = stream.group().identity()
    for value in stream:
        output_value = stream.group().add(output_value, value)

    return output_value
