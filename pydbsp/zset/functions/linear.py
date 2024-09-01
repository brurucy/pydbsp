from typing import Callable, Dict, TypeVar

from pydbsp.zset import ZSet

T = TypeVar("T")

Cmp = Callable[[T], bool]


def select[T](zset: ZSet[T], p: Cmp[T]) -> ZSet[T]:
    """Filters a Z-set based on a predicate function."""
    return ZSet({k: v for k, v in zset.items() if p(k)})


R = TypeVar("R")
Projection = Callable[[T], R]


def project[T, R](zset: ZSet[T], f: Projection[T, R]) -> ZSet[R]:
    """
    Projects a Z-set to a new Z-set by applying a function to each element.

    Returns a new Z-set where each element is the result of applying f to an element
    of the input Z-set. Weights of elements mapping to the same value are summed.
    """
    output: Dict[R, int] = {}
    for value, weight in zset.items():
        fvalue = f(value)
        if fvalue not in output:
            output[fvalue] = weight
        else:
            output[fvalue] += weight

    return ZSet(output)
