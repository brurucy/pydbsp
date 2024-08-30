from typing import Callable, Dict, TypeVar

from pydbsp.zset import ZSet

T = TypeVar("T")

Cmp = Callable[[T], bool]


def select[T](zset: ZSet[T], p: Cmp[T]) -> ZSet[T]:
    return ZSet({k: v for k, v in zset.items() if p(k)})


R = TypeVar("R")
Projection = Callable[[T], R]


def project[T, R](zset: ZSet[T], f: Projection[T, R]) -> ZSet[R]:
    output: Dict[R, int] = {}
    for value, weight in zset.items():
        fvalue = f(value)
        if fvalue not in output:
            output[fvalue] = weight
        else:
            output[fvalue] += weight

    return ZSet(output)
