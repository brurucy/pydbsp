from typing import Callable

from pydbsp.zset import ZSet


def select[T](z: ZSet[T], pred: Callable[[T], bool]) -> ZSet[T]:
    return ZSet({k: v for k, v in z.inner.items() if pred(k)})


def project[A, B](z: ZSet[A], f: Callable[[A], B]) -> ZSet[B]:
    out: dict[B, int] = {}
    for k, v in z.inner.items():
        key = f(k)
        out[key] = out.get(key, 0) + v
    return ZSet({k: v for k, v in out.items() if v != 0})
