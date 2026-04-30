from typing import Callable

from pydbsp.stream import Lift1, Stream
from pydbsp.zset import ZSet, ZSetAddition
from pydbsp.zset.functions.linear import project, select


def Select[V, T: tuple[int, ...]](
    base: Stream[ZSet[V], T],
    pred: Callable[[V], bool],
) -> Stream[ZSet[V], T]:
    """Pointwise selection on a flat product lattice — ``Lift1``
    operates pointwise at every lattice point."""
    return Lift1(base, lambda z: select(z, pred), base.group)


def Project[A, B, T: tuple[int, ...]](
    base: Stream[ZSet[A], T],
    f: Callable[[A], B],
    out_group: ZSetAddition[B],
) -> Stream[ZSet[B], T]:
    """Pointwise projection on a flat product lattice."""
    return Lift1(base, lambda z: project(z, f), out_group)
