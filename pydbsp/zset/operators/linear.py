from typing import Optional, TypeVar

from pydbsp.stream import Lift1, Stream, StreamHandle, step_until_fixpoint_and_return
from pydbsp.zset import ZSet
from pydbsp.zset.functions.linear import Cmp, Projection, project, select

T = TypeVar("T")


class LiftedSelect(Lift1[ZSet[T], ZSet[T]]):
    def __init__(self, stream: Optional[StreamHandle[ZSet[T]]], p: Cmp[T]):
        super().__init__(stream, lambda z: select(z, p), None)


class LiftedLiftedSelect(Lift1[Stream[ZSet[T]], Stream[ZSet[T]]]):
    def __init__(self, stream: Optional[StreamHandle[Stream[ZSet[T]]]], p: Cmp[T]):
        super().__init__(
            stream, lambda x: step_until_fixpoint_and_return(LiftedSelect(StreamHandle(lambda: x), p)), None
        )


R = TypeVar("R")


class LiftedProject(Lift1[ZSet[T], ZSet[R]]):
    def __init__(self, stream: Optional[StreamHandle[ZSet[T]]], f: Projection[T, R]):
        super().__init__(stream, lambda z: project(z, f), None)


class LiftedLiftedProject(Lift1[Stream[ZSet[T]], Stream[ZSet[R]]]):
    def __init__(self, stream: Optional[StreamHandle[Stream[ZSet[T]]]], f: Projection[T, R]):
        super().__init__(
            stream, lambda x: step_until_fixpoint_and_return(LiftedProject(StreamHandle(lambda: x), f)), None
        )
