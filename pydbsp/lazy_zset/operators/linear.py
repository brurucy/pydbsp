from typing import Optional

from pydbsp.lazy_zset import LazyZSet
from pydbsp.lazy_zset.functions.linear import Cmp, Projection, coalesce, project, select
from pydbsp.stream import Lift1, Stream, StreamHandle, step_until_fixpoint_and_return


class LiftedCoalesce[T](Lift1[LazyZSet[T], LazyZSet[T]]):
    def __init__(self, stream: Optional[StreamHandle[LazyZSet[T]]]):
        super().__init__(stream, lambda z: coalesce(z), None)


class LiftedLiftedCoalesce[T](Lift1[Stream[LazyZSet[T]], Stream[LazyZSet[T]]]):
    def __init__(self, stream: Optional[StreamHandle[Stream[LazyZSet[T]]]]):
        super().__init__(
            stream, lambda x: step_until_fixpoint_and_return(LiftedCoalesce(StreamHandle(lambda: x))), None
        )


class LiftedSelect[T](Lift1[LazyZSet[T], LazyZSet[T]]):
    def __init__(self, stream: Optional[StreamHandle[LazyZSet[T]]], p: Cmp[T]):
        super().__init__(stream, lambda z: select(z, p), None)


class LiftedLiftedSelect[T](Lift1[Stream[LazyZSet[T]], Stream[LazyZSet[T]]]):
    def __init__(self, stream: Optional[StreamHandle[Stream[LazyZSet[T]]]], p: Cmp[T]):
        super().__init__(
            stream, lambda x: step_until_fixpoint_and_return(LiftedSelect(StreamHandle(lambda: x), p)), None
        )


class LiftedProject[T, R](Lift1[LazyZSet[T], LazyZSet[R]]):
    def __init__(self, stream: Optional[StreamHandle[LazyZSet[T]]], f: Projection[T, R]):
        super().__init__(stream, lambda z: project(z, f), None)


class LiftedLiftedProject[T, R](Lift1[Stream[LazyZSet[T]], Stream[LazyZSet[R]]]):
    def __init__(self, stream: Optional[StreamHandle[Stream[LazyZSet[T]]]], f: Projection[T, R]):
        super().__init__(
            stream, lambda x: step_until_fixpoint_and_return(LiftedProject(StreamHandle(lambda: x), f)), None
        )
