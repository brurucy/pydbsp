from pydbsp.lazy_zset import LazyZSet
from pydbsp.zset.functions.linear import Cmp, Projection
from pydbsp.zset.functions.linear import project as zset_project
from pydbsp.zset.functions.linear import select as zset_select


def select[T](lazy_zset: LazyZSet[T], p: Cmp[T]) -> LazyZSet[T]:
    """Filters the given Lazy Z-set based on a predicate function."""
    return LazyZSet([zset_select(zset, p) for zset in lazy_zset])


def project[T, R](lazy_zset: LazyZSet[T], f: Projection[T, R]) -> LazyZSet[R]:
    """
    Projects a Lazy Z-set unto a new Lazy Z-set by applying a function to each element.

    Returns a new Lazy Z-set where each element is the result of applying f to an element
    of the input Lazy Z-set. Weights of elements mapping to the same value are summed.
    """
    return LazyZSet([zset_project(zset, f) for zset in lazy_zset])


def coalesce[T](lazy_zset: LazyZSet[T]) -> LazyZSet[T]:
    return LazyZSet([lazy_zset.coalesce()])
