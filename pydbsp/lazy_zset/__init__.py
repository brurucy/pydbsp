from typing import Iterator, List

from pydbsp.core import AbelianGroupOperation
from pydbsp.zset import ZSet, ZSetAddition


class LazyZSet[T]:
    inner: List[ZSet[T]]

    def __init__(self, zsets: List[ZSet[T]]) -> None:
        self.inner = zsets

    def __iter__(self) -> Iterator[ZSet[T]]:
        """
        Returns an iterable of ZSets
        """
        return self.inner.__iter__()

    def __repr__(self) -> str:
        return self.inner.__repr__()

    def is_identity(self) -> bool:
        """
        Returns whether all ZSets are empty
        """
        return all(map(lambda zset: zset.is_identity(), self.inner))

    def coalesce(self) -> ZSet[T]:
        """
        Coalesces the LazyZSet into a single ZSet and returns it. This is a very expensive operation.

        If there is only a single ZSet however, it is cheap.
        """
        if len(self.inner) == 1:
            return self.inner[0]

        group = ZSetAddition[T]()
        coalesced_lazy_zset = group.identity()
        for zset in self:
            coalesced_lazy_zset = group.add(coalesced_lazy_zset, zset)

        return coalesced_lazy_zset

    def __eq__(self, other: object) -> bool:
        """
        Two LazyZSets are equal if their coalesced form match.

        This is a very expensive operation, as it requires coalescing all ZSets. It
        is however cheap to know whether a LazyZSet is the identity element, or if equality
        is being done against a identity element.
        """
        if not isinstance(other, LazyZSet):
            return False

        is_self_identity = self.is_identity()
        is_other_identity = other.is_identity()

        if is_self_identity and is_other_identity:
            return True
        elif is_self_identity and not is_other_identity:
            return False

        coalesced_self = self.coalesce()
        coalesced_other = other.coalesce()  # type: ignore

        return coalesced_self == coalesced_other

    def __contains__(self, item: T) -> bool:
        """
        An item is in the lazy Z-set if the sum of its weight across all ZSets contained within it is greater than zero.
        """
        return self[item] > 0

    def __getitem__(self, item: T) -> int:
        """
        Returns the sum of the weight of a given item across all ZSets.
        """
        weight = 0

        for zset in self.inner:
            weight += zset[item]

        return weight


class LazyZSetAddition[T](AbelianGroupOperation[LazyZSet[T]]):
    def add(self, a: LazyZSet[T], b: LazyZSet[T]) -> LazyZSet[T]:
        if a.is_identity():
            return b

        if b.is_identity():
            return a

        result = LazyZSet(a.inner + b.inner)
        return result

    def neg(self, a: LazyZSet[T]) -> LazyZSet[T]:
        if a.is_identity():
            return a

        group = ZSetAddition[T]()
        return LazyZSet([group.neg(zset) for zset in a])

    def identity(self) -> LazyZSet[T]:
        return LazyZSet([])
