from typing import Dict, Iterable, Tuple

from pydbsp.core import AbelianGroupOperation


class ZSet[T]:
    """
    Represents a Z-set, a generalization of multisets with integer weights.
    Elements can have positive, negative, or zero weights.

    A Z-Set whose elements have all weight one can be interpreted as a set. One where
    all are strictly positive is a bag, and one where they are either one or -1 is a diff.
    """

    inner: Dict[T, int]

    def __init__(self, values: Dict[T, int]) -> None:
        self.inner = values

    def items(self) -> Iterable[Tuple[T, int]]:
        """Returns an iterable of (element, weight) pairs."""
        return self.inner.items()

    def __repr__(self) -> str:
        return self.inner.__repr__()

    def __eq__(self, other: object) -> bool:
        """
        Two Z-sets are equal if they have the same elements with the same weight.
        """
        if not isinstance(other, ZSet):
            return False

        return self.inner == other.inner  # type: ignore

    def __contains__(self, item: T) -> bool:
        """An item is in the Z-set if it has non-zero weight."""
        return self.inner.__contains__(item)

    def __getitem__(self, item: T) -> int:
        """Returns the weight of an item (0 if not present)."""
        if item not in self:
            return 0

        return self.inner[item]

    def is_identity(self) -> bool:
        return len(self.inner) == 0

    def __setitem__(self, key: T, value: int) -> None:
        self.inner[key] = value
    
    def __hash__(self) -> int:
        return hash(frozenset(self.inner.items()))


class ZSetAddition[T](AbelianGroupOperation[ZSet[T]]):
    """
    Defines addition operation for Z-sets, forming an Abelian group.
    """

    def add(self, a: ZSet[T], b: ZSet[T]) -> ZSet[T]:
        """
        Adds two Z-sets by summing weights of common elements.
        Elements with resulting zero weight are removed.
        """
        result = a.inner | b.inner

        for k, v in b.inner.items():
            if k in a.inner:
                new_weight = a.inner[k] + v
                if new_weight == 0:
                    del result[k]
                else:
                    result[k] = new_weight

        return ZSet(result)

    def neg(self, a: ZSet[T]) -> ZSet[T]:
        """Returns the inverse of a Z-set by negating all weights."""
        return ZSet({k: v * -1 for k, v in a.inner.items()})

    def identity(self) -> ZSet[T]:
        """Returns the empty Z-set."""
        return ZSet({})
