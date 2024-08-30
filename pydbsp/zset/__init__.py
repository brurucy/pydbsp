from typing import Dict, Generic, Iterable, Tuple, TypeVar

from pydbsp.core import AbelianGroupOperation

T = TypeVar("T")


class ZSet(Generic[T]):
    inner: Dict[T, int]

    def __init__(self, values: Dict[T, int]) -> None:
        self.inner = values

    def items(self) -> Iterable[Tuple[T, int]]:
        return self.inner.items()

    def __repr__(self) -> str:
        return self.inner.__repr__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZSet):
            return False

        return self.inner == other.inner  # type: ignore

    def __contains__(self, item: T) -> bool:
        return self.inner.__contains__(item)

    def __getitem__(self, item: T) -> int:
        if item not in self:
            return 0

        return self.inner[item]

    def __setitem__(self, key: T, value: int) -> None:
        self.inner[key] = value


class ZSetAddition(Generic[T], AbelianGroupOperation[ZSet[T]]):
    def add(self, a: ZSet[T], b: ZSet[T]) -> ZSet[T]:
        c = {k: v for k, v in a.inner.items() if v != 0}
        for k, v in b.inner.items():
            if k in c:
                new_weight = c[k] + v
                if new_weight != 0:
                    c[k] = new_weight
                else:
                    del c[k]
            else:
                c[k] = v

        return ZSet(c)

    def neg(self, a: ZSet[T]) -> ZSet[T]:
        return ZSet({k: v * -1 for k, v in a.inner.items()})

    def identity(self) -> ZSet[T]:
        return ZSet({})
