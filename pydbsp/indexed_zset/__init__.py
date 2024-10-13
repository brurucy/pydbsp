from bisect import bisect_right, insort
from typing import Callable, Dict, Generator, Iterable, Iterator, List, Set, Tuple, TypeVar

from pydbsp.core import AbelianGroupOperation
from pydbsp.zset import ZSetAddition

T = TypeVar("T")


class AppendOnlySpine[T]:
    """
    An append-only flat B-Tree. Borrowed from `https://github.com/grantjenks/python-sortedcontainers` and `https://github.com/brurucy/indexset`. Only for internal use.
    """

    _len: int
    _load: int
    _lists: List[List[T]]
    _maxes: List[T]

    def __init__(self):
        self._len = 0
        self._load = 1024
        self._lists = []
        self._maxes = []

    def __len__(self) -> int:
        return self._len

    def add(self, value: T):
        _lists = self._lists
        _maxes = self._maxes

        if _maxes:
            pos = bisect_right(_maxes, value)  # type: ignore

            if pos == len(_maxes):
                pos -= 1
                _lists[pos].append(value)
                _maxes[pos] = value
            else:
                insort(_lists[pos], value)  # type: ignore

            self._expand(pos)
        else:
            _lists.append([value])
            _maxes.append(value)

        self._len += 1

    def __iter__(self) -> Generator[T, None, None]:
        for sublist in self._lists:
            yield from sublist

    def _expand(self, pos: int):
        _load = self._load
        _lists = self._lists

        if len(_lists[pos]) > (_load << 1):
            _maxes = self._maxes

            _lists_pos = _lists[pos]
            half = _lists_pos[_load:]
            del _lists_pos[_load:]
            _maxes[pos] = _lists_pos[-1]

            _lists.insert(pos + 1, half)
            _maxes.insert(pos + 1, half[-1])


def sort_merge_join[T](spine1: AppendOnlySpine[T], spine2: AppendOnlySpine[T]) -> Iterator[T]:
    iter1: Generator[T] = iter(spine1)
    iter2: Generator[T] = iter(spine2)

    try:
        item1 = next(iter1)
        item2 = next(iter2)

        while True:
            if item1 < item2:  # type: ignore
                item1 = next(iter1)
            elif item2 < item1:  # type: ignore
                item2 = next(iter2)
            else:
                yield item1
                item1 = next(iter1)
                item2 = next(iter2)

    except StopIteration:
        return


I = TypeVar("I")
Indexer = Callable[[T], I]


class IndexedZSet[I, T]:
    """
    Represents a Z-set, with a B-Tree index. See :func:`~pydbsp.zset.ZSet`.
    """

    inner: Dict[T, int]
    index_to_value: Dict[I, Set[T]]
    indexer: Callable[[T], I]
    index: AppendOnlySpine[I]

    def __init__(self, values: Dict[T, int], indexer: Indexer[T, I]) -> None:
        self.inner = values
        self.index_to_value = {}
        self.index = AppendOnlySpine()
        self.indexer = indexer

        for key, value in self.inner.items():
            self[key] = value

    def items(self) -> Iterable[Tuple[T, int]]:
        return self.inner.items()

    def __repr__(self) -> str:
        return self.inner.__repr__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexedZSet):
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
        indexed_value = self.indexer(key)
        if indexed_value in self.index_to_value:
            self.index_to_value[indexed_value].add(key)
        else:
            self.index_to_value[indexed_value] = {key}

        self.index.add(indexed_value)


class IndexedZSetAddition[I, T](AbelianGroupOperation[IndexedZSet[I, T]]):
    inner_group: ZSetAddition[T]
    indexer: Indexer[T, I]

    def __init__(self, inner_group: ZSetAddition[T], indexer: Indexer[T, I]) -> None:
        self.inner_group = inner_group
        self.indexer = indexer

    def add(self, a: IndexedZSet[I, T], b: IndexedZSet[I, T]) -> IndexedZSet[I, T]:
        c = a.inner | b.inner

        for k, v in b.inner.items():
            if k in a.inner:
                new_weight = a.inner[k] + v
                if new_weight == 0:
                    del c[k]
                else:
                    c[k] = new_weight

        return IndexedZSet(c, self.indexer)

    def neg(self, a: IndexedZSet[I, T]) -> IndexedZSet[I, T]:
        empty_dict: Dict[T, int] = {}
        b = IndexedZSet(empty_dict, a.indexer)
        b.index = a.index
        b.index_to_value = a.index_to_value
        b.inner = {k: v * -1 for k, v in a.inner.items()}

        return b

    def identity(self) -> IndexedZSet[I, T]:
        empty_dict: Dict[T, int] = {}

        return IndexedZSet(empty_dict, self.indexer)
