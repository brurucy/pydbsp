"""Indexed Z-set: a Z-set carrying a side-car B-tree index over some
derived key ``I = indexer(T)``. The abelian group semantics are
identical to ``ZSet`` — the index is an auxiliary structure that
accelerates sort-merge joins.

``IndexedZSetAddition`` is parameterised by the ``indexer`` function;
the indexer is part of the group instance, not the carrier.
"""

from bisect import bisect_right, insort
from collections.abc import Callable, Generator, Iterator
from typing import TypeVar

from pydbsp.core import AbelianGroupOperation
from pydbsp.zset import ZSetAddition


class AppendOnlySpine[T]:
    """Flat B-tree for sorted iteration. Ported from v0.6.0 — a
    leaf-level chunked list keyed by per-chunk max.
    """

    _len: int
    _load: int
    _lists: list[list[T]]
    _maxes: list[T]

    def __init__(self) -> None:
        self._len = 0
        self._load = 1024
        self._lists = []
        self._maxes = []

    def __len__(self) -> int:
        return self._len

    def add(self, value: T) -> None:
        if self._maxes:
            pos = bisect_right(self._maxes, value)  # type: ignore[type-var]
            if pos == len(self._maxes):
                pos -= 1
                self._lists[pos].append(value)
                self._maxes[pos] = value
            else:
                insort(self._lists[pos], value)  # type: ignore[type-var]
            self._expand(pos)
        else:
            self._lists.append([value])
            self._maxes.append(value)
        self._len += 1

    def __iter__(self) -> Generator[T, None, None]:
        for sublist in self._lists:
            yield from sublist

    def _expand(self, pos: int) -> None:
        if len(self._lists[pos]) > (self._load << 1):
            chunk = self._lists[pos]
            half = chunk[self._load :]
            del chunk[self._load :]
            self._maxes[pos] = chunk[-1]
            self._lists.insert(pos + 1, half)
            self._maxes.insert(pos + 1, half[-1])


def sort_merge_keys[K](a: AppendOnlySpine[K], b: AppendOnlySpine[K]) -> Iterator[K]:
    """Yield keys present in both spines (in sorted order). Keys are
    consumed one at a time — callers handle multiplicity."""
    it_a: Iterator[K] = iter(a)
    it_b: Iterator[K] = iter(b)
    try:
        x = next(it_a)
        y = next(it_b)
        while True:
            if x < y:  # type: ignore[operator]
                x = next(it_a)
            elif y < x:  # type: ignore[operator]
                y = next(it_b)
            else:
                yield x
                x = next(it_a)
                y = next(it_b)
    except StopIteration:
        return


I = TypeVar("I")


class IndexedZSet[I, T]:
    """Z-set with an index on ``indexer(value)``. The ``inner`` dict
    is the Z-set carrier; ``index_to_value`` and ``index`` are the
    side-car lookup structures maintained on insertion.
    """

    inner: dict[T, int]
    index_to_value: dict[I, set[T]]
    indexer: Callable[[T], I]
    index: AppendOnlySpine[I]

    def __init__(self, values: dict[T, int], indexer: Callable[[T], I]) -> None:
        self.inner = {}
        self.index_to_value = {}
        self.index = AppendOnlySpine()
        self.indexer = indexer
        for k, v in values.items():
            self[k] = v

    @classmethod
    def _from_parts(
        cls,
        inner: dict[T, int],
        index_to_value: dict[I, set[T]],
        index: AppendOnlySpine[I],
        indexer: Callable[[T], I],
    ) -> "IndexedZSet[I, T]":
        out: IndexedZSet[I, T] = cls.__new__(cls)
        out.inner = inner
        out.index_to_value = index_to_value
        out.index = index
        out.indexer = indexer
        return out

    def __repr__(self) -> str:
        return self.inner.__repr__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexedZSet):
            return False
        return self.inner == other.inner

    def __contains__(self, item: T) -> bool:
        return item in self.inner

    def __getitem__(self, item: T) -> int:
        return self.inner.get(item, 0)

    def __setitem__(self, key: T, weight: int) -> None:
        already = key in self.inner
        self.inner[key] = weight
        if already:
            return
        idx = self.indexer(key)
        bucket = self.index_to_value.get(idx)
        if bucket is None:
            self.index_to_value[idx] = {key}
            self.index.add(idx)
        else:
            bucket.add(key)


class IndexedZSetAddition[I, T](AbelianGroupOperation[IndexedZSet[I, T]]):
    """Abelian group over ``IndexedZSet[I, T]``. Semantics match
    ``ZSetAddition`` — weight-wise add, zero-weight elimination.
    The indexer is captured in the group instance; ``add`` returns
    a fresh indexed zset with a rebuilt index.
    """

    inner_group: ZSetAddition[T]
    indexer: Callable[[T], I]

    def __init__(self, inner_group: ZSetAddition[T], indexer: Callable[[T], I]) -> None:
        self.inner_group = inner_group
        self.indexer = indexer

    def add(self, a: IndexedZSet[I, T], b: IndexedZSet[I, T]) -> IndexedZSet[I, T]:
        # Light COW:
        #   ``inner`` always forks (small, changes every add).
        #   ``index_to_value`` — shallow-copy top-level dict on first
        #   bucket mutation; per-bucket forks only when touched.
        #   ``index`` spine — deep-clone only when a new index key
        #   needs appending (``b``'s keys whose ``indexer(k)`` is
        #   absent from ``a``'s bucket map).
        # Relies on the convention that results are read-only after
        # construction (no post-hoc ``__setitem__``).
        if not b.inner:
            return a
        if not a.inner:
            return b
        inner: dict[T, int] = dict(a.inner)
        i2v: dict[I, set[T]] | None = None
        spine: AppendOnlySpine[I] | None = None
        forked_buckets: set[I] = set()
        for k, v in b.inner.items():
            prev = inner.get(k, 0)
            w = prev + v
            if w == 0:
                inner.pop(k, None)
            else:
                inner[k] = w
            # Index transitions: prev==0,w!=0 → add; prev!=0,w==0 → remove;
            # prev!=0,w!=0 → no change (key already indexed).
            if prev == 0 and w == 0:
                continue
            if prev != 0 and w != 0:
                continue
            idx = self.indexer(k)
            if i2v is None:
                i2v = dict(a.index_to_value)
            if prev == 0:
                # New key: add to bucket/spine.
                bucket = i2v.get(idx)
                if bucket is None:
                    if spine is None:
                        spine = AppendOnlySpine()
                        spine._len = a.index._len
                        spine._load = a.index._load
                        spine._lists = [list(chunk) for chunk in a.index._lists]
                        spine._maxes = list(a.index._maxes)
                    spine.add(idx)
                    i2v[idx] = {k}
                    forked_buckets.add(idx)
                else:
                    if idx not in forked_buckets:
                        bucket = set(bucket)
                        i2v[idx] = bucket
                        forked_buckets.add(idx)
                    bucket.add(k)
            else:
                # Cancellation (w==0): remove k from the bucket; if the
                # bucket becomes empty, drop it from ``i2v``. ``spine`` is
                # append-only so orphan idx entries can remain — downstream
                # ``sort_merge_keys`` tolerates them because buckets are
                # looked up via ``index_to_value`` which we keep clean.
                bucket = i2v.get(idx)
                if bucket is not None:
                    if idx not in forked_buckets:
                        bucket = set(bucket)
                        i2v[idx] = bucket
                        forked_buckets.add(idx)
                    bucket.discard(k)
                    if not bucket:
                        del i2v[idx]
        return IndexedZSet._from_parts(
            inner,
            a.index_to_value if i2v is None else i2v,
            a.index if spine is None else spine,
            self.indexer,
        )

    def neg(self, a: IndexedZSet[I, T]) -> IndexedZSet[I, T]:
        # Keys are unchanged — share ``index_to_value`` and ``index``.
        return IndexedZSet._from_parts(
            {k: -v for k, v in a.inner.items()},
            a.index_to_value,
            a.index,
            self.indexer,
        )

    def identity(self) -> IndexedZSet[I, T]:
        return IndexedZSet({}, self.indexer)
