from typing import List

from pydbsp.indexed_zset import AppendOnlySpine, sort_merge_join


def create_spine_from_list[T](values: List[T]) -> AppendOnlySpine[T]:
    spine: AppendOnlySpine[T] = AppendOnlySpine()
    for value in values:
        spine.add(value)

    return spine


def test_empty_spines():
    spine1: AppendOnlySpine[int] = create_spine_from_list([])
    spine2: AppendOnlySpine[int] = create_spine_from_list([])
    assert list(sort_merge_join(spine1, spine2)) == []


def test_one_empty_spine():
    spine1: AppendOnlySpine[int] = create_spine_from_list([1, 2, 3])
    spine2: AppendOnlySpine[int] = create_spine_from_list([])
    assert list(sort_merge_join(spine1, spine2)) == []
    assert list(sort_merge_join(spine2, spine1)) == []


def test_no_overlap():
    spine1: AppendOnlySpine[int] = create_spine_from_list([1, 2, 3])
    spine2: AppendOnlySpine[int] = create_spine_from_list([4, 5, 6])
    assert list(sort_merge_join(spine1, spine2)) == []


def test_full_overlap():
    spine1: AppendOnlySpine[int] = create_spine_from_list([1, 2, 3])
    spine2: AppendOnlySpine[int] = create_spine_from_list([1, 2, 3])
    expected = [1, 2, 3]
    assert list(sort_merge_join(spine1, spine2)) == expected


def test_partial_overlap():
    spine1: AppendOnlySpine[int] = create_spine_from_list([1, 2, 3, 4])
    spine2: AppendOnlySpine[int] = create_spine_from_list([3, 4, 5, 6])
    expected = [3, 4]
    assert list(sort_merge_join(spine1, spine2)) == expected


def test_duplicate_values():
    spine1: AppendOnlySpine[int] = create_spine_from_list([1, 2, 2, 3])
    spine2: AppendOnlySpine[int] = create_spine_from_list([2, 2, 3, 4])
    expected = [2, 2, 3]
    assert list(sort_merge_join(spine1, spine2)) == expected


def test_large_spines():
    spine1: AppendOnlySpine[int] = create_spine_from_list(reversed(range(0, 10000, 2)))  # type: ignore
    spine2: AppendOnlySpine[int] = create_spine_from_list(range(1, 10000, 2))  # type: ignore
    assert list(sort_merge_join(spine1, spine2)) == []

    spine3: AppendOnlySpine[int] = create_spine_from_list(reversed(range(0, 10000)))  # type: ignore
    spine4: AppendOnlySpine[int] = create_spine_from_list(range(5000, 15000))  # type: ignore
    expected = list(range(5000, 10000))
    assert list(sort_merge_join(spine3, spine4)) == expected
