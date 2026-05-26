"""Tests for ``pydbsp.storage`` — the (NodeId, Time) value store."""

from __future__ import annotations

import pytest

from typing import cast

from pydbsp.progress import Time
from pydbsp.storage import DictStorage, Storage
from pydbsp.core import Antichain, Time1D, Time2D


def _ac1(elements: list[tuple[int]] | None = None) -> Antichain[Time]:
    return cast(Antichain[Time], Antichain(Time1D, elements or []))


def _ac2(elements: list[tuple[int, int]]) -> Antichain[Time]:
    return cast(Antichain[Time], Antichain(Time2D, elements))


# ---- write / read / contains -----------------------------------------------


def test_write_then_read_returns_stored_value() -> None:
    s = DictStorage()
    s.write(0, (5,), 42)
    assert s.read(0, (5,)) == 42


def test_write_overwrites() -> None:
    s = DictStorage()
    s.write(0, (5,), 1)
    s.write(0, (5,), 99)
    assert s.read(0, (5,)) == 99


def test_read_with_default_for_absent_entry() -> None:
    s = DictStorage()
    assert s.read(0, (5,), default=0) == 0


def test_read_raises_keyerror_without_default() -> None:
    s = DictStorage()
    with pytest.raises(KeyError):
        s.read(0, (5,))


def test_read_distinguishes_none_value_from_absent() -> None:
    """The sentinel-based default ensures a stored ``None`` is
    returned as-is, not confused with 'not present'."""
    s = DictStorage()
    s.write(0, (5,), None)
    assert s.read(0, (5,), default="DEFAULT") is None
    # Absent — returns the default.
    assert s.read(0, (7,), default="DEFAULT") == "DEFAULT"


def test_contains_reports_presence() -> None:
    s = DictStorage()
    s.write(0, (5,), 42)
    assert s.contains(0, (5,))
    assert not s.contains(0, (3,))
    assert not s.contains(1, (5,))


# ---- Multiple nodes are independent ----------------------------------------


def test_nodes_are_isolated() -> None:
    s = DictStorage()
    s.write(0, (5,), "node-0")
    s.write(1, (5,), "node-1")
    assert s.read(0, (5,)) == "node-0"
    assert s.read(1, (5,)) == "node-1"


# ---- Eviction --------------------------------------------------------------


def test_evict_drops_one_entry() -> None:
    s = DictStorage()
    s.write(0, (5,), 42)
    s.write(0, (7,), 99)
    s.evict(0, (5,))
    assert not s.contains(0, (5,))
    assert s.contains(0, (7,))


def test_evict_noop_for_absent_entry() -> None:
    s = DictStorage()
    s.evict(0, (5,))  # should not raise
    s.write(0, (5,), 42)
    s.evict(0, (3,))  # also no-op
    assert s.contains(0, (5,))


def test_evict_dominated_drops_everything_in_downset_1d() -> None:
    """Feed a 1D antichain ``{(5,)}`` — covers ``(0,)..(5,)``. All
    those entries should go; ``(6,)`` survives."""
    s = DictStorage()
    for t in range(8):
        s.write(0, (t,), t * 10)
    dropped = s.evict_dominated(0, _ac1([(5,)]))
    assert dropped == 6  # (0,) through (5,)
    assert s.times(0) == [(6,), (7,)]


def test_evict_dominated_drops_zero_when_nothing_covered() -> None:
    """Antichain with no entries covers nothing."""
    s = DictStorage()
    s.write(0, (5,), 42)
    dropped = s.evict_dominated(0, _ac1())
    assert dropped == 0
    assert s.contains(0, (5,))


def test_evict_dominated_2d_partial_cover() -> None:
    """2D antichain ``{(2, 3)}`` covers the 2D box ``[0..2] × [0..3]``."""
    s = DictStorage()
    s.write(0, (0, 0), "a")
    s.write(0, (2, 3), "b")
    s.write(0, (3, 0), "c")  # outside (axis-0 > 2)
    s.write(0, (1, 4), "d")  # outside (axis-1 > 3)
    dropped = s.evict_dominated(0, _ac2([(2, 3)]))
    assert dropped == 2
    assert set(s.times(0)) == {(3, 0), (1, 4)}


def test_evict_dominated_only_touches_target_node() -> None:
    s = DictStorage()
    s.write(0, (5,), "node-0")
    s.write(1, (5,), "node-1")
    s.evict_dominated(0, _ac1([(10,)]))
    assert not s.contains(0, (5,))
    assert s.contains(1, (5,))


# ---- Inspection ------------------------------------------------------------


def test_times_returns_stored_keys() -> None:
    s = DictStorage()
    s.write(0, (1,), "a")
    s.write(0, (3,), "b")
    s.write(0, (2,), "c")
    assert set(s.times(0)) == {(1,), (2,), (3,)}


def test_times_empty_for_unknown_node() -> None:
    s = DictStorage()
    assert s.times(42) == []


def test_size_per_node_and_global() -> None:
    s = DictStorage()
    s.write(0, (1,), "a")
    s.write(0, (2,), "b")
    s.write(1, (1,), "c")
    assert s.size(0) == 2
    assert s.size(1) == 1
    assert s.size() == 3  # global


def test_size_unknown_node_is_zero() -> None:
    s = DictStorage()
    assert s.size(99) == 0


# ---- Storage Protocol conformance ------------------------------------------


def test_dict_storage_satisfies_storage_protocol() -> None:
    """``DictStorage`` is the default backend; a runtime
    ``isinstance`` against the runtime-checkable ``Storage`` protocol
    confirms the structural fit."""
    s = DictStorage()
    assert isinstance(s, Storage)


def test_storage_protocol_accepts_alternate_backend() -> None:
    """Any class with the right methods satisfies ``Storage``. A
    minimal duck-typed mock would; here we just verify the protocol
    is runtime-checkable on a non-DictStorage value."""

    class _NotStorage:
        pass

    assert not isinstance(_NotStorage(), Storage)
