"""Storage. Protocol for the per-(node, time) value store, plus a
default in-memory implementation.

The evaluator reads and writes through a :class:`Storage` for both:

* **Input storage**. Values explicitly pushed at source nodes
  (``operator.Input`` / ``compute.Get``). ``read(input_node, t,
  default=group.identity())`` is how an evaluator implements the
  "absent timestamp returns identity" rule.
* **Compute memo cache**. Values produced by derived nodes' compute
  rules. Cached so a second read of the same ``(node, t)`` does not
  re-invoke ``compute``. Particularly important for the strict
  feedback recurrence inside :class:`compute.Sum` /
  :class:`compute.Foldl`, whose ``self(t)`` reads ``self(t-1)``.

Different backends can satisfy the protocol (in-memory dict, sorted
indexes, persistent, distributed, etc.). :class:`DictStorage` is the
default: a two-level dict ``{NodeId: {Time: value}}``, suitable for
tests and small in-process workloads. Production backends are free to
optimise eviction, layout, persistence, etc.. As long as they
implement the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Protocol, runtime_checkable

from pydbsp.core import Antichain

from pydbsp.progress import NodeId, Time

_MISSING: Final = object()
"""Sentinel passed to ``read`` to distinguish 'absent' from 'stored
value happens to be None'. Shared at the module level so
implementations agree on the same singleton."""


@runtime_checkable
class Storage(Protocol):
    """The contract every storage backend implements. Methods are
    intentionally minimal: ``write``, ``read``, ``contains``,
    ``evict``, and the GC-friendly ``evict_dominated``, plus the
    inspection helpers ``times`` and ``size``.

    Implementations are free to differ in layout, eviction strategy,
    persistence, etc., but the observable behaviour at this interface
    must match :class:`DictStorage`'s semantics.
    """

    def write(self, node: NodeId, t: Time, value: Any) -> None:
        """Store ``value`` at ``(node, t)``, overwriting any prior
        entry. Callers that need to *merge* multiple writes at the
        same timestamp must read-modify-write through the appropriate
        group's ``add``."""
        ...

    def read(self, node: NodeId, t: Time, default: Any = _MISSING) -> Any:
        """Return the value at ``(node, t)``. If absent: return
        ``default`` when supplied (sentinel-based to allow stored
        ``None`` values), else raise ``KeyError``."""
        ...

    def contains(self, node: NodeId, t: Time) -> bool: ...

    def evict(self, node: NodeId, t: Time) -> None:
        """Drop the entry at ``(node, t)`` if present. No-op
        otherwise."""
        ...

    def evict_dominated(
        self,
        node: NodeId,
        dead: Antichain[Time],
    ) -> int:
        """Drop every ``(node, t)`` whose ``t`` is in ``dead``'s
        down-set. Returns the number of entries evicted. Used by
        the GC pass. Feed it ``propagate_backward``'s per-node
        antichain."""
        ...

    def times(self, node: NodeId) -> list[Time]:
        """All timestamps with stored values for ``node``."""
        ...

    def size(self, node: NodeId | None = None) -> int:
        """Number of entries for one node, or globally if
        ``node is None``."""
        ...


@dataclass
class DictStorage(Storage):
    """Default :class:`Storage` backend: a two-level dict
    ``{NodeId: {Time: value}}``. ``evict_dominated`` scans the node's
    timestamps and tests each via ``Antichain.covers``. O(stored)
    per call, fine for tests and small workloads. Production-grade
    backends (sorted indexes, etc.) can be dropped in by implementing
    the protocol."""

    _data: dict[NodeId, dict[Time, Any]] = field(default_factory=dict)

    # ---- Read / write ------------------------------------------------------

    def write(self, node: NodeId, t: Time, value: Any) -> None:
        self._data.setdefault(node, {})[t] = value

    def read(self, node: NodeId, t: Time, default: Any = _MISSING) -> Any:
        v = self._data.get(node, {}).get(t, _MISSING)
        if v is _MISSING:
            if default is _MISSING:
                raise KeyError((node, t))
            return default
        return v

    def contains(self, node: NodeId, t: Time) -> bool:
        return node in self._data and t in self._data[node]

    # ---- Eviction ----------------------------------------------------------

    def evict(self, node: NodeId, t: Time) -> None:
        if node in self._data:
            self._data[node].pop(t, None)

    def evict_dominated(
        self,
        node: NodeId,
        dead: Antichain[Time],
    ) -> int:
        node_data = self._data.get(node)
        if not node_data:
            return 0
        to_drop = [t for t in node_data if dead.covers(t)]
        for t in to_drop:
            del node_data[t]
        return len(to_drop)

    # ---- Inspection --------------------------------------------------------

    def times(self, node: NodeId) -> list[Time]:
        return list(self._data.get(node, {}).keys())

    def size(self, node: NodeId | None = None) -> int:
        if node is not None:
            return len(self._data.get(node, {}))
        return sum(len(ts) for ts in self._data.values())
