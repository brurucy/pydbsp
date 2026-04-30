"""Iterative schedule-driven evaluator.

Given a compiled ``Schedule`` and a target ``(op, t)``, fills a
slot-table ``dict[(id(op), t), V]`` by walking the operator DAG in
topological order. Every read is a single ``dict.__getitem__`` —
no ``Stream.at`` indirection, no recursion.

Each operator class declares:

- ``deps(t) -> iterable of (dep_op, dep_t)`` — the stencil at t.
- ``compute_from(t, slots) -> value`` — direct slot lookups.

The Evaluator composes these: a back-stepped DFS from the target
collects all required ``(op, t)`` pairs; a topo-ordered forward
pass invokes ``compute_from`` on each.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydbsp.history import Schedule, compile_schedule
from pydbsp.stream import Stream


@dataclass
class CellSchedule:
    """Cell-level topological schedule for a specific ``(target_op, target_t)``.

    Each ``layer`` is a list of ``(op_idx, t)`` pairs whose dep closure is
    contained in strictly earlier layers — so within a layer the cells are
    pairwise dep-independent and can run in parallel. Compared to the
    op-level ``Schedule``'s ``(layer, sum(t))`` groups, this is the tight
    P-antichain dispatch: fewer steps, wider batches.

    ``resolved_layers`` is the pre-bound form of ``layers`` used by the
    dispatch loop: each entry is ``(compute_from, slot_key, t)`` — the op's
    compute method bound once, the slot-key precomputed as ``(id(op), t)``.
    Saves per-cell attribute and ``id()`` work during dispatch.
    """

    layers: list[list[tuple[int, tuple[int, ...]]]] = field(default_factory=list)
    resolved_layers: list[list[tuple[Any, tuple, tuple[int, ...]]]] = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        return sum(len(layer) for layer in self.layers)

    @property
    def n_layers(self) -> int:
        return len(self.layers)


def compile_cell_schedule(
    ev: "Evaluator",
    targets: list[tuple[Stream, tuple[int, ...]]]
    | tuple[tuple[Stream, tuple[int, ...]], ...],
    *,
    skip_computed: bool = True,
) -> CellSchedule:
    """Build the P-antichain cell schedule for a batch of targets.

    Walks ``op.deps(t)`` from every target, materialises the transitive
    cell-level dep DAG, then runs Kahn's algorithm to produce
    topologically-sorted layers.

    ``skip_computed``: when True (default) cells already present in the
    slot table are treated as dep-leaves — they don't appear in the
    schedule's work set. This keeps per-call compile work proportional
    to *pending* cells, not total closure size, which is critical for
    saturation loops where successive targets share most of their deps.
    """
    op_idx = ev._op_idx
    slots = ev._slots
    deps_of = ev._deps_of
    deps_map: dict[tuple[int, tuple[int, ...]], set[tuple[int, tuple[int, ...]]]] = {}
    order: list[tuple[int, tuple[int, ...]]] = []
    stack: list[tuple[Stream, tuple[int, ...]]] = list(targets)
    while stack:
        op, t = stack.pop()
        if skip_computed and (id(op), t) in slots:
            continue
        key = (op_idx[id(op)], t)
        if key in deps_map:
            continue
        deps: list[tuple[int, tuple[int, ...]]] = []
        for dop, dt in deps_of(op, t):
            if skip_computed and (id(dop), dt) in slots:
                continue
            deps.append((op_idx[id(dop)], dt))
            stack.append((dop, dt))
        deps_map[key] = set(deps)
        order.append(key)

    # Kahn's algorithm, preserving insertion order within a layer for
    # determinism and for Integrate-style self-recurrences (later-t cells
    # will naturally land in later layers thanks to their prev-t dep).
    layers: list[list[tuple[int, tuple[int, ...]]]] = []
    remaining = {k: set(d) for k, d in deps_map.items()}
    order_pos = {k: i for i, k in enumerate(order)}
    while remaining:
        ready = [k for k in remaining if not remaining[k]]
        if not ready:
            raise RuntimeError("cycle in dep graph")
        ready.sort(key=lambda k: order_pos[k])
        layers.append(ready)
        satisfied = set(ready)
        for k in ready:
            del remaining[k]
        for k in remaining:
            remaining[k] -= satisfied
    # Pre-resolve each cell to (compute_from, slot_key, t) — spares the
    # dispatch loop one list-index, one getattr, and one id() per cell.
    ops_list = ev._schedule.ops
    resolved: list[list[tuple[Any, tuple, tuple[int, ...]]]] = []
    for layer in layers:
        resolved_layer: list[tuple[Any, tuple, tuple[int, ...]]] = []
        for op_i, t in layer:
            op = ops_list[op_i]
            resolved_layer.append((op.compute_from, (id(op), t), t))
        resolved.append(resolved_layer)
    return CellSchedule(layers=layers, resolved_layers=resolved)


class Evaluator:
    """Schedule-driven iterative fill of a slot table.

    Dispatch is **P-antichain** over the cell-level dep DAG. Every
    ``fill_many`` call compiles a fresh ``CellSchedule`` covering the
    still-pending transitive closure of the targets (already-slotted
    cells are dep-leaves), then runs Kahn-style topologically layered
    batches: each layer is a set of ``(op, t)`` cells pairwise
    dep-independent, safe to dispatch in parallel.

    The wavefront alternative — ``(Schedule layer, sum(t))`` groups in
    op-level topological order — is a strict over-serialisation of
    this: it serves as the conservative upper-bound dispatcher, but
    was measured to leave material parallelism on the table for dense
    DAGs (see ``notebooks/progress_tracking.ipynb``). We ship
    P-antichain as the only strategy.

    Self-recurrent ops (``Integrate.at(t) = base.at(t) +
    Integrate.at(t - e_i)``) are handled naturally: the self-dep at
    ``t - e_i`` lives in a strictly earlier cell-schedule layer.
    """

    def __init__(
        self,
        output: Stream,
        *,
        parallelism: int = 1,
        parallel_layer_min_width: int = 4,
        gc: bool = True,
    ) -> None:
        self._output = output
        self._schedule: Schedule = compile_schedule(output)
        self._op_idx: dict[int, int] = {
            id(op): idx for idx, op in enumerate(self._schedule.ops)
        }
        self._slots: dict[tuple[int, tuple[int, ...]], Any] = {}
        self._parallelism = parallelism
        self._parallel_layer_min_width = max(parallel_layer_min_width, 1)
        self._gc_enabled = gc
        self._executor = None
        if parallelism > 1:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=parallelism)
        # Precompute reverse-deps once (op_idx → [consumer_idx, ...])
        self._consumers_of: dict[int, list[int]] = {}
        for c_idx, dep_idxs in self._schedule.deps.items():
            for d_idx in dep_idxs:
                self._consumers_of.setdefault(d_idx, []).append(c_idx)
        # Memoized ``op.deps(t)``. Pure for a fixed circuit, so never
        # invalidated. Used by ``compile_cell_schedule`` and ``gc`` —
        # both traverse the same dep relation many times over the
        # lifetime of an evaluator.
        self._deps_cache: dict[tuple[int, tuple[int, ...]], list[tuple[Stream, tuple[int, ...]]]] = {}

    def _deps_of(self, op: Stream, t: tuple[int, ...]) -> list[tuple[Stream, tuple[int, ...]]]:
        """Memoized ``op.deps(t)`` — pure in a fixed circuit."""
        key = (id(op), t)
        cached = self._deps_cache.get(key)
        if cached is not None:
            return cached
        result = list(op.deps(t))
        self._deps_cache[key] = result
        return result

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    @property
    def slots(self) -> dict[tuple[int, tuple[int, ...]], Any]:
        return self._slots

    @property
    def gc_enabled(self) -> bool:
        return self._gc_enabled

    def at(self, t: tuple[int, ...]) -> Any:
        """Returns the root observable's value at ``t``, computing any
        missing cells on demand.
        """
        return self.at_op(self._output, t)

    def at_op(self, op: Stream, t: tuple[int, ...]) -> Any:
        """Returns any scheduled operator's value at ``t``. ``op`` must be
        reachable from the root passed to ``__init__``.
        """
        if id(op) not in self._op_idx:
            raise KeyError(f"{type(op).__name__} not in this evaluator's schedule")
        self.fill_op(op, t)
        return self._slots[(id(op), t)]

    def fill_op(self, target_op: Stream, target_t: tuple[int, ...]) -> None:
        self.fill_many(((target_op, target_t),))

    def _dispatch_cell_schedule(self, sched: CellSchedule) -> None:
        slots = self._slots
        executor = self._executor
        parallelism = self._parallelism

        def run_chunk(chunk):
            for compute_from, slot_key, t in chunk:
                slots[slot_key] = compute_from(t, slots)

        def run_chunk_buffered(chunk):
            local_slots: dict[tuple[int, tuple[int, ...]], Any] = {}
            for compute_from, slot_key, t in chunk:
                local_slots[slot_key] = compute_from(t, slots)
            return local_slots

        for layer in sched.resolved_layers:
            # Narrow layers are cheaper to run inline than to split into
            # per-thread chunks and merge back into the slot table.
            if executor is not None and len(layer) >= self._parallel_layer_min_width:
                size = max(1, (len(layer) + parallelism - 1) // parallelism)
                chunks = [layer[i : i + size] for i in range(0, len(layer), size)]
                for local_slots in executor.map(run_chunk_buffered, chunks):
                    slots.update(local_slots)
            else:
                run_chunk(layer)

    def fill_many(
        self,
        targets: list[tuple[Stream, tuple[int, ...]]]
        | tuple[tuple[Stream, tuple[int, ...]], ...],
    ) -> None:
        """Fill multiple ``(op, t)`` targets in one pass via a batched
        P-antichain cell schedule. Auto-GCs at the end.
        """
        targets_list = list(targets)
        pending = [
            (op, t) for (op, t) in targets_list
            if (id(op), t) not in self._slots
        ]
        if pending:
            sched = compile_cell_schedule(self, pending)
            self._dispatch_cell_schedule(sched)
        if self._gc_enabled:
            self.gc()

    def gc(self) -> int:
        """Evict cells that every downstream consumer has already read.

        Contract: operators have **monotone-forward deps** (every
        dep ``dep_t`` is ``< t`` componentwise on some axis, or equal
        on pointwise ops). Under this invariant, a consumer that has
        computed up to some ``t_C`` will never re-read a cell it
        already consumed; so cells appearing in
        ``⋂_{C ∈ consumers(X)} (⋃_{t_C ∈ computed(C)} deps(C,t_C)|_X)``
        are safe to drop forever.

        Called automatically at the end of every ``fill_many`` —
        exposed publicly so callers can force an immediate sweep
        between pushes (e.g. after bulk-loading state).

        Returns: number of slots freed.
        """
        slots = self._slots
        ops = self._schedule.ops
        op_idx = self._op_idx

        # Group current slot keys by op_idx.
        cells_by_op: dict[int, list[tuple[int, ...]]] = {}
        for (oid, t) in slots.keys():
            idx = op_idx.get(oid)
            if idx is not None:
                cells_by_op.setdefault(idx, []).append(t)

        evicted = 0
        for X_idx in range(len(ops)):
            X_cells = cells_by_op.get(X_idx)
            if not X_cells:
                continue
            consumers = self._consumers_of.get(X_idx, [])
            # Root ops (no consumers) stay live so external ``at_op``
            # reads don't disappear.
            if not consumers:
                continue

            # Intersection over consumers of (cells-X-read-by-that-C).
            evictable: set[tuple[int, ...]] | None = None
            X_op = ops[X_idx]
            X_op_id = id(X_op)
            for C_idx in consumers:
                C = ops[C_idx]
                C_cells = cells_by_op.get(C_idx, [])
                read_by_C: set[tuple[int, ...]] = set()
                for tc in C_cells:
                    for dep_op, dep_t in self._deps_of(C, tc):
                        if id(dep_op) == X_op_id:
                            read_by_C.add(dep_t)
                if evictable is None:
                    evictable = read_by_C
                else:
                    evictable &= read_by_C
                if not evictable:
                    break

            if evictable:
                for t in evictable:
                    key = (X_op_id, t)
                    if key in slots:
                        del slots[key]
                        evicted += 1

        return evicted
