"""Unit tests for the evaluator's optimizations:

* ``_deps_of`` memoizes ``op.deps(t)`` on first access; never invalidated.
* ``CellSchedule.resolved_layers`` pre-binds ``(compute_from, slot_key, t)``
  per cell so the dispatch loop has no per-cell attribute/id work.
"""

from __future__ import annotations

from pydbsp.core import AbelianGroupOperation, Time2D
from pydbsp.evaluator import Evaluator, compile_cell_schedule
from pydbsp.stream import Input
from pydbsp.stream.operators.linear import Delay, Integrate


class _IntGroup(AbelianGroupOperation[int]):
    def add(self, a: int, b: int) -> int:
        return a + b

    def neg(self, a: int) -> int:
        return -a

    def identity(self) -> int:
        return 0


def _make_tiny_circuit():
    """Integrate(Delay(Input)) on Time2D. Four cells minimum to have
    non-trivial deps: Delay reads prev, Integrate recurses."""
    inp = Input(_IntGroup(), Time2D)
    delayed = Delay(inp, Time2D, axis=1)
    acc = Integrate(delayed, Time2D, axis=1)
    return Evaluator(acc), inp


# ---- E1: deps memoization --------------------------------------------------


def test_deps_of_caches_after_first_call():
    ev, inp = _make_tiny_circuit()
    # Pick any op from the schedule.
    op = ev.schedule.ops[-1]
    t = (0, 1)
    assert (id(op), t) not in ev._deps_cache
    first = ev._deps_of(op, t)
    assert (id(op), t) in ev._deps_cache
    # Same list object returned on second call (identity, not just equality).
    second = ev._deps_of(op, t)
    assert first is second


def test_deps_of_matches_op_deps():
    ev, inp = _make_tiny_circuit()
    for op in ev.schedule.ops:
        for t in [(0, 0), (0, 1), (0, 2)]:
            cached = ev._deps_of(op, t)
            fresh = list(op.deps(t))
            assert cached == fresh, f"{type(op).__name__} at {t}: {cached} != {fresh}"


def test_deps_cache_populated_by_compile_cell_schedule():
    """compile_cell_schedule should populate the deps cache as a side effect."""
    ev, inp = _make_tiny_circuit()
    inp.push((0, 0), 1)
    ev.at((0, 0))
    assert len(ev._deps_cache) > 0


# ---- E3: pre-resolved schedule entries ------------------------------------


def test_resolved_layers_match_layers_shape():
    ev, inp = _make_tiny_circuit()
    inp.push((0, 0), 1)
    sched = compile_cell_schedule(ev, [(ev.schedule.ops[-1], (0, 0))])
    assert len(sched.resolved_layers) == len(sched.layers)
    for rlayer, layer in zip(sched.resolved_layers, sched.layers):
        assert len(rlayer) == len(layer)


def test_resolved_layers_content():
    """Each resolved entry is (compute_from, slot_key, t) where
    compute_from is the op's bound method and slot_key == (id(op), t)."""
    ev, inp = _make_tiny_circuit()
    inp.push((0, 0), 1)
    sched = compile_cell_schedule(ev, [(ev.schedule.ops[-1], (0, 0))])
    for layer, rlayer in zip(sched.layers, sched.resolved_layers):
        for (op_idx, t), (cf, key, rt) in zip(layer, rlayer):
            op = ev.schedule.ops[op_idx]
            assert cf == op.compute_from
            assert key == (id(op), t)
            assert rt == t


def test_dispatch_via_resolved_layers_gives_correct_output():
    """End-to-end: the evaluator uses resolved_layers for dispatch; result
    must match the algebraic spec."""
    inp = Input(_IntGroup(), Time2D)
    acc = Integrate(Delay(inp, Time2D, axis=1), Time2D, axis=1)
    ev = Evaluator(acc)
    for k in range(5):
        inp.push((0, k), k)
    # Integrate(Delay(x), axis=inner)(0,k) = sum_{k'<k} x(0,k'); so
    # at k=0 → 0, k=1 → 0, k=2 → 1, k=3 → 3, k=4 → 6.
    for k in range(5):
        expected = sum(range(k))
        assert ev.at((0, k)) == expected


def test_resolved_layers_reused_across_fill_many_calls():
    """Subsequent fill_many calls build fresh schedules (which have their
    own resolved_layers). The point of E3 is per-schedule, not across —
    verify the schedule builder emits resolved form."""
    ev, inp = _make_tiny_circuit()
    for k in range(3):
        inp.push((0, k), k + 1)
    ev.at((0, 0))
    # After first fill, cache populated.
    sched_after_first = compile_cell_schedule(ev, [(ev.schedule.ops[-1], (0, 5))])
    assert sched_after_first.resolved_layers
    # Each entry is the 3-tuple form.
    for layer in sched_after_first.resolved_layers:
        for entry in layer:
            assert len(entry) == 3
