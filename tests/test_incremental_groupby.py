"""Tests for :class:`DeltaLiftedDeltaLiftedGroupBy` — the
doubly-incremental ``GROUP BY ... AGGREGATE``.

The primary correctness oracle is the *integral-equality* property:
``Integrate(DeltaLiftedDeltaLiftedGroupBy(s))`` must equal the
known-correct (but ``O(|state|)``) cumulative reference
``LiftGroupBy(Integrate(s))`` at every cell, for any aggregate. The
incremental operator only emits the *delta* of that cumulative view,
touching changed keys, so integrating it must reconstruct the
reference snapshot exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.core import Antichain, dbsp_time
from pydbsp.evaluate import Evaluator
from pydbsp.indexed_relational_operators import (
    DeltaLiftedDeltaLiftedGroupBy,
    LiftGroupBy,
    LiftLiftIndex,
)
from pydbsp.indexed_zset import IndexedZSet, IndexedZSetAddition
from pydbsp.operator import Input, Integrate, LiftStreamIntroduction
from pydbsp.progress import Time
from pydbsp.storage import DictStorage
from pydbsp.zset import ZSet, ZSetAddition

Record = tuple[str, int]

Agg = Callable[[Iterable[tuple[Record, int]]], int]


def _build(agg: Agg) -> tuple[Evaluator[Time], int, int, int, int]:
    """Wire src → 2D → index → (incremental groupby, its integral,
    and the cumulative reference). Returns
    ``(e, src, inc, inc_cum, ref)``."""
    e = Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(2)),
        group=ZSetAddition[Record](),
    )
    g_rec: ZSetAddition[Record] = ZSetAddition()
    g_idx: IndexedZSetAddition[str, Record] = IndexedZSetAddition(g_rec, lambda r: r[0])
    g_out: ZSetAddition[tuple[str, int]] = ZSetAddition()

    src = Input[ZSet[Record]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    src_2d = LiftStreamIntroduction[ZSet[Record]](group=g_rec).connect(e.circuit, (src,))
    indexed = LiftLiftIndex[Record, str](indexer=lambda r: r[0]).connect(e.circuit, (src_2d,))

    inc = DeltaLiftedDeltaLiftedGroupBy[Record, str, int](
        aggregate=agg, group=g_idx, out_group=g_out
    ).connect(e.circuit, (indexed,))
    inc_cum = Integrate[ZSet[tuple[str, int]]](group=g_out).connect(e.circuit, (inc,))

    int_indexed = Integrate[IndexedZSet[str, Record]](group=g_idx).connect(e.circuit, (indexed,))
    ref = LiftGroupBy[Record, str, int](aggregate=agg).connect(e.circuit, (int_indexed,))
    return e, src, inc, inc_cum, ref


def _sum_agg(items: Iterable[tuple[Record, int]]) -> int:
    return sum(r[1] * w for r, w in items)


def _max_agg(items: Iterable[tuple[Record, int]]) -> int:
    return max(r[1] for r, w in items if w > 0)


def _assert_matches_reference(agg: Agg, pushes: list[ZSet[Record]]) -> None:
    """The integral-equality oracle: integrating the incremental
    output reproduces the cumulative reference at every outer tick."""
    e, src, _inc, inc_cum, ref = _build(agg)
    for p in pushes:
        e.push(src, p)
    for t in range(len(pushes)):
        assert e.read(inc_cum, (t, 0)) == e.read(ref, (t, 0)), f"cumulative mismatch at t={t}"


# ---------------------------------------------------------------------------
# Exact per-tick deltas (linear aggregate: sum)
# ---------------------------------------------------------------------------


def test_sum_emits_retraction_plus_assertion() -> None:
    e, src, inc, _inc_cum, _ref = _build(_sum_agg)

    e.push(src, ZSet({("a", 10): 1, ("a", 5): 1, ("b", 7): 1}))
    e.push(src, ZSet({("a", 3): 1}))
    e.push(src, ZSet({("a", 10): -1}))

    # t0: groups first appear → assertion only, no stale to retract.
    assert e.read(inc, (0, 0)) == ZSet({("a", 15): 1, ("b", 7): 1})
    # t1: only group "a" changed (15→18); "b" untouched → no emission.
    assert e.read(inc, (1, 0)) == ZSet({("a", 15): -1, ("a", 18): 1})
    # t2: "a" drops the (a,10) record (18→8).
    assert e.read(inc, (2, 0)) == ZSet({("a", 18): -1, ("a", 8): 1})


def test_unchanged_keys_emit_nothing() -> None:
    """A tick that touches only one key must not re-emit other groups."""
    e, src, inc, _inc_cum, _ref = _build(_sum_agg)

    e.push(src, ZSet({("a", 1): 1, ("b", 2): 1, ("c", 3): 1}))
    e.push(src, ZSet({("b", 5): 1}))  # only "b" changes

    assert e.read(inc, (1, 0)) == ZSet({("b", 2): -1, ("b", 7): 1})


def test_group_emptied_emits_only_retraction() -> None:
    """When a group's records all cancel, emit the retraction and no
    assertion (no aggregate over an empty bucket)."""
    e, src, inc, _inc_cum, _ref = _build(_sum_agg)

    e.push(src, ZSet({("a", 4): 1, ("b", 9): 1}))
    e.push(src, ZSet({("a", 4): -1}))  # group "a" empties

    assert e.read(inc, (1, 0)) == ZSet({("a", 4): -1})


# ---------------------------------------------------------------------------
# Non-linear aggregate (max): the changed bucket is re-scanned in full,
# so dropping the current max correctly falls back to the runner-up.
# ---------------------------------------------------------------------------


def test_max_falls_back_on_retraction() -> None:
    e, src, inc, _inc_cum, _ref = _build(_max_agg)

    e.push(src, ZSet({("a", 10): 1, ("a", 5): 1}))
    e.push(src, ZSet({("a", 20): 1}))
    e.push(src, ZSet({("a", 20): -1}))  # remove the max → back to 10

    assert e.read(inc, (0, 0)) == ZSet({("a", 10): 1})
    assert e.read(inc, (1, 0)) == ZSet({("a", 10): -1, ("a", 20): 1})
    assert e.read(inc, (2, 0)) == ZSet({("a", 20): -1, ("a", 10): 1})


# ---------------------------------------------------------------------------
# Integral-equality oracle across a variety of aggregates / sequences
# ---------------------------------------------------------------------------


def test_oracle_sum() -> None:
    _assert_matches_reference(
        _sum_agg,
        [
            ZSet({("a", 10): 1, ("a", 5): 1, ("b", 7): 1}),
            ZSet({("a", 3): 1, ("c", 1): 1}),
            ZSet({("a", 10): -1, ("b", 7): -1}),  # "b" empties
            ZSet({("c", 4): 1}),
        ],
    )


def test_oracle_max() -> None:
    _assert_matches_reference(
        _max_agg,
        [
            ZSet({("a", 1): 1, ("a", 9): 1, ("b", 3): 1}),
            ZSet({("a", 9): -1}),  # drop current max
            ZSet({("b", 100): 1}),
            ZSet({("a", 50): 1, ("b", 100): -1}),
        ],
    )


def test_oracle_count() -> None:
    count_agg: Agg = lambda items: sum(w for _r, w in items)
    _assert_matches_reference(
        count_agg,
        [
            ZSet({("a", 1): 1, ("a", 2): 1}),
            ZSet({("a", 3): 1, ("b", 1): 1}),
            ZSet({("a", 1): -1}),
        ],
    )


def test_oracle_multiplicity() -> None:
    """Records arriving with weight > 1, then partial retraction."""
    _assert_matches_reference(
        _sum_agg,
        [
            ZSet({("a", 5): 3}),
            ZSet({("a", 5): -2}),
            ZSet({("a", 5): -1}),  # group "a" empties
        ],
    )
