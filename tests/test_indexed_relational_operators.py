"""Tests for ``pydbsp.indexed_relational_operators``."""

from __future__ import annotations

from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.evaluate import Evaluator
from pydbsp.indexed_relational_operators import LiftGroupBy, LiftIndex
from pydbsp.operator import Input
from pydbsp.progress import Time
from pydbsp.storage import DictStorage
from pydbsp.core import Antichain, dbsp_time
from pydbsp.zset import ZSet, ZSetAddition


Record = tuple[str, int]  # (dept, amount)


def _eval() -> Evaluator[Time]:
    return Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(1)),
        group=ZSetAddition[Record](),
    )


def test_groupby_sum_per_key() -> None:
    """``GROUP BY dept SUM(amount * weight)``. Each tick's aggregate
    is emitted as ``(dept, total) → 1`` in the output Z-set."""
    e = _eval()
    src = Input[ZSet[Record]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    indexed = LiftIndex[Record, str](indexer=lambda r: r[0]).connect(e.circuit, (src,))
    agg = LiftGroupBy[Record, str, int](
        aggregate=lambda items: sum(r[1] * w for r, w in items),
    ).connect(e.circuit, (indexed,))

    e.push(
        src,
        ZSet(
            {
                ("eng", 100): 1,
                ("eng", 200): 1,
                ("sales", 50): 1,
                ("sales", 75): 2,  # weighted twice
            }
        ),
    )

    result = e.read(agg, (0,))
    expected = ZSet(
        {
            ("eng", 300): 1,  # 100 + 200
            ("sales", 50 + 75 * 2): 1,  # 50 + 150 = 200
        }
    )
    assert result == expected


def test_groupby_count_per_key() -> None:
    """``GROUP BY dept COUNT(*)``, where count = Σ multiplicities."""
    e = _eval()
    src = Input[ZSet[Record]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    indexed = LiftIndex[Record, str](indexer=lambda r: r[0]).connect(e.circuit, (src,))
    agg = LiftGroupBy[Record, str, int](
        aggregate=lambda items: sum(w for _, w in items),
    ).connect(e.circuit, (indexed,))

    e.push(
        src,
        ZSet(
            {
                ("eng", 100): 1,
                ("eng", 200): 1,
                ("eng", 300): 1,
                ("sales", 50): 2,
                ("sales", 75): 1,
            }
        ),
    )

    result = e.read(agg, (0,))
    assert result == ZSet(
        {
            ("eng", 3): 1,
            ("sales", 3): 1,
        }
    )


def test_groupby_max_per_key() -> None:
    """``GROUP BY dept MAX(amount)`` — non-monoidal aggregate.
    LiftGroupBy works the same; the aggregator's logic differs."""
    e = _eval()
    src = Input[ZSet[Record]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    indexed = LiftIndex[Record, str](indexer=lambda r: r[0]).connect(e.circuit, (src,))
    agg = LiftGroupBy[Record, str, int](
        aggregate=lambda items: max(r[1] for r, _w in items),
    ).connect(e.circuit, (indexed,))

    e.push(
        src,
        ZSet(
            {
                ("eng", 100): 1,
                ("eng", 500): 1,
                ("eng", 200): 1,
                ("sales", 90): 1,
                ("sales", 30): 1,
            }
        ),
    )

    result = e.read(agg, (0,))
    assert result == ZSet(
        {
            ("eng", 500): 1,
            ("sales", 90): 1,
        }
    )


def test_groupby_empty_input_emits_nothing() -> None:
    """No records → no groups → empty output Z-set."""
    e = _eval()
    src = Input[ZSet[Record]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    indexed = LiftIndex[Record, str](indexer=lambda r: r[0]).connect(e.circuit, (src,))
    agg = LiftGroupBy[Record, str, int](
        aggregate=lambda items: sum(r[1] * w for r, w in items),
    ).connect(e.circuit, (indexed,))

    e.push(src, ZSet({}))
    assert e.read(agg, (0,)) == ZSet({})


def test_groupby_works_at_2d_arity() -> None:
    """LiftGroupBy is pointwise (Lift1), so it works at any arity.
    Here the source is a 2-D Input populated directly at distinct
    ``(outer, inner)`` cells; each cell is grouped independently."""
    eg = ZSetAddition[Record]()
    e = Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(2)),
        group=eg,
    )
    src = Input[ZSet[Record]](frontier=Antichain(dbsp_time(2))).connect(e.circuit, ())
    indexed = LiftIndex[Record, str](indexer=lambda r: r[0]).connect(e.circuit, (src,))
    agg = LiftGroupBy[Record, str, int](
        aggregate=lambda items: sum(r[1] * w for r, w in items),
    ).connect(e.circuit, (indexed,))

    # Two distinct 2-D cells; each carries its own batch.
    e.push(src, ZSet({("eng", 100): 1, ("eng", 200): 1}), t=(0, 0))
    e.push(src, ZSet({("eng", 50): 1, ("sales", 80): 1}), t=(1, 2))

    assert e.read(agg, (0, 0)) == ZSet({("eng", 300): 1})
    assert e.read(agg, (1, 2)) == ZSet(
        {
            ("eng", 50): 1,
            ("sales", 80): 1,
        }
    )


def test_groupby_shared_index_with_multiple_aggregates() -> None:
    """Two aggregates downstream of the same LiftIndex share the
    grouping work. Verifies both emit correctly."""
    e = _eval()
    src = Input[ZSet[Record]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    indexed = LiftIndex[Record, str](indexer=lambda r: r[0]).connect(e.circuit, (src,))
    total = LiftGroupBy[Record, str, int](
        aggregate=lambda items: sum(r[1] * w for r, w in items),
    ).connect(e.circuit, (indexed,))
    peak = LiftGroupBy[Record, str, int](
        aggregate=lambda items: max(r[1] for r, _w in items),
    ).connect(e.circuit, (indexed,))

    e.push(
        src,
        ZSet(
            {
                ("eng", 100): 1,
                ("eng", 500): 1,
                ("sales", 30): 1,
                ("sales", 90): 1,
            }
        ),
    )

    assert e.read(total, (0,)) == ZSet(
        {
            ("eng", 600): 1,
            ("sales", 120): 1,
        }
    )
    assert e.read(peak, (0,)) == ZSet(
        {
            ("eng", 500): 1,
            ("sales", 90): 1,
        }
    )
