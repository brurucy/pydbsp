"""Tests for ``pydbsp.relational_operators`` — the composite Z-set
operators (Select, Project, DLDJoin, DLDDistinct).

Each test drives the operator through the evaluator on a small
example and verifies the resulting Z-sets. The bilinear operators are
exercised on a 2-axis lattice with 1-D inputs lifted via
``TimeAxisIntroduction``."""

from __future__ import annotations

from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.evaluate import Evaluator
from pydbsp.operator import Input, LiftStreamIntroduction
from pydbsp.progress import Time
from pydbsp.relational_operators import (
    DeltaLiftedDeltaLiftedDistinct,
    DeltaLiftedDeltaLiftedJoin,
    LiftProject,
    LiftSelect,
)
from pydbsp.storage import DictStorage
from pydbsp.core import Antichain, dbsp_time
from pydbsp.zset import ZSet, ZSetAddition


# ---- Fixtures --------------------------------------------------------------


Edge = tuple[int, int]


def _eval_1d() -> Evaluator[Time]:
    """1-D evaluator (for Select / Project tests)."""
    return Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(1)),
        group=ZSetAddition[int](),
    )


def _eval_2d_edges() -> Evaluator[Time]:
    """2-D evaluator over ``ZSet[Edge]`` (for the bilinear / distinct
    tests). 1-D inputs are pushed, then lifted via
    ``TimeAxisIntroduction(axis=1)``."""
    return Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(2)),
        group=ZSetAddition[Edge](),
    )


# ============================================================================
# Select — pointwise filter
# ============================================================================


def test_select_filters_zset_by_predicate() -> None:
    e = _eval_1d()
    src = Input[ZSet[int]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    out = LiftSelect[int](pred=lambda x: x > 0).connect(e.circuit, (src,))

    e.push(src, ZSet({1: 1, -1: 1, 5: 2, -10: 1}))
    assert e.read(out, (0,)) == ZSet({1: 1, 5: 2})


def test_select_preserves_weights() -> None:
    """Linearity: a Z-set with non-unit weights keeps them under
    ``select``."""
    e = _eval_1d()
    src = Input[ZSet[int]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    out = LiftSelect[int](pred=lambda x: x % 2 == 0).connect(e.circuit, (src,))

    e.push(src, ZSet({2: 3, 3: 1, 4: -2}))
    assert e.read(out, (0,)) == ZSet({2: 3, 4: -2})


# ============================================================================
# Project — pointwise transform
# ============================================================================


def test_project_transforms_elements() -> None:
    e = _eval_1d()
    src = Input[ZSet[int]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    out = LiftProject[int, int](f=lambda x: x * 2).connect(e.circuit, (src,))

    e.push(src, ZSet({1: 1, 2: 1, 3: 1}))
    assert e.read(out, (0,)) == ZSet({2: 1, 4: 1, 6: 1})


def test_project_sums_weights_on_collisions() -> None:
    """When ``f`` maps two distinct elements to the same key, the
    weights add (the group's structure-preserving collapse)."""
    e = _eval_1d()
    src = Input[ZSet[int]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    out = LiftProject[int, int](f=lambda x: x // 2).connect(e.circuit, (src,))

    e.push(src, ZSet({2: 1, 3: 1, 4: 2}))
    # 2 → 1 (weight 1), 3 → 1 (weight 1), 4 → 2 (weight 2). 1 collapses to 2.
    assert e.read(out, (0,)) == ZSet({1: 2, 2: 2})


# ============================================================================
# DeltaLiftedDeltaLiftedJoin — 4-term bilinear
# ============================================================================


def test_dld_join_at_origin_equals_join_of_inputs() -> None:
    """At ``(0, 0)`` the four terms reduce to the j2 term:
    ``J(IᵒIⁱa(0,0), b(0,0)) = J(a's initial, b's initial)``. The
    other three terms involve ``z⁻¹ᵒ`` or ``z⁻¹ⁱ`` factors that
    evaluate to identity at the lattice bottom."""
    e = _eval_2d_edges()
    eg = ZSetAddition[Edge]()

    a_1d = Input[ZSet[Edge]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    b_1d = Input[ZSet[Edge]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    a_2d = LiftStreamIntroduction[ZSet[Edge]](group=eg).connect(e.circuit, (a_1d,))
    b_2d = LiftStreamIntroduction[ZSet[Edge]](group=eg).connect(e.circuit, (b_1d,))
    joined = DeltaLiftedDeltaLiftedJoin[Edge, Edge, Edge](
        pred=lambda l, r: l[1] == r[0],
        proj=lambda l, r: (l[0], r[1]),
        group_a=eg,
        group_b=eg,
        out_group=eg,
    ).connect(e.circuit, (a_2d, b_2d))

    e.push(a_1d, ZSet({(0, 1): 1}))
    e.push(b_1d, ZSet({(1, 2): 1}))

    assert e.read(joined, (0, 0)) == ZSet({(0, 2): 1})


def test_dld_join_off_impulse_returns_identity() -> None:
    """At ``(0, k > 0)`` both lifted inputs return identity (the δ_k
    impulse only fires at ``k = 0``); the join is identity too."""
    e = _eval_2d_edges()
    eg = ZSetAddition[Edge]()

    a_1d = Input[ZSet[Edge]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    b_1d = Input[ZSet[Edge]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    a_2d = LiftStreamIntroduction[ZSet[Edge]](group=eg).connect(e.circuit, (a_1d,))
    b_2d = LiftStreamIntroduction[ZSet[Edge]](group=eg).connect(e.circuit, (b_1d,))
    joined = DeltaLiftedDeltaLiftedJoin[Edge, Edge, Edge](
        pred=lambda l, r: l[1] == r[0],
        proj=lambda l, r: (l[0], r[1]),
        group_a=eg,
        group_b=eg,
        out_group=eg,
    ).connect(e.circuit, (a_2d, b_2d))

    e.push(a_1d, ZSet({(0, 1): 1}))
    e.push(b_1d, ZSet({(1, 2): 1}))

    # δ_1 puts inputs at (0, 0) only; at (0, 1) both are identity, so
    # the join is identity.
    assert e.read(joined, (0, 1)) == ZSet({})


def test_dld_join_outer_delta_at_origin() -> None:
    """Pushing different ``a``-values at outer ticks 0 and 1, with a
    constant ``b``, exercises the outer-axis delta. At ``(1, 0)`` the
    join should reflect the second tick's ``a``-delta joined with
    ``b``."""
    e = _eval_2d_edges()
    eg = ZSetAddition[Edge]()

    a_1d = Input[ZSet[Edge]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    b_1d = Input[ZSet[Edge]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    a_2d = LiftStreamIntroduction[ZSet[Edge]](group=eg).connect(e.circuit, (a_1d,))
    b_2d = LiftStreamIntroduction[ZSet[Edge]](group=eg).connect(e.circuit, (b_1d,))
    joined = DeltaLiftedDeltaLiftedJoin[Edge, Edge, Edge](
        pred=lambda l, r: l[1] == r[0],
        proj=lambda l, r: (l[0], r[1]),
        group_a=eg,
        group_b=eg,
        out_group=eg,
    ).connect(e.circuit, (a_2d, b_2d))

    # Tick 0: a = {(0,1):1}. Tick 1: a-delta = {(2,1):1}.
    # b only pushed at tick 0: b = {(1,2):1}. b at tick 1 = identity.
    e.push(a_1d, ZSet({(0, 1): 1}))
    e.push(b_1d, ZSet({(1, 2): 1}))

    # At (0, 0): only j2 fires — {(0,1)} ⋈ {(1,2)} = {(0,2)}.
    assert e.read(joined, (0, 0)) == ZSet({(0, 2): 1})

    # Push the next outer delta on a; b stays at outer-frontier (0,).
    e.push(a_1d, ZSet({(2, 1): 1}))
    # At (1, 0): the doubly-incremental join's outer-delta term picks
    # up the new a-delta joined with the cumulative b (which is just
    # {(1,2):1}). Expected outer-delta = {(2,2):1}.
    assert e.read(joined, (1, 0)) == ZSet({(2, 2): 1})


# ============================================================================
# DeltaLiftedDeltaLiftedDistinct — H-based incremental distinct
# ============================================================================


def test_dld_distinct_first_arrival_passes_through() -> None:
    """The first time an element appears with positive weight, the
    H-based distinct emits it with weight +1."""
    e = _eval_2d_edges()
    eg = ZSetAddition[Edge]()

    src_1d = Input[ZSet[Edge]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    src_2d = LiftStreamIntroduction[ZSet[Edge]](group=eg).connect(e.circuit, (src_1d,))
    out = DeltaLiftedDeltaLiftedDistinct[Edge](inner_group=eg).connect(e.circuit, (src_2d,))

    e.push(src_1d, ZSet({(0, 1): 3}))  # weight 3 → only +1 in distinct
    assert e.read(out, (0, 0)) == ZSet({(0, 1): 1})


def test_dld_distinct_no_threshold_crossing_yields_identity() -> None:
    """A second positive delta on an already-positive element doesn't
    cross any +/- threshold; the distinct output is identity at the
    new outer tick."""
    e = _eval_2d_edges()
    eg = ZSetAddition[Edge]()

    src_1d = Input[ZSet[Edge]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    src_2d = LiftStreamIntroduction[ZSet[Edge]](group=eg).connect(e.circuit, (src_1d,))
    out = DeltaLiftedDeltaLiftedDistinct[Edge](inner_group=eg).connect(e.circuit, (src_2d,))

    e.push(src_1d, ZSet({(0, 1): 1}))  # entry → distinct emits +1
    e.push(src_1d, ZSet({(0, 1): 1}))  # second delta — already in
    assert e.read(out, (0, 0)) == ZSet({(0, 1): 1})
    assert e.read(out, (1, 0)) == ZSet({})


def test_dld_distinct_threshold_exit_emits_negative_one() -> None:
    """When a previously-positive element's total weight drops to ≤ 0,
    distinct emits -1 at the crossing tick."""
    e = _eval_2d_edges()
    eg = ZSetAddition[Edge]()

    src_1d = Input[ZSet[Edge]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    src_2d = LiftStreamIntroduction[ZSet[Edge]](group=eg).connect(e.circuit, (src_1d,))
    out = DeltaLiftedDeltaLiftedDistinct[Edge](inner_group=eg).connect(e.circuit, (src_2d,))

    e.push(src_1d, ZSet({(0, 1): 1}))  # enters: +1
    e.push(src_1d, ZSet({(0, 1): -1}))  # leaves: -1
    assert e.read(out, (0, 0)) == ZSet({(0, 1): 1})
    assert e.read(out, (1, 0)) == ZSet({(0, 1): -1})
