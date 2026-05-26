"""Pinning tests for the 3-D triply-incremental bilinear primitives.

The 3-D analogs of ``DeltaLiftedDeltaLiftedJoin`` /
``DeltaLiftedDeltaLiftedDistinct`` live in
:mod:`pydbsp.datalog_stratified`. We exercise them here in
isolation so the derivation is pinned independently of the
stratified-Datalog body that sits on top.

Strategy
--------

We compute the **ground truth** for both primitives independently and
cell-by-cell from the same input streams, then assert that the 3-D
operator reproduces it across the full lattice region we pushed.

For the bilinear join, ground truth is the 3-D inclusion-exclusion of
the cumulative join

    Δ³c(o, s, i)  =  Σ_{S ⊆ {o, s, i}} (-1)^|S| · c(corner_S)

where ``c(o, s, i) = J(Iᵒ Iˢ Iⁱ a, Iᵒ Iˢ Iⁱ b)(o, s, i)`` and
``corner_S`` is ``(o, s, i)`` with each axis in ``S`` shifted back by
one (treating out-of-range as ZSet identity). If the 8-term formula is
correct, the operator's output at every cell equals Δ³c.

For the distinct, ground truth is the discrete triple-delta of
``distinct(Iᵒ Iˢ Iⁱ s)``."""

from __future__ import annotations

import random
from typing import Callable

import pytest

from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.core import Antichain, dbsp_time
from pydbsp.datalog_stratified import (
    DeltaLiftedDeltaLiftedDeltaLiftedDistinct,
    DeltaLiftedDeltaLiftedDeltaLiftedJoin,
)
from pydbsp.evaluate import Evaluator
from pydbsp.operator import Input
from pydbsp.progress import Time
from pydbsp.storage import DictStorage
from pydbsp.zset import ZSet, ZSetAddition
from pydbsp.zset.functions.bilinear import join


Edge = tuple[int, int]


def distinct(z: ZSet[Edge]) -> ZSet[Edge]:
    """Value-level distinct: positive-weight entries map to weight 1.
    Used as the ground-truth post-cumulative for the 3-D distinct's
    triple-delta check."""
    return ZSet({k: 1 for k, w in z.inner.items() if w > 0})


# ---- Ground-truth helpers --------------------------------------------------


def _cumulative(
    pushes: dict[tuple[int, int, int], ZSet[Edge]],
    o: int,
    s: int,
    i: int,
    group: ZSetAddition[Edge],
) -> ZSet[Edge]:
    """``(Iᵒ Iˢ Iⁱ stream)(o, s, i)`` — the cumulative Z-set summed
    over every cell with coordinates ≤ (o, s, i) componentwise. Cells
    not in ``pushes`` are identity."""
    if o < 0 or s < 0 or i < 0:
        return group.identity()
    acc = group.identity()
    for (oc, sc, ic), v in pushes.items():
        if oc <= o and sc <= s and ic <= i:
            acc = group.add(acc, v)
    return acc


def _cumulative_join(
    pushes_a: dict[tuple[int, int, int], ZSet[Edge]],
    pushes_b: dict[tuple[int, int, int], ZSet[Edge]],
    o: int,
    s: int,
    i: int,
    group: ZSetAddition[Edge],
    pred: Callable[[Edge, Edge], bool],
    proj: Callable[[Edge, Edge], Edge],
) -> ZSet[Edge]:
    """``J(Iᵒ Iˢ Iⁱ a, Iᵒ Iˢ Iⁱ b)(o, s, i)``. Out-of-range gives
    identity."""
    if o < 0 or s < 0 or i < 0:
        return ZSet({})
    a_cum = _cumulative(pushes_a, o, s, i, group)
    b_cum = _cumulative(pushes_b, o, s, i, group)
    return join(a_cum, b_cum, pred, proj)


def _triple_delta(
    f: Callable[[int, int, int], ZSet[Edge]],
    o: int,
    s: int,
    i: int,
    group: ZSetAddition[Edge],
) -> ZSet[Edge]:
    """3-D inclusion-exclusion delta of ``f`` at ``(o, s, i)``."""
    acc = group.identity()
    for do in (0, -1):
        for ds in (0, -1):
            for di in (0, -1):
                sign = 1 if (do + ds + di) % 2 == 0 else -1
                contrib = f(o + do, s + ds, i + di)
                if sign == 1:
                    acc = group.add(acc, contrib)
                else:
                    acc = group.add(acc, group.neg(contrib))
    return acc


def _expected_delta_join(
    pushes_a: dict[tuple[int, int, int], ZSet[Edge]],
    pushes_b: dict[tuple[int, int, int], ZSet[Edge]],
    o: int,
    s: int,
    i: int,
    group: ZSetAddition[Edge],
    pred: Callable[[Edge, Edge], bool],
    proj: Callable[[Edge, Edge], Edge],
) -> ZSet[Edge]:
    return _triple_delta(
        lambda oo, ss, ii: _cumulative_join(pushes_a, pushes_b, oo, ss, ii, group, pred, proj),
        o,
        s,
        i,
        group,
    )


def _expected_delta_distinct(
    pushes: dict[tuple[int, int, int], ZSet[Edge]],
    o: int,
    s: int,
    i: int,
    group: ZSetAddition[Edge],
) -> ZSet[Edge]:
    return _triple_delta(
        lambda oo, ss, ii: (
            distinct(_cumulative(pushes, oo, ss, ii, group)) if oo >= 0 and ss >= 0 and ii >= 0 else ZSet({})
        ),
        o,
        s,
        i,
        group,
    )


# ---- Circuit builders ------------------------------------------------------


def _eval_3d() -> Evaluator[Time]:
    return Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(3)),
        group=ZSetAddition[Edge](),
    )


def _wire_join(
    e: Evaluator[Time],
    pred: Callable[[Edge, Edge], bool],
    proj: Callable[[Edge, Edge], Edge],
) -> tuple[int, int, int]:
    """Two 3-D Inputs feeding an 8-term 3-D join. Returns
    ``(input_a, input_b, joined)`` node ids."""
    eg = ZSetAddition[Edge]()
    a_in = Input[ZSet[Edge]](frontier=Antichain(dbsp_time(3))).connect(e.circuit, ())
    b_in = Input[ZSet[Edge]](frontier=Antichain(dbsp_time(3))).connect(e.circuit, ())
    joined = DeltaLiftedDeltaLiftedDeltaLiftedJoin[Edge, Edge, Edge](
        pred=pred,
        proj=proj,
        group_a=eg,
        group_b=eg,
        out_group=eg,
    ).connect(e.circuit, (a_in, b_in))
    return a_in, b_in, joined


def _wire_distinct(e: Evaluator[Time]) -> tuple[int, int]:
    eg = ZSetAddition[Edge]()
    src = Input[ZSet[Edge]](frontier=Antichain(dbsp_time(3))).connect(e.circuit, ())
    out = DeltaLiftedDeltaLiftedDeltaLiftedDistinct[Edge](inner_group=eg).connect(e.circuit, (src,))
    return src, out


# ---- Origin smoke test (the j_∅ term carries the whole thing) ----------


def test_join_at_origin_picks_up_only_one_term() -> None:
    """At ``(0, 0, 0)`` every term except ``Sₐ = ∅`` involves a
    ``z⁻¹ᵏ`` that reads the lattice bottom and returns identity. The
    surviving term is ``J(a, Iᵒ Iˢ Iⁱ b)(0, 0, 0) = J(a, b)`` at the
    origin."""
    e = _eval_3d()
    a_in, b_in, joined = _wire_join(
        e,
        pred=lambda l, r: l[1] == r[0],
        proj=lambda l, r: (l[0], r[1]),
    )

    e.push(a_in, ZSet({(0, 1): 1, (3, 4): 1}), t=(0, 0, 0))
    e.push(b_in, ZSet({(1, 2): 1, (4, 5): 1}), t=(0, 0, 0))

    assert e.read(joined, (0, 0, 0)) == ZSet({(0, 2): 1, (3, 5): 1})


def test_join_off_impulse_is_identity_when_only_one_axis_advances() -> None:
    """Pushing only at ``(0, 0, 0)`` and reading at ``(0, 0, 1)``: the
    cumulative join is constant across the inner axis, so Δ³ is zero
    at every cell except ``(0, 0, 0)``."""
    e = _eval_3d()
    a_in, b_in, joined = _wire_join(
        e,
        pred=lambda l, r: l[1] == r[0],
        proj=lambda l, r: (l[0], r[1]),
    )
    e.push(a_in, ZSet({(0, 1): 1}), t=(0, 0, 0))
    e.push(b_in, ZSet({(1, 2): 1}), t=(0, 0, 0))

    assert e.read(joined, (0, 0, 1)) == ZSet({})
    assert e.read(joined, (0, 1, 0)) == ZSet({})
    assert e.read(joined, (1, 0, 0)) == ZSet({})
    assert e.read(joined, (1, 1, 1)) == ZSet({})


# ---- The pinning test for the join ---------------------------------------


# Push patterns chosen to exercise every axis combination: cells
# distributed across the 3-D lattice so every 2³ sub-cube boundary
# straddles a real push. Each value is a small Z-set of edges.
JOIN_PUSH_PATTERNS: list[
    tuple[
        dict[tuple[int, int, int], ZSet[Edge]],
        dict[tuple[int, int, int], ZSet[Edge]],
    ]
] = [
    # Single-cell impulses at the origin.
    (
        {(0, 0, 0): ZSet({(0, 1): 1, (1, 2): 1, (2, 3): 1})},
        {(0, 0, 0): ZSet({(1, 10): 1, (2, 20): 1, (3, 30): 1})},
    ),
    # a at origin, b spread across the lattice — exercises terms
    # where b carries integration on every axis.
    (
        {(0, 0, 0): ZSet({(0, 1): 1, (1, 2): 1})},
        {
            (0, 0, 0): ZSet({(1, 100): 1}),
            (1, 0, 0): ZSet({(1, 101): 1}),
            (0, 1, 0): ZSet({(1, 102): 1}),
            (0, 0, 1): ZSet({(2, 200): 1}),
            (1, 1, 1): ZSet({(2, 201): 1}),
        },
    ),
    # Both a and b spread; non-unit weights and a few negatives to
    # exercise the bilinear cancellations.
    (
        {
            (0, 0, 0): ZSet({(0, 1): 2, (1, 2): 1}),
            (1, 0, 0): ZSet({(0, 1): -1}),
            (0, 1, 0): ZSet({(2, 3): 1}),
            (0, 0, 1): ZSet({(1, 2): 1}),
            (1, 1, 0): ZSet({(3, 4): 1}),
            (1, 0, 1): ZSet({(0, 1): 1}),
        },
        {
            (0, 0, 0): ZSet({(1, 10): 1, (2, 20): 1}),
            (1, 1, 1): ZSet({(3, 30): 1, (4, 40): 1}),
            (0, 1, 1): ZSet({(1, 11): 2}),
            (1, 0, 0): ZSet({(2, 21): 1}),
        },
    ),
]


@pytest.mark.parametrize(
    "pushes_a,pushes_b",
    JOIN_PUSH_PATTERNS,
    ids=["origin-impulses", "a-impulse-b-spread", "both-spread"],
)
def test_join_matches_triple_delta_of_cumulative(
    pushes_a: dict[tuple[int, int, int], ZSet[Edge]],
    pushes_b: dict[tuple[int, int, int], ZSet[Edge]],
) -> None:
    """Push diff streams ``a`` and ``b`` at scattered 3-D cells, then
    assert that the 8-term operator's output at every cell of the
    enclosing bounding cube equals the 3-D inclusion-exclusion delta
    of the cumulative join.

    This is the pinning test for the derivation: if the 8-term formula
    is correct, the operator agrees with the ground-truth Δ³c at
    *every* cell, for *every* push pattern."""
    eg = ZSetAddition[Edge]()
    pred = lambda l, r: l[1] == r[0]
    proj = lambda l, r: (l[0], r[1])

    e = _eval_3d()
    a_in, b_in, joined = _wire_join(e, pred=pred, proj=proj)
    for t, v in pushes_a.items():
        e.push(a_in, v, t=t)
    for t, v in pushes_b.items():
        e.push(b_in, v, t=t)

    # Bound the inspection cube one past the largest pushed coord on
    # each axis. Δ³ at (o, s, i) reads cells up to (o, s, i); checking
    # one beyond max ensures we cover the trailing identity cells too.
    max_o = max((t[0] for t in (*pushes_a, *pushes_b)), default=0) + 1
    max_s = max((t[1] for t in (*pushes_a, *pushes_b)), default=0) + 1
    max_i = max((t[2] for t in (*pushes_a, *pushes_b)), default=0) + 1

    for o in range(max_o + 1):
        for s in range(max_s + 1):
            for i in range(max_i + 1):
                got = e.read(joined, (o, s, i))
                want = _expected_delta_join(
                    pushes_a,
                    pushes_b,
                    o,
                    s,
                    i,
                    eg,
                    pred,
                    proj,
                )
                assert got == want, f"join mismatch at ({o}, {s}, {i}): got {got.inner}, want {want.inner}"


def test_join_matches_triple_delta_under_random_pushes() -> None:
    """Property-style: random small push patterns over a 3³ cube,
    comparing the 8-term sum against the inclusion-exclusion ground
    truth at every cell. Catches any axis-permutation bug that the
    handcrafted patterns might miss."""
    rng = random.Random(0xC0FFEE)
    eg = ZSetAddition[Edge]()
    pred = lambda l, r: l[1] == r[0]
    proj = lambda l, r: (l[0], r[1])

    for trial in range(5):
        pushes_a: dict[tuple[int, int, int], ZSet[Edge]] = {}
        pushes_b: dict[tuple[int, int, int], ZSet[Edge]] = {}
        for o in range(2):
            for s in range(2):
                for i in range(2):
                    if rng.random() < 0.5:
                        pushes_a[(o, s, i)] = ZSet(
                            {
                                (rng.randint(0, 3), rng.randint(0, 3)): rng.choice([-1, 1, 2]),
                            }
                        )
                    if rng.random() < 0.5:
                        pushes_b[(o, s, i)] = ZSet(
                            {
                                (rng.randint(0, 3), rng.randint(0, 3)): rng.choice([-1, 1, 2]),
                            }
                        )

        e = _eval_3d()
        a_in, b_in, joined = _wire_join(e, pred=pred, proj=proj)
        for t, v in pushes_a.items():
            e.push(a_in, v, t=t)
        for t, v in pushes_b.items():
            e.push(b_in, v, t=t)

        for o in range(3):
            for s in range(3):
                for i in range(3):
                    got = e.read(joined, (o, s, i))
                    want = _expected_delta_join(
                        pushes_a,
                        pushes_b,
                        o,
                        s,
                        i,
                        eg,
                        pred,
                        proj,
                    )
                    assert got == want, f"trial {trial}, cell ({o}, {s}, {i}): got {got.inner}, want {want.inner}"


# ---- Distinct pinning ------------------------------------------------------


DISTINCT_PUSH_PATTERNS: list[dict[tuple[int, int, int], ZSet[Edge]]] = [
    {(0, 0, 0): ZSet({(0, 1): 1, (1, 2): 1, (2, 3): 1})},
    {
        (0, 0, 0): ZSet({(0, 1): 1}),
        (1, 0, 0): ZSet({(0, 1): 1}),  # duplicate already-positive
        (0, 1, 0): ZSet({(2, 3): -1}),  # retract a non-existent entry
        (0, 0, 1): ZSet({(0, 1): -2}),  # cross back below threshold
    },
    {
        (0, 0, 0): ZSet({(0, 1): 1, (1, 2): 1}),
        (1, 1, 1): ZSet({(0, 1): -1, (2, 3): 1}),
        (0, 1, 0): ZSet({(1, 2): -1}),
    },
]


@pytest.mark.parametrize(
    "pushes",
    DISTINCT_PUSH_PATTERNS,
    ids=["single-positive-impulse", "duplicates-and-retractions", "mixed-deltas"],
)
def test_distinct_matches_triple_delta_of_distinct_of_cumulative(
    pushes: dict[tuple[int, int, int], ZSet[Edge]],
) -> None:
    """The 3-D distinct's output at each cell should equal the 3-D
    inclusion-exclusion delta of ``distinct(Iᵒ Iˢ Iⁱ s)``."""
    eg = ZSetAddition[Edge]()
    e = _eval_3d()
    src, out = _wire_distinct(e)
    for t, v in pushes.items():
        e.push(src, v, t=t)

    max_o = max((t[0] for t in pushes), default=0) + 1
    max_s = max((t[1] for t in pushes), default=0) + 1
    max_i = max((t[2] for t in pushes), default=0) + 1

    for o in range(max_o + 1):
        for s in range(max_s + 1):
            for i in range(max_i + 1):
                got = e.read(out, (o, s, i))
                want = _expected_delta_distinct(pushes, o, s, i, eg)
                assert got == want, f"distinct mismatch at ({o}, {s}, {i}): got {got.inner}, want {want.inner}"
