"""Parity tests for the 3-D Datalog bodies in **streaming mode**.

``test_3d_datalog.py`` and ``test_3d_datalog_negation.py`` push the
program + EDB once at outer 0 and saturate the inner axis (single
batch). This file extends coverage to the streaming case: push some
EDB at outer 0, saturate, then push *more* EDB at outer 1 (and 2, …)
and saturate again. The 3-D bodies should match the 2-D bodies
per-(outer, inner) cell across all batches.

Reasoning. With the stratum axis held at 0 and inputs only varying
on ``(outer, inner)`` (the 3-D streams are δ-impulses on the stratum
axis at ``s = 0``), Δ³c collapses cleanly:

    I^o I^s I^i a (o, 0, i) = I^o I^i a' (o, i)   for s ≥ 0

where ``a'`` is the underlying 1-D delta lifted to 2-D. Δ³c at
``(o, 0, i)`` then equals the 2-D double delta on ``(o, i)`` — the
``s = −1`` corners of the 3-D inclusion-exclusion are identity. So
the 3-D body at ``(o, 0, i)`` must agree with the 2-D body at
``(o, i)`` for every outer batch."""

from __future__ import annotations

from typing import NamedTuple

import pytest

from pydbsp import datalog as dlg
from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.core import Antichain, dbsp_time
from pydbsp.datalog_stratified import IncrementalDatalogBodyWithNegation
from pydbsp.evaluate import Evaluator
from pydbsp.indexed_relational_operators import (
    IndexedIncrementalDatalogBody,
    IndexedIncrementalDatalogWithNegationBody,
)
from pydbsp.operator import Input, LiftStreamIntroduction
from pydbsp.progress import NodeId, Time
from pydbsp.storage import DictStorage
from pydbsp.zset import ZSet, ZSetAddition


# ---- Wirings ---------------------------------------------------------------


class Wiring2D(NamedTuple):
    e: Evaluator[Time]
    edb_1d: NodeId
    program_1d: NodeId
    state_facts: NodeId
    state_rewrites: NodeId
    seed_1d: NodeId
    body_out: NodeId
    fact_group: ZSetAddition[dlg.Fact]


class Wiring3D(NamedTuple):
    e: Evaluator[Time]
    edb_1d: NodeId
    program_1d: NodeId
    state_facts: NodeId
    state_rewrites: NodeId
    seed_1d: NodeId
    body_out: NodeId
    fact_group: ZSetAddition[dlg.Fact]


def _build_2d(*, negation: bool) -> Wiring2D:
    fact_group = ZSetAddition[dlg.Fact]()
    rewrite_group = ZSetAddition[dlg.ProvenanceIndexedRewrite]()
    rule_group = ZSetAddition[dlg.Rule]()
    e = Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(2)),
        group=fact_group,
    )
    edb_1d = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    program_1d = Input[ZSet[dlg.Rule]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    state_facts_2d = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(2))).connect(e.circuit, ())
    state_rewrites_2d = Input[ZSet[dlg.ProvenanceIndexedRewrite]](frontier=Antichain(dbsp_time(2))).connect(
        e.circuit, ()
    )
    seed_1d = Input[ZSet[dlg.ProvenanceIndexedRewrite]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    program_2d = LiftStreamIntroduction[ZSet[dlg.Rule]](group=rule_group).connect(e.circuit, (program_1d,))
    body_cls = IndexedIncrementalDatalogWithNegationBody if negation else IndexedIncrementalDatalogBody
    body = body_cls(
        fact_group=fact_group,
        rule_group=rule_group,
        rewrite_group=rewrite_group,
        signal_group=ZSetAddition[dlg.Signal](),
        ext_dir_group=ZSetAddition[dlg.ExtendedDirection](),
        jorder_group=ZSetAddition[tuple[str, dlg.ColumnReference]](),
        gatekeep_group=ZSetAddition[dlg.IndexedGatekeepEntry](),
        indexed_fact_group=ZSetAddition[dlg.IndexedFact](),
    ).connect(
        e.circuit,
        (edb_1d, program_2d, state_facts_2d, state_rewrites_2d, seed_1d),
    )
    return Wiring2D(
        e=e,
        edb_1d=edb_1d,
        program_1d=program_1d,
        state_facts=state_facts_2d,
        state_rewrites=state_rewrites_2d,
        seed_1d=seed_1d,
        body_out=body,
        fact_group=fact_group,
    )


def _build_3d(*, negation: bool) -> Wiring3D:
    fact_group = ZSetAddition[dlg.Fact]()
    rewrite_group = ZSetAddition[dlg.ProvenanceIndexedRewrite]()
    rule_group = ZSetAddition[dlg.Rule]()
    e = Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(3)),
        group=fact_group,
    )
    edb_1d = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    program_1d = Input[ZSet[dlg.Rule]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    state_facts_3d = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(3))).connect(e.circuit, ())
    state_rewrites_3d = Input[ZSet[dlg.ProvenanceIndexedRewrite]](frontier=Antichain(dbsp_time(3))).connect(
        e.circuit, ()
    )
    seed_1d = Input[ZSet[dlg.ProvenanceIndexedRewrite]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    program_2d_step = LiftStreamIntroduction[ZSet[dlg.Rule]](group=rule_group).connect(e.circuit, (program_1d,))
    program_3d = LiftStreamIntroduction[ZSet[dlg.Rule]](group=rule_group).connect(e.circuit, (program_2d_step,))
    # The 3-D negation body subsumes the positive body — on programs
    # without negation atoms the anti-product fires nothing.
    _ = negation
    body = IncrementalDatalogBodyWithNegation(
        fact_group=fact_group,
        rule_group=rule_group,
        rewrite_group=rewrite_group,
        signal_group=ZSetAddition[dlg.Signal](),
        dir_group=ZSetAddition[dlg.Direction](),
        gatekeep_group=(ZSetAddition[dlg.AtomWithSourceRewriteAndProvenance]()),
    ).connect(
        e.circuit,
        (edb_1d, program_3d, state_facts_3d, state_rewrites_3d, seed_1d),
    )
    return Wiring3D(
        e=e,
        edb_1d=edb_1d,
        program_1d=program_1d,
        state_facts=state_facts_3d,
        state_rewrites=state_rewrites_3d,
        seed_1d=seed_1d,
        body_out=body,
        fact_group=fact_group,
    )


# ---- Streaming drivers -----------------------------------------------------


def _stream_2d(
    w: Wiring2D,
    program: ZSet[dlg.Rule],
    edb_batches: list[ZSet[dlg.Fact]],
    max_k: int = 64,
) -> list[list[tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]]]]:
    """Drive the 2-D body across multiple outer ticks. ``edb_batches[t]``
    is pushed at outer ``t``. The program is pushed once at outer 0
    (and stays cumulative). Returns a list-of-lists: per outer tick,
    the per-inner-tick ``(diff_facts, diff_rewrites)`` until convergence."""
    w.e.push(w.program_1d, program)
    w.e.push(w.seed_1d, ZSet({(0, dlg._rewrite_monoid.identity()): 1}))
    out: list[list[tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]]]] = []
    deepest_inner_so_far = -1
    for t, edb in enumerate(edb_batches):
        w.e.push(w.edb_1d, edb, t=(t,))
        # min_inner: the deepest inner k at which any prior outer
        # pushed state. We must drive past this depth before declaring
        # convergence at outer t, so the cross-outer integrals
        # propagate.
        min_inner = deepest_inner_so_far if t > 0 else -1
        per_inner: list[tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]]] = []
        for k in range(max_k):
            diff_facts, diff_rewrites = w.e.read(w.body_out, (t, k))
            empty = not diff_facts.inner and not diff_rewrites.inner
            if empty and k > min_inner:
                break
            per_inner.append((diff_facts, diff_rewrites))
            w.e.push(w.state_facts, diff_facts, t=(t, k))
            w.e.push(w.state_rewrites, diff_rewrites, t=(t, k))
        out.append(per_inner)
        if per_inner:
            deepest_inner_so_far = max(deepest_inner_so_far, len(per_inner) - 1)
    return out


def _stream_3d(
    w: Wiring3D,
    program: ZSet[dlg.Rule],
    edb_batches: list[ZSet[dlg.Fact]],
    max_k: int = 64,
) -> list[list[tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]]]]:
    """Same as :func:`_stream_2d` but on the 3-D body at ``(t, 0, k)``.
    The stratum axis is held at 0; ``state_facts_3d`` is pushed at
    ``(t, 0, k)``."""
    w.e.push(w.program_1d, program)
    w.e.push(w.seed_1d, ZSet({(0, dlg._rewrite_monoid.identity()): 1}))
    out: list[list[tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]]]] = []
    deepest_inner_so_far = -1
    for t, edb in enumerate(edb_batches):
        w.e.push(w.edb_1d, edb, t=(t,))
        min_inner = deepest_inner_so_far if t > 0 else -1
        per_inner: list[tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]]] = []
        for k in range(max_k):
            diff_facts, diff_rewrites = w.e.read(w.body_out, (t, 0, k))
            empty = not diff_facts.inner and not diff_rewrites.inner
            if empty and k > min_inner:
                break
            per_inner.append((diff_facts, diff_rewrites))
            w.e.push(w.state_facts, diff_facts, t=(t, 0, k))
            w.e.push(w.state_rewrites, diff_rewrites, t=(t, 0, k))
        out.append(per_inner)
        if per_inner:
            deepest_inner_so_far = max(deepest_inner_so_far, len(per_inner) - 1)
    return out


# ---- Programs / batches ----------------------------------------------------


_TC = ZSet(
    {
        (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
        (
            ("reach", ("?X", "?Z")),
            ("edge", ("?X", "?Y")),
            ("reach", ("?Y", "?Z")),
        ): 1,
    }
)


# A negation program where new EDB across batches can only extend the
# answer monotonically — no cycles through negation, so retraction is
# not needed and the streaming semantics match plain Datalog with `!`.
_TC_WITH_FILTER = ZSet(
    {
        (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
        (
            ("reach", ("?X", "?Z")),
            ("edge", ("?X", "?Y")),
            ("reach", ("?Y", "?Z")),
        ): 1,
        # `live(X)` iff `node(X)` and `X` is not in `dead`.
        (("live", ("?X",)), ("node", ("?X",)), ("!dead", ("?X",))): 1,
    }
)


# ---- Streaming parity cases ------------------------------------------------


# Each case lists EDB batches (one per outer tick). The 2-D and 3-D
# bodies must agree per-(outer, inner) cell across every batch.
POSITIVE_STREAMING_CASES: list[tuple[str, ZSet[dlg.Rule], list[ZSet[dlg.Fact]]]] = [
    (
        "tc-two-batches",
        _TC,
        [
            ZSet({("edge", (0, 1)): 1, ("edge", (1, 2)): 1}),
            ZSet({("edge", (2, 3)): 1}),
        ],
    ),
    (
        "tc-three-batches-with-retraction",
        _TC,
        [
            ZSet({("edge", (0, 1)): 1, ("edge", (1, 2)): 1}),
            ZSet({("edge", (2, 3)): 1, ("edge", (3, 0)): 1}),
            ZSet({("edge", (1, 2)): -1}),  # retract one edge
        ],
    ),
    (
        "tc-empty-second-batch",
        _TC,
        [
            ZSet({("edge", (0, 1)): 1, ("edge", (1, 2)): 1}),
            ZSet({}),
        ],
    ),
]


NEGATION_STREAMING_CASES: list[tuple[str, ZSet[dlg.Rule], list[ZSet[dlg.Fact]]]] = [
    (
        "tc-with-filter-two-batches",
        _TC_WITH_FILTER,
        [
            ZSet(
                {
                    ("edge", (0, 1)): 1,
                    ("edge", (1, 2)): 1,
                    ("node", (0,)): 1,
                    ("node", (1,)): 1,
                    ("node", (2,)): 1,
                    ("dead", (1,)): 1,
                }
            ),
            ZSet(
                {
                    ("edge", (2, 3)): 1,
                    ("node", (3,)): 1,
                }
            ),
        ],
    ),
    (
        # Add a new dead node in batch 2 — a `live` row from batch 1
        # must retract. Tests that the anti-product subtraction is
        # bilinear across the outer axis too.
        "anti-join-with-blocker-arriving-late",
        ZSet(
            {
                (("live", ("?X",)), ("node", ("?X",)), ("!dead", ("?X",))): 1,
            }
        ),
        [
            ZSet(
                {
                    ("node", ("alice",)): 1,
                    ("node", ("bob",)): 1,
                }
            ),
            ZSet({("dead", ("alice",)): 1}),
        ],
    ),
]


# ---- The parity asserter ---------------------------------------------------


def _assert_parity(
    label: str,
    batches_2d: list[list[tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]]]],
    batches_3d: list[list[tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]]]],
) -> None:
    assert len(batches_2d) == len(batches_3d), (
        f"{label}: batch count differs: 2-D={len(batches_2d)}, 3-D={len(batches_3d)}"
    )
    for t, (b2, b3) in enumerate(zip(batches_2d, batches_3d)):
        assert len(b2) == len(b3), f"{label}: outer {t}: inner-tick count differs: 2-D={len(b2)}, 3-D={len(b3)}"
        for k, ((f2, r2), (f3, r3)) in enumerate(zip(b2, b3)):
            assert f2 == f3, f"{label}: outer {t}, inner {k}: diff_facts mismatch: 2-D={f2.inner}, 3-D={f3.inner}"
            assert r2 == r3, f"{label}: outer {t}, inner {k}: diff_rewrites mismatch"


@pytest.mark.parametrize(
    "label,program,batches",
    POSITIVE_STREAMING_CASES,
    ids=[c[0] for c in POSITIVE_STREAMING_CASES],
)
def test_3d_positive_body_streaming_matches_2d(
    label: str,
    program: ZSet[dlg.Rule],
    batches: list[ZSet[dlg.Fact]],
) -> None:
    """Drive the positive 2-D body and the 3-D body across the same
    sequence of EDB batches. Per-(outer, inner) outputs must agree
    cell-by-cell across all batches."""
    w2 = _build_2d(negation=False)
    w3 = _build_3d(negation=False)
    b2 = _stream_2d(w2, program, batches)
    b3 = _stream_3d(w3, program, batches)
    _assert_parity(label, b2, b3)


@pytest.mark.parametrize(
    "label,program,batches",
    NEGATION_STREAMING_CASES,
    ids=[c[0] for c in NEGATION_STREAMING_CASES],
)
def test_3d_negation_body_streaming_matches_2d(
    label: str,
    program: ZSet[dlg.Rule],
    batches: list[ZSet[dlg.Fact]],
) -> None:
    """Same as the positive parity but for the negation bodies. The
    anti-product algebra has to be bilinear on the outer axis too —
    a `!dead(X)` row pushed at outer 1 must correctly retract a
    `live(X)` derived at outer 0."""
    w2 = _build_2d(negation=True)
    w3 = _build_3d(negation=True)
    b2 = _stream_2d(w2, program, batches)
    b3 = _stream_3d(w3, program, batches)
    _assert_parity(label, b2, b3)


# ---- Fact retraction across outer ticks ------------------------------------
#
# The 3-D bodies (positive and negation) unsay derived facts correctly
# when EDB support is retracted at a later outer — same `D^o` mechanism
# as the 2-D bodies.


def test_3d_positive_body_fact_retraction() -> None:
    """Outer 0 derives b(1) from p(1); outer 1 retracts p(1). Sum of
    body outer-deltas at outer 0..1 should equal what a fresh
    evaluator on the empty cumulative EDB would produce — nothing."""
    program = ZSet({(("b", ("?X",)), ("p", ("?X",))): 1})
    edb_0 = ZSet({("p", (1,)): 1})
    edb_1 = ZSet({("p", (1,)): -1})

    w = _build_3d(negation=False)
    batches = _stream_3d(w, program, [edb_0, edb_1])
    streamed = w.fact_group.identity()
    for per_inner in batches:
        for diff_facts, _ in per_inner:
            streamed = w.fact_group.add(streamed, diff_facts)

    expected = w.fact_group.identity()
    assert streamed == expected, (
        f"3-D positive body fact retraction:\n  streamed: {dict(streamed.inner)}\n  expected: {dict(expected.inner)}"
    )


def test_3d_negation_body_fact_retraction() -> None:
    """``alive(?X) :- person(?X), !dead(?X)`` with person(alice) at
    outer 0; retract at outer 1. Streamed cumulative should match
    fresh-on-empty (nothing)."""
    program = ZSet(
        {
            (
                ("alive", ("?X",)),
                ("person", ("?X",)),
                ("!dead", ("?X",)),
            ): 1,
        }
    )
    edb_0 = ZSet({("person", ("alice",)): 1})
    edb_1 = ZSet({("person", ("alice",)): -1})

    w = _build_3d(negation=True)
    batches = _stream_3d(w, program, [edb_0, edb_1])
    streamed = w.fact_group.identity()
    for per_inner in batches:
        for diff_facts, _ in per_inner:
            streamed = w.fact_group.add(streamed, diff_facts)

    expected = w.fact_group.identity()
    assert streamed == expected, (
        f"3-D negation body fact retraction:\n  streamed: {dict(streamed.inner)}\n  expected: {dict(expected.inner)}"
    )


# ---- Rule retraction across outer ticks ------------------------------------
#
# Push a rule + EDB at outer 0 (deriving some facts). At outer 1, push
# the rule with weight −1. The streamed cumulative matches a fresh
# evaluator on the empty post-retraction program (just the EDB). This
# used to fail because `dlg.sig` and `dlg.dir` dropped negative-weight
# rules; fixed at the value layer.


def _stream_3d_with_rule_deltas(
    w: Wiring3D,
    rule_batches: list[ZSet[dlg.Rule]],
    edb_batches: list[ZSet[dlg.Fact]],
    max_k: int = 64,
) -> list[list[tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]]]]:
    """Like :func:`_stream_3d` but accepts a rule delta per outer
    instead of pushing the full program once at outer 0."""
    w.e.push(w.seed_1d, ZSet({(0, dlg._rewrite_monoid.identity()): 1}))
    out: list[list[tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]]]] = []
    deepest_inner_so_far = -1
    for t, (rules, edb) in enumerate(zip(rule_batches, edb_batches, strict=True)):
        if rules.inner:
            w.e.push(w.program_1d, rules, t=(t,))
        if edb.inner:
            w.e.push(w.edb_1d, edb, t=(t,))
        min_inner = deepest_inner_so_far if t > 0 else -1
        per_inner: list[tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]]] = []
        for k in range(max_k):
            diff_facts, diff_rewrites = w.e.read(w.body_out, (t, 0, k))
            empty = not diff_facts.inner and not diff_rewrites.inner
            if empty and k > min_inner:
                break
            per_inner.append((diff_facts, diff_rewrites))
            w.e.push(w.state_facts, diff_facts, t=(t, 0, k))
            w.e.push(w.state_rewrites, diff_rewrites, t=(t, 0, k))
        out.append(per_inner)
        if per_inner:
            deepest_inner_so_far = max(deepest_inner_so_far, len(per_inner) - 1)
    return out


def test_3d_positive_body_rule_retraction() -> None:
    """Outer 0: push rule b :- p + EDB p(1) → derives b(1). Outer 1:
    push the rule with weight −1. Streamed cumulative should equal
    the EDB only — no b(1)."""
    rule = (("b", ("?X",)), ("p", ("?X",)))
    edb = ZSet({("p", (1,)): 1})

    w = _build_3d(negation=False)
    batches = _stream_3d_with_rule_deltas(
        w,
        [ZSet({rule: 1}), ZSet({rule: -1})],
        [edb, ZSet({})],
    )
    streamed = w.fact_group.identity()
    for per_inner in batches:
        for diff_facts, _ in per_inner:
            streamed = w.fact_group.add(streamed, diff_facts)

    expected = edb
    assert streamed == expected, (
        f"3-D positive body rule retraction:\n  streamed: {dict(streamed.inner)}\n  expected: {dict(expected.inner)}"
    )


def test_3d_negation_body_rule_retraction() -> None:
    """``alive(?X) :- person(?X), !dead(?X)`` derives alive(alice) at
    outer 0. Retract the rule at outer 1. Streamed should equal the
    EDB only — no alive(alice)."""
    rule = (
        ("alive", ("?X",)),
        ("person", ("?X",)),
        ("!dead", ("?X",)),
    )
    edb = ZSet({("person", ("alice",)): 1})

    w = _build_3d(negation=True)
    batches = _stream_3d_with_rule_deltas(
        w,
        [ZSet({rule: 1}), ZSet({rule: -1})],
        [edb, ZSet({})],
    )
    streamed = w.fact_group.identity()
    for per_inner in batches:
        for diff_facts, _ in per_inner:
            streamed = w.fact_group.add(streamed, diff_facts)

    expected = edb
    assert streamed == expected, (
        f"3-D negation body rule retraction:\n  streamed: {dict(streamed.inner)}\n  expected: {dict(expected.inner)}"
    )
