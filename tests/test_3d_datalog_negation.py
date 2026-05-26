"""Parity tests for the 3-D negation-aware Datalog body.

:class:`pydbsp.datalog_stratified.IncrementalDatalogBodyWithNegation`
is the 3-D analog of
:class:`pydbsp.indexed_relational_operators.IndexedIncrementalDatalogWithNegationBody`
on the ``(outer, stratum, inner)`` lattice. When both axes added at
positions 0 and 1 are held at 0, the 3-D body should produce the
**same per-(inner)** output as the 2-D body produces per-(outer=0,
inner). These programs include negation but no cycle-through-negation
that would require stratification — they converge correctly at a
single stratum."""

from __future__ import annotations

from typing import NamedTuple

import pytest

from pydbsp import datalog as dlg
from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.core import Antichain, dbsp_time
from pydbsp.datalog_stratified import IncrementalDatalogBodyWithNegation
from pydbsp.evaluate import Evaluator
from pydbsp.operator import Input, LiftStreamIntroduction
from pydbsp.progress import NodeId, Time
from pydbsp.indexed_relational_operators import (
    IndexedIncrementalDatalogWithNegationBody,
)
from pydbsp.storage import DictStorage
from pydbsp.zset import ZSet, ZSetAddition


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


def _build_2d() -> Wiring2D:
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
    body = IndexedIncrementalDatalogWithNegationBody(
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


def _build_3d() -> Wiring3D:
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


def _drive_2d(
    w: Wiring2D,
    edb: ZSet[dlg.Fact],
    program: ZSet[dlg.Rule],
    max_k: int = 64,
) -> tuple[list[ZSet[dlg.Fact]], list[ZSet[dlg.ProvenanceIndexedRewrite]]]:
    w.e.push(w.edb_1d, edb)
    w.e.push(w.program_1d, program)
    w.e.push(w.seed_1d, ZSet({(0, dlg._rewrite_monoid.identity()): 1}))
    diffs_facts: list[ZSet[dlg.Fact]] = []
    diffs_rewrites: list[ZSet[dlg.ProvenanceIndexedRewrite]] = []
    for k in range(max_k):
        diff_facts, diff_rewrites = w.e.read(w.body_out, (0, k))
        if not diff_facts.inner and not diff_rewrites.inner:
            break
        diffs_facts.append(diff_facts)
        diffs_rewrites.append(diff_rewrites)
        w.e.push(w.state_facts, diff_facts, t=(0, k))
        w.e.push(w.state_rewrites, diff_rewrites, t=(0, k))
    return diffs_facts, diffs_rewrites


def _drive_3d(
    w: Wiring3D,
    edb: ZSet[dlg.Fact],
    program: ZSet[dlg.Rule],
    max_k: int = 64,
) -> tuple[list[ZSet[dlg.Fact]], list[ZSet[dlg.ProvenanceIndexedRewrite]]]:
    w.e.push(w.edb_1d, edb)
    w.e.push(w.program_1d, program)
    w.e.push(w.seed_1d, ZSet({(0, dlg._rewrite_monoid.identity()): 1}))
    diffs_facts: list[ZSet[dlg.Fact]] = []
    diffs_rewrites: list[ZSet[dlg.ProvenanceIndexedRewrite]] = []
    for k in range(max_k):
        diff_facts, diff_rewrites = w.e.read(w.body_out, (0, 0, k))
        if not diff_facts.inner and not diff_rewrites.inner:
            break
        diffs_facts.append(diff_facts)
        diffs_rewrites.append(diff_rewrites)
        w.e.push(w.state_facts, diff_facts, t=(0, 0, k))
        w.e.push(w.state_rewrites, diff_rewrites, t=(0, 0, k))
    return diffs_facts, diffs_rewrites


# ---- Negation test programs (no cycle-through-negation needed) ------------


PARITY_CASES: list[tuple[str, ZSet[dlg.Rule], ZSet[dlg.Fact]]] = [
    (
        # Plain anti-join: derive `alive(X)` for every `person(X)`
        # except the ones marked `dead`. No recursion.
        "anti-join-no-blocker",
        ZSet(
            {
                (
                    ("alive", ("?X",)),
                    ("person", ("?X",)),
                    ("!dead", ("?X",)),
                ): 1,
            }
        ),
        ZSet(
            {
                ("person", ("alice",)): 1,
                ("person", ("bob",)): 1,
            }
        ),
    ),
    (
        "anti-join-with-blocker",
        ZSet(
            {
                (
                    ("alive", ("?X",)),
                    ("person", ("?X",)),
                    ("!dead", ("?X",)),
                ): 1,
            }
        ),
        ZSet(
            {
                ("person", ("alice",)): 1,
                ("person", ("bob",)): 1,
                ("dead", ("bob",)): 1,
            }
        ),
    ),
    (
        # Recursive positive layered with an anti-join, but the
        # negation does NOT close a cycle. ``reach`` saturates first,
        # then ``unreachable`` reads it via negation. Safe at single
        # stratum.
        "positive-recursion-then-anti-join",
        ZSet(
            {
                (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
                (
                    ("reach", ("?X", "?Z")),
                    ("edge", ("?X", "?Y")),
                    ("reach", ("?Y", "?Z")),
                ): 1,
                (
                    ("only-reaches-self", ("?X",)),
                    ("node", ("?X",)),
                    ("!reach", ("?X", "?Y")),
                ): 1,
            }
        ),
        ZSet(
            {
                # 0 -> 1, 1 -> 2: 0 reaches 1 and 2, 1 reaches 2, 2 is alone.
                ("edge", (0, 1)): 1,
                ("edge", (1, 2)): 1,
                ("node", (0,)): 1,
                ("node", (1,)): 1,
                ("node", (2,)): 1,
            }
        ),
    ),
    (
        # Two-layer anti-join with no recursion: `mid` filters `bot`,
        # then `top` filters `mid`. Both layers converge in a single
        # stratum because neither references its own head.
        "two-layer-anti-join",
        ZSet(
            {
                (
                    ("mid", ("?X",)),
                    ("bot", ("?X",)),
                    ("!excluded", ("?X",)),
                ): 1,
                (
                    ("top", ("?X",)),
                    ("mid", ("?X",)),
                    ("!blocked", ("?X",)),
                ): 1,
            }
        ),
        ZSet(
            {
                ("bot", (0,)): 1,
                ("bot", (1,)): 1,
                ("bot", (2,)): 1,
                ("excluded", (2,)): 1,
                ("blocked", (1,)): 1,
            }
        ),
    ),
    (
        # No negation at all — should behave exactly like the positive
        # body. Sanity check that the negation algebra collapses to
        # the positive case when no `!`-prefixed atoms appear.
        "no-negation-at-all",
        ZSet(
            {
                (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
                (
                    ("reach", ("?X", "?Z")),
                    ("edge", ("?X", "?Y")),
                    ("reach", ("?Y", "?Z")),
                ): 1,
            }
        ),
        ZSet(
            {
                ("edge", (0, 1)): 1,
                ("edge", (1, 2)): 1,
                ("edge", (2, 0)): 1,
            }
        ),
    ),
]


@pytest.mark.parametrize(
    "label,program,edb",
    PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_3d_negation_body_matches_2d_negation_body_per_tick(
    label: str,
    program: ZSet[dlg.Rule],
    edb: ZSet[dlg.Fact],
) -> None:
    """The 3-D negation body, driven at ``(0, 0, k)``, must produce
    the same per-tick ``(diff_facts, diff_rewrites)`` as the 2-D
    negation body driven at ``(0, k)``. Same inputs, same convergence
    depth, same outputs."""
    w2 = _build_2d()
    w3 = _build_3d()
    facts_2d, rewrites_2d = _drive_2d(w2, edb, program)
    facts_3d, rewrites_3d = _drive_3d(w3, edb, program)

    assert len(facts_3d) == len(facts_2d), (
        f"{label}: convergence depth differs: 2-D={len(facts_2d)}, 3-D={len(facts_3d)}"
    )
    for k, (f2, f3) in enumerate(zip(facts_2d, facts_3d)):
        assert f2 == f3, f"{label}: diff_facts mismatch at inner tick {k}: 2-D={f2.inner}, 3-D={f3.inner}"
    for k, (r2, r3) in enumerate(zip(rewrites_2d, rewrites_3d)):
        assert r2 == r3, f"{label}: diff_rewrites mismatch at inner tick {k}"

    cum_2d = w2.fact_group.identity()
    for f in facts_2d:
        cum_2d = w2.fact_group.add(cum_2d, f)
    cum_3d = w3.fact_group.identity()
    for f in facts_3d:
        cum_3d = w3.fact_group.add(cum_3d, f)
    assert cum_2d == cum_3d
