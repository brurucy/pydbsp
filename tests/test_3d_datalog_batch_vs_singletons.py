"""Batch-vs-singletons invariant for the 3-D Datalog bodies.

Mirrors the existing ``test_datalog_batch_eq_singletons_*`` cluster in
:mod:`tests.test_integration`. For each ``(program, edb)`` pair, the
cumulative derived-facts set must be the same whether we

* push the whole ``edb`` at one outer tick and saturate once (batch),
  or
* push one ``(fact, weight)`` per outer tick, saturate at each tick,
  and sum the per-outer outer-deltas (singletons).

This is the streaming-bilinear invariant: an incremental algebra must
agree with itself across any partition of the input on the outer
axis. The 2-D bodies satisfy it; the 3-D bodies must too if they are
to substitute for the 2-D bodies in production pipelines."""

from __future__ import annotations

from typing import NamedTuple

from pydbsp import datalog as dlg
from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.core import Antichain, dbsp_time
from pydbsp.datalog_stratified import IncrementalDatalogBodyWithNegation
from pydbsp.evaluate import Evaluator
from pydbsp.operator import Input, LiftStreamIntroduction
from pydbsp.progress import NodeId, Time
from pydbsp.storage import DictStorage
from pydbsp.zset import ZSet, ZSetAddition


class Wiring(NamedTuple):
    e: Evaluator[Time]
    edb_1d: NodeId
    program_1d: NodeId
    state_facts: NodeId
    state_rewrites: NodeId
    seed_1d: NodeId
    body_out: NodeId
    fact_group: ZSetAddition[dlg.Fact]


def _build_3d() -> Wiring:
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
    return Wiring(
        e=e,
        edb_1d=edb_1d,
        program_1d=program_1d,
        state_facts=state_facts_3d,
        state_rewrites=state_rewrites_3d,
        seed_1d=seed_1d,
        body_out=body,
        fact_group=fact_group,
    )


def _saturate_3d_at_outer(
    w: Wiring,
    edb_delta: ZSet[dlg.Fact],
    program_delta: ZSet[dlg.Rule],
    outer_tick: int,
    *,
    push_seed: bool,
    max_k: int = 64,
) -> ZSet[dlg.Fact]:
    """Push the per-outer-tick deltas, drive the inner fixpoint at
    ``(outer_tick, 0, k)``, and return the cumulative ``diff_facts``
    contributed by this outer tick.

    Driver structure mirrors :func:`tests.test_integration._saturate_datalog`
    point-for-point, just on the 3-D lattice with the stratum axis
    held at 0."""
    w.e.push(w.edb_1d, edb_delta, t=(outer_tick,))
    w.e.push(w.program_1d, program_delta, t=(outer_tick,))
    if push_seed:
        w.e.push(
            w.seed_1d,
            ZSet({(0, dlg._rewrite_monoid.identity()): 1}),
        )

    # ``min_inner``: deepest inner k touched at any prior outer. We
    # cannot declare convergence below that depth at outer > 0 because
    # the cross-outer integrals propagate up through inner ticks ≥
    # that depth. Same trick the 2-D driver uses.
    min_inner = -1
    fr = w.e.frontiers()[w.state_facts]
    for elt in fr.elements:
        o, s, k_state = elt
        if o < outer_tick and k_state > min_inner:
            min_inner = k_state

    cumulative = w.fact_group.identity()
    for k in range(max_k):
        diff_facts, diff_rewrites = w.e.read(w.body_out, (outer_tick, 0, k))
        empty = not diff_facts.inner and not diff_rewrites.inner
        if empty and k > min_inner:
            break
        cumulative = w.fact_group.add(cumulative, diff_facts)
        w.e.push(w.state_facts, diff_facts, t=(outer_tick, 0, k))
        w.e.push(
            w.state_rewrites,
            diff_rewrites,
            t=(outer_tick, 0, k),
        )
    return cumulative


def _run_3d_batch(
    program: ZSet[dlg.Rule],
    edb: ZSet[dlg.Fact],
) -> ZSet[dlg.Fact]:
    """Push the whole ``edb`` and ``program`` at outer 0; saturate
    once; return the cumulative derived facts."""
    w = _build_3d()
    return _saturate_3d_at_outer(
        w,
        edb,
        program,
        outer_tick=0,
        push_seed=True,
    )


def _run_3d_singletons(
    program: ZSet[dlg.Rule],
    edb: ZSet[dlg.Fact],
) -> ZSet[dlg.Fact]:
    """Push one ``(fact, weight)`` per outer tick. Program is pushed
    once at outer 0. Saturate at each outer tick; accumulate the
    per-outer outer-deltas."""
    fact_list = list(edb.inner.items())
    if not fact_list:
        return _run_3d_batch(program, edb)

    w = _build_3d()
    total = w.fact_group.identity()
    for t0, (fact, weight) in enumerate(fact_list):
        outer_delta = _saturate_3d_at_outer(
            w,
            ZSet({fact: weight}),
            program if t0 == 0 else ZSet({}),
            outer_tick=t0,
            push_seed=(t0 == 0),
        )
        total = w.fact_group.add(total, outer_delta)
    return total


def _assert_batch_eq_singletons(
    program: ZSet[dlg.Rule],
    edb: ZSet[dlg.Fact],
) -> None:
    batched = _run_3d_batch(program, edb)
    singletons = _run_3d_singletons(program, edb)
    assert batched == singletons, (
        f"batch vs singletons differ:\n  batch:      {batched.inner}\n  singletons: {singletons.inner}"
    )


# ---- Positive Datalog cases (mirroring test_integration.py) --------------


def test_3d_positive_batch_eq_singletons_empty_program() -> None:
    edb = ZSet({("E", (0,)): 1, ("E", (1,)): 1})
    _assert_batch_eq_singletons(ZSet({}), edb)


def test_3d_positive_batch_eq_singletons_single_rule() -> None:
    program = ZSet({(("A", ("?X",)), ("E", ("?X",))): 1})
    edb = ZSet({("E", (0,)): 1, ("E", (1,)): 1})
    _assert_batch_eq_singletons(program, edb)


def test_3d_positive_batch_eq_singletons_unification() -> None:
    program = ZSet(
        {
            (("P", ("?X", "?Y")), ("E", ("?X", "?Y")), ("N", ("?Y",))): 1,
        }
    )
    edb = ZSet(
        {
            ("E", (0, 1)): 1,
            ("E", (2, 3)): 1,
            ("N", (1,)): 1,
        }
    )
    _assert_batch_eq_singletons(program, edb)


def test_3d_positive_batch_eq_singletons_transitive_closure() -> None:
    program = ZSet(
        {
            (("T", ("?X", "?Y")), ("E", ("?X", "?Y"))): 1,
            (
                ("T", ("?X", "?Z")),
                ("E", ("?X", "?Y")),
                ("T", ("?Y", "?Z")),
            ): 1,
        }
    )
    edb = ZSet(
        {
            ("E", (0, 1)): 1,
            ("E", (1, 2)): 1,
            ("E", (2, 3)): 1,
        }
    )
    _assert_batch_eq_singletons(program, edb)


def test_3d_positive_batch_eq_singletons_two_rules_union() -> None:
    program = ZSet(
        {
            (("Good", ("?X",)), ("B", ("?X",))): 1,
            (("Good", ("?X",)), ("C", ("?X",))): 1,
        }
    )
    edb = ZSet({("B", (0,)): 1, ("C", (1,)): 1})
    _assert_batch_eq_singletons(program, edb)


def test_3d_positive_batch_eq_singletons_constant_filter() -> None:
    program = ZSet(
        {
            (("OnlyTwo", ("?X",)), ("E", ("?X", 2))): 1,
        }
    )
    edb = ZSet(
        {
            ("E", (0, 2)): 1,
            ("E", (1, 3)): 1,
            ("E", (5, 2)): 1,
        }
    )
    _assert_batch_eq_singletons(program, edb)


# ---- Negation cases (no cycles through negation) -------------------------


def test_3d_negation_batch_eq_singletons_anti_join_no_blocker() -> None:
    program = ZSet(
        {
            (("Live", ("?X",)), ("Node", ("?X",)), ("!Dead", ("?X",))): 1,
        }
    )
    edb = ZSet(
        {
            ("Node", ("alice",)): 1,
            ("Node", ("bob",)): 1,
        }
    )
    _assert_batch_eq_singletons(program, edb)


def test_3d_negation_batch_eq_singletons_anti_join_with_blocker() -> None:
    program = ZSet(
        {
            (("Live", ("?X",)), ("Node", ("?X",)), ("!Dead", ("?X",))): 1,
        }
    )
    edb = ZSet(
        {
            ("Node", ("alice",)): 1,
            ("Node", ("bob",)): 1,
            ("Dead", ("bob",)): 1,
        }
    )
    _assert_batch_eq_singletons(program, edb)


def test_3d_negation_batch_eq_singletons_tc_plus_filter() -> None:
    """Positive recursion (TC) followed by a non-recursive anti-join.
    The body must reach the same answer regardless of whether the
    EDB arrives all at once or one row at a time. Stresses the
    cross-outer bilinear cancellation that the negation algebra
    introduces."""
    program = ZSet(
        {
            (("T", ("?X", "?Y")), ("E", ("?X", "?Y"))): 1,
            (
                ("T", ("?X", "?Z")),
                ("E", ("?X", "?Y")),
                ("T", ("?Y", "?Z")),
            ): 1,
            (
                ("Reachable", ("?X",)),
                ("T", ("?X", "?Y")),
                ("!Banned", ("?Y",)),
            ): 1,
        }
    )
    edb = ZSet(
        {
            ("E", (0, 1)): 1,
            ("E", (1, 2)): 1,
            ("E", (2, 3)): 1,
            ("Banned", (3,)): 1,
        }
    )
    _assert_batch_eq_singletons(program, edb)


def test_3d_negation_batch_eq_singletons_two_layer_anti_join() -> None:
    program = ZSet(
        {
            (
                ("Mid", ("?X",)),
                ("Bot", ("?X",)),
                ("!Excluded", ("?X",)),
            ): 1,
            (
                ("Top", ("?X",)),
                ("Mid", ("?X",)),
                ("!Blocked", ("?X",)),
            ): 1,
        }
    )
    edb = ZSet(
        {
            ("Bot", (0,)): 1,
            ("Bot", (1,)): 1,
            ("Bot", (2,)): 1,
            ("Excluded", (2,)): 1,
            ("Blocked", (1,)): 1,
        }
    )
    _assert_batch_eq_singletons(program, edb)
