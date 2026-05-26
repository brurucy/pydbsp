"""End-to-end integration tests over the full algebra stack.

Reach (transitive closure on edges) and positive Datalog. Both run
on a 2-D lattice: axis 0 = outer (batch progression), axis 1 = inner
(fixpoint iteration). State Inputs are 2-D directly — callers push
state at explicit ``(outer, inner)`` coordinates. The fixpoint loop
is driven via ``e.saturate_inner(body, outer_tick, is_empty=…)`` and
final observations are read via ``e.latest(node)``.

The state-feedback cycle is broken at the Circuit level — state is an
Input, not a derived node. Each iteration's body diff is pushed back
as the next state in test code.
"""

from __future__ import annotations

from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.evaluate import Evaluator
from pydbsp.operator import Input, LiftIntegrate, LiftStreamIntroduction
from pydbsp.progress import NodeId, Time
from pydbsp.indexed_relational_operators import (
    IndexedIncrementalDatalogBody,
    IndexedIncrementalDatalogWithNegationBody,
    IndexedIncrementalReachabilityBody,
)
from pydbsp.storage import DictStorage
from pydbsp import datalog as dlg
from pydbsp.core import Antichain, dbsp_time
from pydbsp.zset import ZSet, ZSetAddition

Edge = tuple[int, int]


def _build_reachability_circuit() -> tuple[
    Evaluator[Time],
    NodeId,  # edges_1d
    NodeId,  # state_2d
    NodeId,  # body_out
    NodeId,  # running_sum
    ZSetAddition[Edge],
]:
    """Wires up the IncrementalReachability circuit. State is a 2-D
    Input; callers push it at explicit ``(outer, inner)`` coords."""
    eg = ZSetAddition[Edge]()
    e = Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(2)),
        group=eg,
    )

    edges_1d = Input[ZSet[Edge]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    state_2d = Input[ZSet[Edge]](frontier=Antichain(dbsp_time(2))).connect(e.circuit, ())

    body_out = IndexedIncrementalReachabilityBody[int](edge_group=eg).connect(e.circuit, (edges_1d, state_2d))
    running_sum = LiftIntegrate[ZSet[Edge]](group=eg).connect(e.circuit, (body_out,))

    return e, edges_1d, state_2d, body_out, running_sum, eg


def _saturate(
    e: Evaluator[Time],
    edges_1d: NodeId,
    state_2d: NodeId,
    body_out: NodeId,
    all_edges: ZSet[Edge],
    eg: ZSetAddition[Edge],
    outer_tick: int = 0,
) -> tuple[ZSet[Edge], int]:
    """Push ``all_edges`` once at outer=0, then drive the inner fixpoint
    at ``outer_tick`` via ``e.saturate_inner``. Returns
    ``(outer_delta, k_final)``."""
    e.push(edges_1d, all_edges)
    cumulative = eg.identity()
    k_final = 0
    for k, diff in e.saturate_inner(
        body_out,
        outer_tick,
        is_empty=lambda d: not d.inner,
    ):
        cumulative = eg.add(cumulative, diff)
        e.push(state_2d, diff, t=(outer_tick, k))
        k_final = k + 1
    return cumulative, k_final


# ============================================================================
# Tests
# ============================================================================


def test_triangle_3_cycle_reaches_all_pairs() -> None:
    """``0 → 1 → 2 → 0`` — every node reaches every node (including
    itself via the 3-hop cycle). Expected: 9 reachable pairs.
    Converges in 3 inner ticks (direct, 1-hop, 2-hop, then empty)."""
    e, edges_1d, state_2d, body_out, _, eg = _build_reachability_circuit()
    all_edges = ZSet({(0, 1): 1, (1, 2): 1, (2, 0): 1})

    reach, k_final = _saturate(e, edges_1d, state_2d, body_out, all_edges, eg)

    expected = ZSet({(u, v): 1 for u in range(3) for v in range(3)})
    assert reach == expected
    assert k_final == 3


def test_linear_chain_reaches_diameter_pairs() -> None:
    """``0 → 1 → 2 → 3 → 4`` — every ``(i, j)`` with ``i < j`` is
    reachable. Expected: 10 pairs. Diameter is 4 hops; converges in
    ``diameter + 1`` inner ticks."""
    e, edges_1d, state_2d, body_out, _, eg = _build_reachability_circuit()
    all_edges = ZSet({(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 4): 1})

    reach, k_final = _saturate(e, edges_1d, state_2d, body_out, all_edges, eg)

    expected = ZSet({(i, j): 1 for i in range(5) for j in range(i + 1, 5)})
    assert reach == expected
    # Diameter 4 hops; one initial tick + 3 more before empty.
    assert k_final == 4


def test_empty_graph_reaches_nothing() -> None:
    """No edges → no reachable pairs. Converges immediately at
    ``k = 0`` because ``body_out(0, 0)`` is empty."""
    e, edges_1d, state_2d, body_out, _, eg = _build_reachability_circuit()
    all_edges: ZSet[Edge] = ZSet({})

    reach, k_final = _saturate(e, edges_1d, state_2d, body_out, all_edges, eg)

    assert reach == ZSet({})
    assert k_final == 0


def test_running_sum_at_latest_equals_cumulative() -> None:
    """Sanity check: the cumulative-state Z-set built up in the test
    loop must equal ``e.latest(running_sum)`` — the running sum at
    the freshest known frontier element."""
    e, edges_1d, state_2d, body_out, running_sum, eg = _build_reachability_circuit()
    all_edges = ZSet({(0, 1): 1, (1, 2): 1, (2, 0): 1})

    reach, _ = _saturate(e, edges_1d, state_2d, body_out, all_edges, eg)

    assert e.latest(running_sum) == reach


# ============================================================================
# IncrementalDatalog — positive Datalog interpreter
# ============================================================================


def _build_datalog_circuit() -> tuple[
    Evaluator[Time],
    NodeId,  # edb_1d
    NodeId,  # program_1d
    NodeId,  # state_facts_2d
    NodeId,  # state_rewrites_2d
    NodeId,  # seed_1d
    NodeId,  # body_out (pair value)
    ZSetAddition[dlg.Fact],
]:
    """Wire the positive Datalog body on a 2-D lattice. Three 1-D
    Inputs (edb, program, seed) + two 2-D state Inputs feed into one
    body operator that mirrors
    :func:`pydbsp.algorithms.datalog.IncrementalDatalog`."""
    fact_group = ZSetAddition[dlg.Fact]()
    # The Evaluator's global group is used as the default for missing
    # reads. For Datalog we touch multiple V-types (Fact, Rule,
    # ProvenanceIndexedRewrite), but we always push before reading, so
    # the default is never actually returned. Pick any one ZSet group.
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

    program_2d = LiftStreamIntroduction[ZSet[dlg.Rule]](group=ZSetAddition[dlg.Rule]()).connect(
        e.circuit, (program_1d,)
    )

    body_out = IndexedIncrementalDatalogBody(
        fact_group=fact_group,
        rule_group=ZSetAddition[dlg.Rule](),
        rewrite_group=ZSetAddition[dlg.ProvenanceIndexedRewrite](),
        signal_group=ZSetAddition[dlg.Signal](),
        ext_dir_group=ZSetAddition[dlg.ExtendedDirection](),
        jorder_group=ZSetAddition[tuple[str, dlg.ColumnReference]](),
        gatekeep_group=ZSetAddition[dlg.IndexedGatekeepEntry](),
        indexed_fact_group=ZSetAddition[dlg.IndexedFact](),
    ).connect(
        e.circuit,
        (edb_1d, program_2d, state_facts_2d, state_rewrites_2d, seed_1d),
    )

    return (
        e,
        edb_1d,
        program_1d,
        state_facts_2d,
        state_rewrites_2d,
        seed_1d,
        body_out,
        fact_group,
    )


def _saturate_datalog(
    e: Evaluator[Time],
    edb_1d: NodeId,
    program_1d: NodeId,
    state_facts_2d: NodeId,
    state_rewrites_2d: NodeId,
    seed_1d: NodeId,
    body_out: NodeId,
    edb_facts: ZSet[dlg.Fact],
    program: ZSet[dlg.Rule],
    fact_group: ZSetAddition[dlg.Fact],
    outer_tick: int = 0,
) -> ZSet[dlg.Fact]:
    """Push initial data + seed, drive the inner fixpoint at
    ``outer_tick`` via ``e.saturate_inner``. Returns the outer-delta
    — the cumulative derived facts contributed by this outer tick
    (includes the EDB pushed at this tick since ``next_facts`` is
    ``DLDDistinct(ground + edb)``).

    ``min_inner`` is derived from the state-facts frontier: at
    ``outer_tick > 0``, we won't converge until ``k`` exceeds the
    deepest inner tick of any prior outer's state."""
    e.push(edb_1d, edb_facts)
    e.push(program_1d, program)
    if outer_tick == 0:
        e.push(seed_1d, ZSet({(0, dlg._rewrite_monoid.identity()): 1}))

    # Prior outers' deepest inner state-push — bounds early convergence.
    min_inner = -1
    fr = e.frontiers()[state_facts_2d]
    for o, k_state in fr.elements:
        if o < outer_tick and k_state > min_inner:
            min_inner = k_state

    cumulative = fact_group.identity()
    for k, (diff_facts, diff_rewrites) in e.saturate_inner(
        body_out,
        outer_tick,
        is_empty=lambda p: not p[0].inner and not p[1].inner,
        min_inner=min_inner,
    ):
        cumulative = fact_group.add(cumulative, diff_facts)
        e.push(state_facts_2d, diff_facts, t=(outer_tick, k))
        e.push(state_rewrites_2d, diff_rewrites, t=(outer_tick, k))
    return cumulative


def test_datalog_triangle_transitive_closure() -> None:
    """Encode transitive closure as a 2-rule Datalog program and run it
    over a 3-cycle. Expected derived facts: the 3 input edges plus the
    9 ``reach(u, v)`` pairs (every node reaches every node)."""
    e, edb_1d, program_1d, state_facts_2d, state_rewrites_2d, seed_1d, body_out, fg = _build_datalog_circuit()

    rule_base = (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y")))
    rule_step = (
        ("reach", ("?X", "?Z")),
        ("edge", ("?X", "?Y")),
        ("reach", ("?Y", "?Z")),
    )
    program = ZSet({rule_base: 1, rule_step: 1})

    edb = ZSet(
        {
            ("edge", (0, 1)): 1,
            ("edge", (1, 2)): 1,
            ("edge", (2, 0)): 1,
        }
    )

    derived = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        state_facts_2d,
        state_rewrites_2d,
        seed_1d,
        body_out,
        edb,
        program,
        fg,
    )

    expected_edges = {("edge", e): 1 for e in [(0, 1), (1, 2), (2, 0)]}
    expected_reach = {("reach", (u, v)): 1 for u in range(3) for v in range(3)}
    assert derived == ZSet({**expected_edges, **expected_reach})


def test_datalog_linear_chain_transitive_closure() -> None:
    """``0 → 1 → 2 → 3``: 3 edges, 6 reach pairs (every ``(i, j)``
    with ``i < j``)."""
    e, edb_1d, program_1d, state_facts_2d, state_rewrites_2d, seed_1d, body_out, fg = _build_datalog_circuit()

    program = ZSet(
        {
            (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
            (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
        }
    )
    edb = ZSet(
        {
            ("edge", (0, 1)): 1,
            ("edge", (1, 2)): 1,
            ("edge", (2, 3)): 1,
        }
    )

    derived = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        state_facts_2d,
        state_rewrites_2d,
        seed_1d,
        body_out,
        edb,
        program,
        fg,
    )

    expected_edges = {("edge", e): 1 for e in [(0, 1), (1, 2), (2, 3)]}
    expected_reach = {("reach", (i, j)): 1 for i in range(4) for j in range(i + 1, 4)}
    assert derived == ZSet({**expected_edges, **expected_reach})


def test_datalog_empty_program_returns_edb() -> None:
    """No rules → derived facts equal the EDB unchanged."""
    e, edb_1d, program_1d, sf, sr, seed, body, fg = _build_datalog_circuit()
    edb = ZSet({("E", (0,)): 1, ("E", (1,)): 1})
    derived = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        edb,
        ZSet({}),
        fg,
    )
    assert derived == edb


def test_datalog_empty_edb_returns_empty() -> None:
    """No facts → no derivations possible regardless of rules."""
    e, edb_1d, program_1d, sf, sr, seed, body, fg = _build_datalog_circuit()
    program = ZSet(
        {
            (("A", ("?X",)), ("E", ("?X",))): 1,
        }
    )
    derived = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        ZSet({}),
        program,
        fg,
    )
    assert derived == ZSet({})


def test_datalog_single_rule_projects_edb_into_head() -> None:
    """``A(X) :- E(X)`` — every E fact yields a matching A fact."""
    e, edb_1d, program_1d, sf, sr, seed, body, fg = _build_datalog_circuit()
    program = ZSet(
        {
            (("A", ("?X",)), ("E", ("?X",))): 1,
        }
    )
    edb = ZSet({("E", (0,)): 1, ("E", (1,)): 1})
    derived = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        edb,
        program,
        fg,
    )
    assert derived == ZSet(
        {
            ("E", (0,)): 1,
            ("E", (1,)): 1,
            ("A", (0,)): 1,
            ("A", (1,)): 1,
        }
    )


def test_datalog_unification_binds_variables_across_body() -> None:
    """``P(X, Y) :- E(X, Y), N(Y)`` — only E pairs whose ``Y`` also
    appears in ``N`` make it into ``P``."""
    e, edb_1d, program_1d, sf, sr, seed, body, fg = _build_datalog_circuit()

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
    derived = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        edb,
        program,
        fg,
    )
    # P(0, 1) is derivable (N(1) matches); P(2, 3) is not (no N(3)).
    assert derived.inner.get(("P", (0, 1)), 0) == 1
    assert derived.inner.get(("P", (2, 3)), 0) == 0


def test_datalog_two_rules_into_same_head_are_unioned() -> None:
    """``Good(X) :- B(X)`` and ``Good(X) :- C(X)`` — derived facts are
    the union of what each rule produces."""
    e, edb_1d, program_1d, sf, sr, seed, body, fg = _build_datalog_circuit()

    program = ZSet(
        {
            (("Good", ("?X",)), ("B", ("?X",))): 1,
            (("Good", ("?X",)), ("C", ("?X",))): 1,
        }
    )
    edb = ZSet({("B", (0,)): 1, ("C", (1,)): 1})
    derived = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        edb,
        program,
        fg,
    )
    assert derived.inner.get(("Good", (0,)), 0) == 1
    assert derived.inner.get(("Good", (1,)), 0) == 1


def test_datalog_constant_term_in_body_filters() -> None:
    """``OnlyTwo(X) :- E(X, 2)`` — a constant in the body filters EDB
    facts to those whose second column is exactly 2."""
    e, edb_1d, program_1d, sf, sr, seed, body, fg = _build_datalog_circuit()

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
    derived = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        edb,
        program,
        fg,
    )
    assert derived.inner.get(("OnlyTwo", (0,)), 0) == 1
    assert derived.inner.get(("OnlyTwo", (5,)), 0) == 1
    assert derived.inner.get(("OnlyTwo", (1,)), 0) == 0


# ============================================================================
# Batch-vs-singletons equivalence (multi-outer-tick incremental Datalog)
# ============================================================================


def _run_datalog_batch(program: ZSet[dlg.Rule], edb: ZSet[dlg.Fact]) -> ZSet[dlg.Fact]:
    """Push the entire ``edb`` and ``program`` at outer=0, saturate
    once, return the cumulative derived facts."""
    e, edb_1d, program_1d, sf, sr, seed, body, fg = _build_datalog_circuit()
    return _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        edb,
        program,
        fg,
    )


def _run_datalog_singletons(program: ZSet[dlg.Rule], edb: ZSet[dlg.Fact]) -> ZSet[dlg.Fact]:
    """Push one ``(fact, weight)`` per outer tick. ``program`` at
    ``t0 = 0`` only, empty thereafter. Saturate at each tick;
    accumulate the per-outer outer-deltas. Returns the cumulative
    derived facts (which should match :func:`_run_datalog_batch`)."""
    fact_list = list(edb.inner.items())
    if not fact_list:
        return _run_datalog_batch(program, edb)

    e, edb_1d, program_1d, sf, sr, seed, body, fg = _build_datalog_circuit()
    total = fg.identity()
    for t0, (fact, weight) in enumerate(fact_list):
        outer_delta = _saturate_datalog(
            e,
            edb_1d,
            program_1d,
            sf,
            sr,
            seed,
            body,
            ZSet({fact: weight}),
            program if t0 == 0 else ZSet({}),
            fg,
            outer_tick=t0,
        )
        total = fg.add(total, outer_delta)
    return total


def _assert_datalog_batch_eq_singletons(program: ZSet[dlg.Rule], edb: ZSet[dlg.Fact]) -> None:
    batched = _run_datalog_batch(program, edb)
    singletons = _run_datalog_singletons(program, edb)
    assert batched == singletons, (
        f"batch vs singletons differ:\n  batch: {batched.inner}\n  singletons: {singletons.inner}"
    )


def test_datalog_batch_eq_singletons_empty_program() -> None:
    edb = ZSet({("E", (0,)): 1, ("E", (1,)): 1})
    _assert_datalog_batch_eq_singletons(ZSet({}), edb)


def test_datalog_batch_eq_singletons_single_rule() -> None:
    program = ZSet(
        {
            (("A", ("?X",)), ("E", ("?X",))): 1,
        }
    )
    edb = ZSet({("E", (0,)): 1, ("E", (1,)): 1})
    _assert_datalog_batch_eq_singletons(program, edb)


def test_datalog_batch_eq_singletons_unification() -> None:

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
    _assert_datalog_batch_eq_singletons(program, edb)


def test_datalog_batch_eq_singletons_transitive_closure() -> None:

    program = ZSet(
        {
            (("T", ("?X", "?Y")), ("E", ("?X", "?Y"))): 1,
            (("T", ("?X", "?Z")), ("E", ("?X", "?Y")), ("T", ("?Y", "?Z"))): 1,
        }
    )
    edb = ZSet(
        {
            ("E", (0, 1)): 1,
            ("E", (1, 2)): 1,
            ("E", (2, 3)): 1,
        }
    )
    _assert_datalog_batch_eq_singletons(program, edb)


def test_datalog_batch_eq_singletons_two_rules_union() -> None:

    program = ZSet(
        {
            (("Good", ("?X",)), ("B", ("?X",))): 1,
            (("Good", ("?X",)), ("C", ("?X",))): 1,
        }
    )
    edb = ZSet({("B", (0,)): 1, ("C", (1,)): 1})
    _assert_datalog_batch_eq_singletons(program, edb)


def test_datalog_batch_eq_singletons_constant_filter() -> None:

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
    _assert_datalog_batch_eq_singletons(program, edb)


# ============================================================================
# IncrementalDatalogWithNegation — stratified-negation Datalog
# ============================================================================


def _build_datalog_negation_circuit() -> tuple[
    Evaluator[Time],
    NodeId,  # edb_1d
    NodeId,  # program_1d
    NodeId,  # state_facts_2d
    NodeId,  # state_rewrites_2d
    NodeId,  # seed_1d
    NodeId,  # body_out (pair)
    ZSetAddition[dlg.Fact],
]:
    """Same wiring as :func:`_build_datalog_circuit` but with the
    negation-aware body."""
    fact_group = ZSetAddition[dlg.Fact]()
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

    program_2d = LiftStreamIntroduction[ZSet[dlg.Rule]](group=ZSetAddition[dlg.Rule]()).connect(
        e.circuit, (program_1d,)
    )

    body_out = IndexedIncrementalDatalogWithNegationBody(
        fact_group=fact_group,
        rule_group=ZSetAddition[dlg.Rule](),
        rewrite_group=ZSetAddition[dlg.ProvenanceIndexedRewrite](),
        signal_group=ZSetAddition[dlg.Signal](),
        ext_dir_group=ZSetAddition[dlg.ExtendedDirection](),
        jorder_group=ZSetAddition[tuple[str, dlg.ColumnReference]](),
        gatekeep_group=ZSetAddition[dlg.IndexedGatekeepEntry](),
        indexed_fact_group=ZSetAddition[dlg.IndexedFact](),
    ).connect(
        e.circuit,
        (edb_1d, program_2d, state_facts_2d, state_rewrites_2d, seed_1d),
    )

    return (
        e,
        edb_1d,
        program_1d,
        state_facts_2d,
        state_rewrites_2d,
        seed_1d,
        body_out,
        fact_group,
    )


def _run_datalog_negation(program: ZSet[dlg.Rule], edb: ZSet[dlg.Fact]) -> ZSet[dlg.Fact]:
    """Single-batch run of the negation-aware Datalog body. Pushes
    everything at outer=0, saturates, returns cumulative derived
    facts."""
    e, edb_1d, program_1d, sf, sr, seed, body, fg = _build_datalog_negation_circuit()
    return _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        edb,
        program,
        fg,
    )


def test_datalog_negation_no_blocker_passes() -> None:
    """``A(X) :- E(X), !B(X)`` with no ``B`` facts — every ``E`` makes
    it to ``A``."""

    program = ZSet(
        {
            (("A", ("?X",)), ("E", ("?X",)), ("!B", ("?X",))): 1,
        }
    )
    edb = ZSet({("E", (0,)): 1, ("E", (1,)): 1})
    derived = _run_datalog_negation(program, edb)
    assert derived.inner.get(("A", (0,)), 0) == 1
    assert derived.inner.get(("A", (1,)), 0) == 1


def test_datalog_negation_blocker_filters() -> None:
    """``A(X) :- E(X), !B(X)`` with ``B(0)`` present — ``A(0)`` is
    blocked, ``A(1)`` still derived."""

    program = ZSet(
        {
            (("A", ("?X",)), ("E", ("?X",)), ("!B", ("?X",))): 1,
        }
    )
    edb = ZSet(
        {
            ("E", (0,)): 1,
            ("E", (1,)): 1,
            ("B", (0,)): 1,
        }
    )
    derived = _run_datalog_negation(program, edb)
    assert derived.inner.get(("A", (0,)), 0) == 0
    assert derived.inner.get(("A", (1,)), 0) == 1


def test_datalog_negation_dual_support_with_blocker() -> None:
    """``A`` derived via two positive rules, then blocked: ``Good(X)
    :- A(X), !Bad(X)``. With ``Bad(0)`` present, ``Good(0)`` is
    blocked even though both supports for ``A(0)`` fire."""

    program = ZSet(
        {
            (("A", ("?X",)), ("B", ("?X",))): 1,
            (("A", ("?X",)), ("C", ("?X",))): 1,
            (("Good", ("?X",)), ("A", ("?X",)), ("!Bad", ("?X",))): 1,
        }
    )
    edb = ZSet(
        {
            ("B", (0,)): 1,
            ("C", (0,)): 1,
            ("Bad", (0,)): 1,
        }
    )
    derived = _run_datalog_negation(program, edb)
    assert derived.inner.get(("A", (0,)), 0) == 1
    assert derived.inner.get(("Good", (0,)), 0) == 0


def test_datalog_negation_recursive_with_blocker() -> None:
    """Recursive ``T`` (transitive closure) plus a negation-based
    ``Good`` predicate. With no ``Block``, ``Good(0)`` holds (no
    upstream ``T(0, _)`` matches ``Block``)."""

    program = ZSet(
        {
            (("T", ("?X", "?Y")), ("E", ("?X", "?Y"))): 1,
            (("T", ("?X", "?Z")), ("E", ("?X", "?Y")), ("T", ("?Y", "?Z"))): 1,
            (("Bad", ("?X",)), ("T", ("?X", "?Y")), ("Block", ("?Y",))): 1,
            (("Good", ("?X",)), ("Node", ("?X",)), ("!Bad", ("?X",))): 1,
        }
    )
    edb = ZSet(
        {
            ("Node", (0,)): 1,
            ("Node", (1,)): 1,
            ("E", (0, 1)): 1,
        }
    )
    derived = _run_datalog_negation(program, edb)
    # T(0, 1) is derived; no Block facts → Bad is empty → Good fires for all Node.
    assert derived.inner.get(("T", (0, 1)), 0) == 1
    assert derived.inner.get(("Good", (0,)), 0) == 1
    assert derived.inner.get(("Good", (1,)), 0) == 1
    assert derived.inner.get(("Bad", (0,)), 0) == 0


# ============================================================================
# Batch-vs-singletons equivalence for the negation body
# ============================================================================


def _run_datalog_negation_batch(program: ZSet[dlg.Rule], edb: ZSet[dlg.Fact]) -> ZSet[dlg.Fact]:
    return _run_datalog_negation(program, edb)


def _run_datalog_negation_singletons(program: ZSet[dlg.Rule], edb: ZSet[dlg.Fact]) -> ZSet[dlg.Fact]:
    """Mirror of :func:`_run_datalog_singletons` over the negation body.
    Pushes one ``(fact, weight)`` per outer tick into a single evaluator
    and accumulates the per-outer outer-deltas."""
    fact_list = list(edb.inner.items())
    if not fact_list:
        return _run_datalog_negation_batch(program, edb)

    e, edb_1d, program_1d, sf, sr, seed, body, fg = _build_datalog_negation_circuit()
    total = fg.identity()
    for t0, (fact, weight) in enumerate(fact_list):
        outer_delta = _saturate_datalog(
            e,
            edb_1d,
            program_1d,
            sf,
            sr,
            seed,
            body,
            ZSet({fact: weight}),
            program if t0 == 0 else ZSet({}),
            fg,
            outer_tick=t0,
        )
        total = fg.add(total, outer_delta)
    return total


def _assert_datalog_negation_batch_eq_singletons(program: ZSet[dlg.Rule], edb: ZSet[dlg.Fact]) -> None:
    batched = _run_datalog_negation_batch(program, edb)
    singletons = _run_datalog_negation_singletons(program, edb)
    assert batched == singletons, (
        f"negation batch vs singletons differ:\n  batch: {batched.inner}\n  singletons: {singletons.inner}"
    )


def test_datalog_negation_batch_eq_singletons_no_blocker() -> None:
    program = ZSet(
        {
            (("A", ("?X",)), ("E", ("?X",)), ("!B", ("?X",))): 1,
        }
    )
    edb = ZSet({("E", (0,)): 1, ("E", (1,)): 1})
    _assert_datalog_negation_batch_eq_singletons(program, edb)


def test_datalog_negation_batch_eq_singletons_blocker_filters() -> None:
    program = ZSet(
        {
            (("A", ("?X",)), ("E", ("?X",)), ("!B", ("?X",))): 1,
        }
    )
    edb = ZSet(
        {
            ("E", (0,)): 1,
            ("E", (1,)): 1,
            ("B", (0,)): 1,
        }
    )
    _assert_datalog_negation_batch_eq_singletons(program, edb)


def test_datalog_negation_batch_eq_singletons_dual_support_with_blocker() -> None:
    program = ZSet(
        {
            (("A", ("?X",)), ("B", ("?X",))): 1,
            (("A", ("?X",)), ("C", ("?X",))): 1,
            (("Good", ("?X",)), ("A", ("?X",)), ("!Bad", ("?X",))): 1,
        }
    )
    edb = ZSet(
        {
            ("B", (0,)): 1,
            ("C", (0,)): 1,
            ("Bad", (0,)): 1,
        }
    )
    _assert_datalog_negation_batch_eq_singletons(program, edb)


def test_datalog_negation_batch_eq_singletons_recursive_with_blocker() -> None:
    program = ZSet(
        {
            (("T", ("?X", "?Y")), ("E", ("?X", "?Y"))): 1,
            (
                ("T", ("?X", "?Z")),
                ("E", ("?X", "?Y")),
                ("T", ("?Y", "?Z")),
            ): 1,
            (("Bad", ("?X",)), ("T", ("?X", "?Y")), ("Block", ("?Y",))): 1,
            (("Good", ("?X",)), ("Node", ("?X",)), ("!Bad", ("?X",))): 1,
        }
    )
    edb = ZSet(
        {
            ("Node", (0,)): 1,
            ("Node", (1,)): 1,
            ("E", (0, 1)): 1,
        }
    )
    _assert_datalog_negation_batch_eq_singletons(program, edb)


# ============================================================================
# Fact retraction across outer ticks. The 2-D bodies (positive and
# negation) correctly unsay derived facts when their EDB support is
# retracted at a later outer — `D^o` of cum_distinct(ground + edb)
# emits the expected negative deltas. (The 3-D bodies and the
# stratified driver do *not* — see tests/test_3d_datalog_streaming.py
# and tests/test_datalog_stratified_streaming.py for those documented
# bugs.)
# ============================================================================


def test_datalog_positive_body_fact_retraction() -> None:
    """At outer 0, derive b(1) from p(1) via the rule b :- p. At outer
    1, retract p(1). The streamed cumulative should equal what a fresh
    evaluator on the empty EDB would produce — i.e., nothing."""
    program = ZSet({(("b", ("?X",)), ("p", ("?X",))): 1})
    edb_0 = ZSet({("p", (1,)): 1})
    edb_1 = ZSet({("p", (1,)): -1})

    e, edb_1d, program_1d, sf, sr, seed, body, fg = _build_datalog_circuit()
    delta_0 = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        edb_0,
        program,
        fg,
        outer_tick=0,
    )
    delta_1 = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        edb_1,
        ZSet({}),
        fg,
        outer_tick=1,
    )
    streamed = fg.add(delta_0, delta_1)
    expected = fg.identity()  # post-retraction cumulative EDB is empty

    assert streamed == expected, (
        f"2-D positive body fact retraction:\n  streamed: {dict(streamed.inner)}\n  expected: {dict(expected.inner)}"
    )


def test_datalog_negation_body_fact_retraction() -> None:
    """``alive(?X) :- person(?X), !dead(?X)`` with ``person(alice)`` at
    outer 0 — derives alive(alice). At outer 1, retract person(alice).
    The streamed cumulative should match a fresh evaluator on the empty
    EDB (nothing derived)."""
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

    e, edb_1d, program_1d, sf, sr, seed, body, fg = _build_datalog_negation_circuit()
    delta_0 = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        edb_0,
        program,
        fg,
        outer_tick=0,
    )
    delta_1 = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        edb_1,
        ZSet({}),
        fg,
        outer_tick=1,
    )
    streamed = fg.add(delta_0, delta_1)
    expected = fg.identity()

    assert streamed == expected, (
        f"2-D negation body fact retraction:\n  streamed: {dict(streamed.inner)}\n  expected: {dict(expected.inner)}"
    )


# ============================================================================
# Rule retraction across outer ticks. Push a rule and EDB at outer 0
# (deriving some facts), then push the rule with weight −1 at outer 1.
# The cumulative matches a fresh evaluator on the empty post-retraction
# program (just the EDB). This used to be broken — `dlg.sig` and
# `dlg.dir` filtered out negative-weight rules with `if weight <= 0`,
# silently dropping retractions at the value layer. Fixed by changing
# the gate to `if weight == 0` so signed weights propagate.
# ============================================================================


def test_datalog_positive_body_rule_retraction() -> None:
    """Outer 0: rule b :- p + EDB p(1) → derives b(1). Outer 1: push
    the rule with weight −1. Streamed cumulative should match a fresh
    evaluator on the empty program with the same EDB — i.e., just
    p(1), no b(1)."""
    rule = (("b", ("?X",)), ("p", ("?X",)))
    edb = ZSet({("p", (1,)): 1})

    e, edb_1d, program_1d, sf, sr, seed, body, fg = _build_datalog_circuit()
    delta_0 = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        edb,
        ZSet({rule: 1}),
        fg,
        outer_tick=0,
    )
    delta_1 = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        ZSet({}),
        ZSet({rule: -1}),
        fg,
        outer_tick=1,
    )
    streamed = fg.add(delta_0, delta_1)
    expected = edb

    assert streamed == expected, (
        f"2-D positive body rule retraction:\n  streamed: {dict(streamed.inner)}\n  expected: {dict(expected.inner)}"
    )


def test_datalog_negation_body_rule_retraction() -> None:
    """``alive(?X) :- person(?X), !dead(?X)`` derives alive(alice) at
    outer 0. Retract the rule at outer 1. Streamed should match fresh
    on empty program — just the EDB."""
    rule = (
        ("alive", ("?X",)),
        ("person", ("?X",)),
        ("!dead", ("?X",)),
    )
    edb = ZSet({("person", ("alice",)): 1})

    e, edb_1d, program_1d, sf, sr, seed, body, fg = _build_datalog_negation_circuit()
    delta_0 = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        edb,
        ZSet({rule: 1}),
        fg,
        outer_tick=0,
    )
    delta_1 = _saturate_datalog(
        e,
        edb_1d,
        program_1d,
        sf,
        sr,
        seed,
        body,
        ZSet({}),
        ZSet({rule: -1}),
        fg,
        outer_tick=1,
    )
    streamed = fg.add(delta_0, delta_1)
    expected = edb

    assert streamed == expected, (
        f"2-D negation body rule retraction:\n  streamed: {dict(streamed.inner)}\n  expected: {dict(expected.inner)}"
    )
