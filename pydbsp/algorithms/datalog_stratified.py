"""Stratified Datalog on a 3-D lattice ``(outer, stratum, inner)``.

The public evaluator in this module is the dynamic-level variant:

- external fact and rule updates arrive on the outer axis,
- a positive sidecar incrementally maintains predicate levels and
  same-level positive recursion,
- the wrapped 3-D circuit saturates the active level(s) on the inner
  axis.
"""

from dataclasses import dataclass

from typing import cast

from pydbsp.core import DBSPTime, ProductGroup, Time1D, Time2D
from pydbsp.evaluator import Evaluator
from pydbsp.indexed_zset.operators.bilinear import (
    DLDSortMergeJoin3D,
    DLDSortMergeJoin3DStaticInnerRight,
)
from pydbsp.indexed_zset.operators.linear import Index
from pydbsp.stream import Input, Lift1, Lift2, Stream
from pydbsp.stream.functions.linear import (
    StreamIntroduction,
    TimeAxisElimination,
    TimeAxisIntroduction,
)
from pydbsp.stream.operators.linear import Delay, Integrate
from pydbsp.stream.zset.operators.binary import DLDDistinct3D
from pydbsp.stream.zset.operators.linear import Project, Select
from pydbsp.zset import ZSet, ZSetAddition

from pydbsp.algorithms.datalog import (
    EDB,
    DatalogCircuit,
    Fact,
    Program,
    ProvenanceIndexedRewrite,
    Rule,
    Signal,
    Variable,
    saturate,
    _rewrite_monoid,
    sig,
)
from pydbsp.algorithms.datalog_indexed import (
    ColumnReference,
    ExtendedDirection,
    IndexedFact,
    IndexedGatekeepEntry,
    _gatekeep_join_key,
    _index_fact,
    _indexed_product_proj,
    ext_dir,
    IncrementalDatalogWithIndexing,
    jorder,
)


Time3D: DBSPTime[tuple[int, int, int]] = DBSPTime(nestedness=3)

PredicateName = str
PredicateDependency = tuple[PredicateName, PredicateName]
PredicateLevel = tuple[PredicateName, int]
RecursivePredicatePair = tuple[PredicateName, PredicateName]

_LV_P = Variable("P")
_LV_Q = Variable("Q")
_LV_S = Variable("S")
_LV_S1 = Variable("S1")


# ---- Program-analysis sidecar ---------------------------------------------


def head_predicates(program: ZSet[Rule]) -> ZSet[PredicateName]:
    """Unique head predicates present in the current rule delta."""
    return ZSet(
        {
            rule[0][0]: 1
            for rule, weight in program.inner.items()
            if weight > 0
        }
    )


def positive_dependencies(program: ZSet[Rule]) -> ZSet[PredicateDependency]:
    """Positive predicate-dependency edges ``head <- body`` for the
    current rule delta.
    """
    edges: dict[PredicateDependency, int] = {}
    for rule, weight in program.inner.items():
        if weight <= 0:
            continue
        head_pred = rule[0][0]
        for body_atom in rule[1:]:
            body_pred = body_atom[0]
            if body_pred.startswith("!"):
                continue
            edges[(head_pred, body_pred)] = 1
    return ZSet(edges)


def negative_dependencies(program: ZSet[Rule]) -> ZSet[PredicateDependency]:
    """Negative predicate-dependency edges ``head <- !body`` for the
    current rule delta.
    """
    edges: dict[PredicateDependency, int] = {}
    for rule, weight in program.inner.items():
        if weight <= 0:
            continue
        head_pred = rule[0][0]
        for body_atom in rule[1:]:
            body_pred = body_atom[0]
            if not body_pred.startswith("!"):
                continue
            edges[(head_pred, body_pred[1:])] = 1
    return ZSet(edges)


@dataclass(slots=True)
class DynamicLevelSidecar:
    """Incremental dependency-analysis streams for dynamic stratification."""

    head_predicates_3d: Stream[ZSet[PredicateName], tuple[int, int, int]]
    positive_dependencies_3d: Stream[ZSet[PredicateDependency], tuple[int, int, int]]
    negative_dependencies_3d: Stream[ZSet[PredicateDependency], tuple[int, int, int]]
    positive_reach_3d: Stream[ZSet[RecursivePredicatePair], tuple[int, int, int]] | None
    level_at_least_3d: Stream[ZSet[PredicateLevel], tuple[int, int, int]] | None
    same_level_recursive_3d: Stream[ZSet[RecursivePredicatePair], tuple[int, int, int]] | None


def _level_program() -> Program:
    """Positive Datalog program computing ``level_at_least(P,S)`` from
    ``head(P)``, ``pos_dep(P,Q)``, ``neg_dep(P,Q)``, and ``succ(S,S1)``.
    """
    return ZSet(
        {
            (("level", (_LV_P, 0)), ("head", (_LV_P,))): 1,
            (("level", (_LV_P, _LV_S)), ("pos_dep", (_LV_P, _LV_Q)), ("level", (_LV_Q, _LV_S))): 1,
            (("level", (_LV_P, _LV_S1)), ("neg_dep", (_LV_P, _LV_Q)), ("level", (_LV_Q, _LV_S)), ("succ", (_LV_S, _LV_S1))): 1,
        }
    )


def _reach_program() -> Program:
    """Positive reachability over the positive dependency graph."""
    return ZSet(
        {
            (("reach", (_LV_P, _LV_Q)), ("pos_dep", (_LV_P, _LV_Q))): 1,
            (("reach", (_LV_P, _LV_S)), ("pos_dep", (_LV_P, _LV_Q)), ("reach", (_LV_Q, _LV_S))): 1,
        }
    )


def _program_delta_to_level_edb(program_delta: Program) -> EDB:
    """Translate a rule delta into dependency facts for the level
    sidecar.
    """
    facts: dict[Fact, int] = {}
    heads = head_predicates(program_delta)
    pos = positive_dependencies(program_delta)
    neg = negative_dependencies(program_delta)
    for pred, weight in heads.inner.items():
        facts[("head", (pred,))] = facts.get(("head", (pred,)), 0) + weight
    for dep, weight in pos.inner.items():
        facts[("pos_dep", dep)] = facts.get(("pos_dep", dep), 0) + weight
    for dep, weight in neg.inner.items():
        facts[("neg_dep", dep)] = facts.get(("neg_dep", dep), 0) + weight
    return ZSet({k: v for k, v in facts.items() if v != 0})


def _succ_facts(start: int, stop: int) -> EDB:
    """Generate ``succ(i, i+1)`` facts for ``start <= i < stop``."""
    if stop <= start:
        return ZSet({})
    return ZSet({("succ", (i, i + 1)): 1 for i in range(start, stop)})


def _extract_level_facts(facts: EDB) -> ZSet[PredicateLevel]:
    return ZSet(
        {
            cast(PredicateLevel, (args[0], args[1])): weight
            for (pred, args), weight in facts.inner.items()
            if pred == "level"
        }
    )


def _extract_reach_facts(facts: EDB) -> ZSet[RecursivePredicatePair]:
    return ZSet(
        {
            cast(RecursivePredicatePair, (args[0], args[1])): weight
            for (pred, args), weight in facts.inner.items()
            if pred == "reach"
        }
    )


def same_level_recursive_pairs(
    reach_facts: ZSet[RecursivePredicatePair],
    level_facts: ZSet[PredicateLevel],
) -> ZSet[RecursivePredicatePair]:
    """Ordered predicate pairs in the same current level and mutually
    positively reachable.
    """
    reach = {
        pair
        for pair, weight in reach_facts.inner.items()
        if weight > 0
    }
    level_map: dict[PredicateName, int] = {}
    for (pred, level), weight in level_facts.inner.items():
        if weight <= 0:
            continue
        prior = level_map.get(pred)
        if prior is None or level > prior:
            level_map[pred] = level
    pairs: dict[RecursivePredicatePair, int] = {}
    for p, q in reach:
        if (q, p) not in reach:
            continue
        if level_map.get(p) != level_map.get(q):
            continue
        pairs[(p, q)] = 1
    return ZSet(pairs)


# ---- 3-D circuit ----------------------------------------------------------


def _build_stratified_circuit(
    *,
    parallelism: int = 1,
) -> "_StratifiedKernel":
    """Shared 3-D kernel for the canonical dynamic-level evaluator.

    Time axes are ``(outer=0, stratum=1, inner=2)``.

    Inputs:
      - ``edb: Input(..., Time1D)`` pushed at ``(outer,)``.
      - ``program: Input(..., Time2D)`` pushed at ``(outer, stratum)``.
      - ``state: Input(..., Time3D)`` pushed at ``(outer, stratum, inner)``
        by the saturation driver (diff output of ``body_out``).

    The kernel always uses:
      - hybrid cross-stratum fact carryover,
      - inner-local rewrites,
      - static-inner-right join specialization for the program-derived
        join sites.
    """
    fact_group: ZSetAddition[Fact] = ZSetAddition()
    rule_group: ZSetAddition[Rule] = ZSetAddition()
    rewrite_group: ZSetAddition[ProvenanceIndexedRewrite] = ZSetAddition()
    ext_dir_group: ZSetAddition[ExtendedDirection] = ZSetAddition()
    jorder_group: ZSetAddition[tuple[str, ColumnReference]] = ZSetAddition()
    signal_group: ZSetAddition[Signal] = ZSetAddition()
    gatekeep_group: ZSetAddition[IndexedGatekeepEntry] = ZSetAddition()
    indexed_fact_group: ZSetAddition[IndexedFact] = ZSetAddition()
    product_group: ProductGroup[EDB, ZSet[ProvenanceIndexedRewrite]] = ProductGroup(
        fact_group, rewrite_group
    )

    lattice = Time3D

    # edb: 1D → 3D via axis=1, axis=2 chained.
    edb = Input(fact_group, Time1D)
    edb_2d = TimeAxisIntroduction(edb, fact_group, Time2D, axis=1)
    edb_3d = TimeAxisIntroduction(edb_2d, fact_group, lattice, axis=2)

    # program: 2D (outer, stratum) → 3D via axis=2.
    program = Input(rule_group, Time2D)
    sig_2d = Lift1(program, sig, signal_group)
    sig_stream = TimeAxisIntroduction(sig_2d, signal_group, lattice, axis=2)
    ext_dir_2d = Lift1(program, ext_dir, ext_dir_group)
    ext_dir_stream = TimeAxisIntroduction(ext_dir_2d, ext_dir_group, lattice, axis=2)
    jorder_2d = Lift1(program, jorder, jorder_group)
    jorder_stream = TimeAxisIntroduction(jorder_2d, jorder_group, lattice, axis=2)

    rewrites_seed = StreamIntroduction(
        ZSet({(0, _rewrite_monoid.identity()): 1}),
        rewrite_group, lattice,
    )

    state = Input(product_group, lattice)
    state_facts = Lift1(state, lambda p: p[0], fact_group)
    state_rewrites = Lift1(state, lambda p: p[1], rewrite_group)
    within_diff_facts = Delay(state_facts, lattice, axis=2)
    within_diff_rewrites = Delay(state_rewrites, lattice, axis=2)
    # Hybrid feedback. Each stratum sees:
    #   - prior stratum's fixpoint as a single diff at (o, s, k=0),
    #   - its own within-stratum diffs at (o, s, k>0).
    # DLD's internal ``Integrate(diff, axis=inner)`` then broadcasts
    # the seed across every ``k≥0`` while summing within-stratum diffs
    # incrementally. Sum stays a clean diff stream (no pre-integration),
    # so each join cell processes ``O(Δ)`` not ``O(|full state|)``.
    #
    # Only facts are carried across strata. Rewrites remain local to
    # the current stratum's inner recursion.
    fp_facts_2d = TimeAxisElimination(
        state_facts, axis=2, lattice_in=lattice, lattice_out=Time2D,
    )
    prev_fp_facts_2d = Delay(fp_facts_2d, Time2D, axis=1)
    seed_facts = TimeAxisIntroduction(
        prev_fp_facts_2d, fact_group, lattice, axis=2,
    )
    distinct_facts = within_diff_facts + seed_facts
    distinct_rewrites = within_diff_rewrites

    axes3 = (0, 1, 2)
    gatekeep = DLDSortMergeJoin3DStaticInnerRight(
        Index(distinct_rewrites, lambda rw: rw[0], rewrite_group),
        Index(ext_dir_stream, lambda ed: ed[0], ext_dir_group),
        lambda _k, l, r: (r[1], r[2], l[1], r[3]),
        gatekeep_group, lattice, axes=axes3,
    )

    indexed_facts_raw = DLDSortMergeJoin3DStaticInnerRight(
        Index(distinct_facts, lambda f: f[0], fact_group),
        Index(jorder_stream, lambda j: j[0], jorder_group),
        lambda _k, fact, jo: _index_fact(jo[1], fact),
        indexed_fact_group, lattice, axes=axes3,
    )
    distinct_indexed_facts = DLDDistinct3D(
        indexed_facts_raw, indexed_fact_group, lattice, axes=axes3,
    )

    positive_atoms = Select(
        gatekeep,
        lambda gk: gk[1] is None or ("!" not in gk[1][0]),
    )
    negative_atoms = Select(
        gatekeep,
        lambda gk: not (gk[1] is None or ("!" not in gk[1][0])),
    )

    product = DLDSortMergeJoin3D(
        Index(positive_atoms, _gatekeep_join_key, gatekeep_group),
        Index(distinct_indexed_facts, lambda ifact: ifact[0], indexed_fact_group),
        _indexed_product_proj,
        rewrite_group, lattice, axes=axes3,
    )

    proj = Project(
        negative_atoms,
        lambda gk: (gk[0], gk[2]),
        rewrite_group,
    )

    anti_product = DLDSortMergeJoin3D(
        Index(negative_atoms, _gatekeep_join_key, gatekeep_group),
        Index(distinct_indexed_facts, lambda ifact: ifact[0], indexed_fact_group),
        lambda _k, gk, _ifact: (gk[0], gk[2]),
        rewrite_group, lattice, axes=axes3,
    )

    final_product = product + proj - anti_product

    ground = DLDSortMergeJoin3DStaticInnerRight(
        Index(final_product, lambda r: r[0], rewrite_group),
        Index(sig_stream, lambda s: s[0], signal_group),
        lambda _k, l, r: l[1].apply(r[1]),
        fact_group, lattice, axes=axes3,
    )

    next_facts = DLDDistinct3D(
        ground + edb_3d, fact_group, lattice, axes=axes3,
    )
    next_rewrites = DLDDistinct3D(
        final_product + rewrites_seed, rewrite_group, lattice, axes=axes3,
    )

    body_out = Lift2(next_facts, next_rewrites, lambda a, b: (a, b), product_group)
    # Collapse stratum axis (sum across strata) then inner (sum across
    # inner ticks) to get the final fact set per outer tick.
    observable_facts = Lift1(body_out, lambda p: p[0], fact_group)
    obs_no_inner = TimeAxisElimination(
        observable_facts, axis=2, lattice_in=lattice, lattice_out=Time2D,
    )
    observable = TimeAxisElimination(
        obs_no_inner, axis=1, lattice_in=Time2D, lattice_out=Time1D,
    )

    evaluator = Evaluator(observable, parallelism=parallelism)
    setattr(observable, "_evaluator", evaluator)
    c = _StratifiedKernel(
        observable=observable,
        body_out=body_out,
        evaluator=evaluator,
        state=state,
        edb=edb,
        program=program,
        product_group=product_group,
        lattice=lattice,
    )
    c.gatekeep = gatekeep
    c.product = product
    c.final_product = final_product
    c.positive_atoms = positive_atoms
    c.negative_atoms = negative_atoms
    c.proj = proj
    c.anti_product = anti_product
    c.ground = ground
    c.next_facts = next_facts
    c.next_rewrites = next_rewrites
    c.distinct_facts = distinct_facts
    c.distinct_rewrites = distinct_rewrites
    c.seed_facts = seed_facts
    c.ext_dir_stream = ext_dir_stream
    c.sig_stream = sig_stream
    c.indexed_facts_raw = indexed_facts_raw
    c.distinct_indexed_facts = distinct_indexed_facts
    return c


def IncrementalDatalogStratified(
    parallelism: int = 1,
) -> "DynamicStratifiedCircuit":
    """Canonical stratified Datalog evaluator."""
    return _IncrementalDatalogStratifiedDynamicLevels(
        parallelism=parallelism,
    )


# ---- Driver --------------------------------------------------------------


class _StratifiedKernel:
    def __init__(
        self,
        *,
        observable,
        body_out,
        evaluator: Evaluator,
        state: Input,
        edb: Input,
        program: Input,
        product_group,
        lattice,
    ) -> None:
        self.observable = observable
        self.body_out = body_out
        self.evaluator = evaluator
        self.state = state
        self.edb = edb
        self.program = program
        self.product_group = product_group
        self.lattice = lattice

    def body_at(self, t):
        return self.evaluator.at_op(self.body_out, t)

    def observable_at(self, t):
        return self.evaluator.at(t)


class DynamicStratifiedCircuit(_StratifiedKernel):
    """Dynamic-level wrapper around the hybrid 3D stratified circuit.

    External callers push raw fact and rule deltas on the public
    ``edb`` and ``program`` inputs (both 1-D outer-time streams). A
    positive Datalog sidecar incrementally maintains ``level(P,S)``
    facts from the program dependency deltas. ``saturate_stratified``
    reads the current cumulative levels and routes rule deltas into
    the wrapped 3D circuit at the corresponding stratum cells.
    """

    def __init__(
        self,
        *,
        observable,
        body_out,
        evaluator: Evaluator,
        state: Input,
        edb: Input,
        program: Input,
        product_group,
        lattice,
        edb_outer: Input,
        program_outer: Input,
        level_circuit: DatalogCircuit,
        reach_circuit: DatalogCircuit,
        analysis_sidecar: DynamicLevelSidecar,
    ) -> None:
        super().__init__(
            observable=observable,
            body_out=body_out,
            evaluator=evaluator,
            state=state,
            edb=edb,
            program=program,
            product_group=product_group,
            lattice=lattice,
        )
        self.analysis_sidecar = analysis_sidecar
        self.edb_outer = edb_outer
        self.program_outer = program_outer
        self.level_circuit = level_circuit
        self.reach_circuit = reach_circuit
        self._level_facts_cumulative: dict[Fact, int] = {}
        self._reach_facts_cumulative: dict[Fact, int] = {}
        self._program_cumulative: dict[Rule, int] = {}
        self._max_level_hint = 0
        self._succ_upto = 0
        self._recursive_predicates: set[PredicateName] = set()
        self._recursive_levels: set[int] = set()


def _IncrementalDatalogStratifiedDynamicLevels(
    parallelism: int = 1,
) -> DynamicStratifiedCircuit:
    """Dynamic-level stratified Datalog.

    Uses a positive Datalog sidecar to incrementally maintain
    ``level(P,S)`` facts from rule-head and dependency deltas, then
    routes raw rule deltas into the working hybrid 3D circuit by the
    current head-predicate level.
    """
    base = _build_stratified_circuit(parallelism=parallelism)

    fact_group: ZSetAddition[Fact] = ZSetAddition()
    rule_group: ZSetAddition[Rule] = ZSetAddition()
    predicate_group: ZSetAddition[PredicateName] = ZSetAddition()
    dependency_group: ZSetAddition[PredicateDependency] = ZSetAddition()
    level_group: ZSetAddition[PredicateLevel] = ZSetAddition()

    edb_outer = Input(fact_group, Time1D)
    program_outer = Input(rule_group, Time1D)
    level_circuit = IncrementalDatalogWithIndexing(parallelism=parallelism)
    level_circuit.program.push((0,), _level_program())
    reach_circuit = IncrementalDatalogWithIndexing(parallelism=parallelism)
    reach_circuit.program.push((0,), _reach_program())

    level_current_outer = Integrate(
        Lift1(level_circuit.observable, _extract_level_facts, level_group),
        Time1D,
        axis=0,
    )
    reach_pair_group: ZSetAddition[RecursivePredicatePair] = ZSetAddition()
    reach_current_outer = Integrate(
        Lift1(reach_circuit.observable, _extract_reach_facts, reach_pair_group),
        Time1D,
        axis=0,
    )
    same_level_recursive_outer = Lift2(
        reach_current_outer,
        level_current_outer,
        same_level_recursive_pairs,
        reach_pair_group,
    )

    head_predicates_outer = Lift1(program_outer, head_predicates, predicate_group)
    positive_dependencies_outer = Lift1(program_outer, positive_dependencies, dependency_group)
    negative_dependencies_outer = Lift1(program_outer, negative_dependencies, dependency_group)

    analysis_sidecar = DynamicLevelSidecar(
        head_predicates_3d=TimeAxisIntroduction(
            TimeAxisIntroduction(head_predicates_outer, predicate_group, Time2D, axis=1),
            predicate_group, base.lattice, axis=2,
        ),
        positive_dependencies_3d=TimeAxisIntroduction(
            TimeAxisIntroduction(positive_dependencies_outer, dependency_group, Time2D, axis=1),
            dependency_group, base.lattice, axis=2,
        ),
        negative_dependencies_3d=TimeAxisIntroduction(
            TimeAxisIntroduction(negative_dependencies_outer, dependency_group, Time2D, axis=1),
            dependency_group, base.lattice, axis=2,
        ),
        positive_reach_3d=TimeAxisIntroduction(
            TimeAxisIntroduction(reach_current_outer, reach_pair_group, Time2D, axis=1),
            reach_pair_group, base.lattice, axis=2,
        ),
        level_at_least_3d=TimeAxisIntroduction(
            TimeAxisIntroduction(level_current_outer, level_group, Time2D, axis=1),
            level_group, base.lattice, axis=2,
        ),
        same_level_recursive_3d=TimeAxisIntroduction(
            TimeAxisIntroduction(same_level_recursive_outer, reach_pair_group, Time2D, axis=1),
            reach_pair_group, base.lattice, axis=2,
        ),
    )

    dynamic = DynamicStratifiedCircuit(
        observable=base.observable,
        body_out=base.body_out,
        evaluator=base.evaluator,
        state=base.state,
        edb=base.edb,
        program=base.program,
        product_group=base.product_group,
        lattice=base.lattice,
        edb_outer=edb_outer,
        program_outer=program_outer,
        level_circuit=level_circuit,
        reach_circuit=reach_circuit,
        analysis_sidecar=analysis_sidecar,
    )
    for name, value in base.__dict__.items():
        setattr(dynamic, name, value)
    dynamic.analysis_sidecar = analysis_sidecar
    dynamic.edb_inner = base.edb
    dynamic.program_inner = base.program
    dynamic.edb_outer = edb_outer
    dynamic.program_outer = program_outer
    dynamic.edb = edb_outer
    dynamic.program = program_outer
    dynamic.level_circuit = level_circuit
    dynamic.reach_circuit = reach_circuit
    dynamic._level_facts_cumulative = {}
    dynamic._reach_facts_cumulative = {}
    dynamic._program_cumulative = {}
    dynamic._max_level_hint = 0
    dynamic._succ_upto = 0
    dynamic._recursive_predicates = set()
    dynamic._recursive_levels = set()
    return dynamic


def saturate_stratified(
    circuit: DynamicStratifiedCircuit,
    *,
    outer_tick: int = 0,
    max_inner: int = 1 << 16,
) -> list[int]:
    """Drive dynamic stratified Datalog to an inner fixpoint at ``outer_tick``.

    Returns the final inner tick reached per active stratum.
    """
    return _saturate_stratified_dynamic(
        circuit,
        outer_tick=outer_tick,
        max_inner=max_inner,
    )


def _update_level_cumulative(
    circuit: DynamicStratifiedCircuit,
    outer_tick: int,
) -> dict[PredicateName, int]:
    level_delta = circuit.level_circuit.observable_at((outer_tick,))
    cumulative = circuit._level_facts_cumulative
    for fact, weight in level_delta.inner.items():
        cumulative[fact] = cumulative.get(fact, 0) + weight
        if cumulative[fact] == 0:
            del cumulative[fact]
    levels: dict[PredicateName, int] = {}
    for (pred, args), weight in cumulative.items():
        if pred != "level" or weight <= 0:
            continue
        name = cast(PredicateName, args[0])
        level = cast(int, args[1])
        prior = levels.get(name)
        if prior is None or level > prior:
            levels[name] = level
    return levels


def _update_reach_cumulative(
    circuit: DynamicStratifiedCircuit,
    outer_tick: int,
) -> set[RecursivePredicatePair]:
    reach_delta = circuit.reach_circuit.observable_at((outer_tick,))
    cumulative = circuit._reach_facts_cumulative
    for fact, weight in reach_delta.inner.items():
        cumulative[fact] = cumulative.get(fact, 0) + weight
        if cumulative[fact] == 0:
            del cumulative[fact]
    return {
        cast(RecursivePredicatePair, (args[0], args[1]))
        for (pred, args), weight in cumulative.items()
        if pred == "reach" and weight > 0
    }


def _saturate_stratified_dynamic(
    circuit: DynamicStratifiedCircuit,
    *,
    outer_tick: int = 0,
    max_inner: int = 1 << 16,
) -> list[int]:
    """Incremental dynamic-level scheduler.

    1. Consume raw rule/fact deltas from the public outer-time
       ``program`` / ``edb`` inputs.
    2. Push dependency facts into the level sidecar and saturate it.
    3. Read the current cumulative ``level(P,S)`` facts.
    4. Route raw rule deltas into the wrapped 3D circuit by head level.
    5. Saturate every active stratum ``0 .. max_level``.
    """
    raw_program_delta = circuit.program._values.get((outer_tick,), ZSet({}))
    raw_edb_delta = circuit.edb._values.get((outer_tick,), ZSet({}))

    for rule, weight in raw_program_delta.inner.items():
        circuit._program_cumulative[rule] = circuit._program_cumulative.get(rule, 0) + weight
        if circuit._program_cumulative[rule] == 0:
            del circuit._program_cumulative[rule]

    current_program = ZSet({k: v for k, v in circuit._program_cumulative.items() if v > 0})
    num_heads = len(head_predicates(current_program).inner)
    level_edb_delta = _program_delta_to_level_edb(raw_program_delta)
    if num_heads > circuit._succ_upto:
        succ_delta = _succ_facts(circuit._succ_upto, num_heads)
        level_edb_delta = ZSetAddition[Fact]().add(level_edb_delta, succ_delta)
        circuit._succ_upto = num_heads
    circuit.level_circuit.edb.push((outer_tick,), level_edb_delta)
    circuit.reach_circuit.edb.push(
        (outer_tick,),
        ZSet(
            {
                ("pos_dep", dep): weight
                for dep, weight in positive_dependencies(raw_program_delta).inner.items()
            }
        ),
    )

    saturate(circuit.level_circuit, outer_tick=outer_tick, max_inner=max_inner)
    saturate(circuit.reach_circuit, outer_tick=outer_tick, max_inner=max_inner)
    level_map = _update_level_cumulative(circuit, outer_tick)
    reach_pairs = _update_reach_cumulative(circuit, outer_tick)
    recursive_predicates: set[PredicateName] = set()
    for p, q in reach_pairs:
        if (q, p) not in reach_pairs:
            continue
        if level_map.get(p) != level_map.get(q):
            continue
        recursive_predicates.add(p)
        recursive_predicates.add(q)
    circuit._recursive_predicates = recursive_predicates
    circuit._recursive_levels = {level_map[p] for p in recursive_predicates if p in level_map}
    max_level = max(level_map.values(), default=0)
    circuit._max_level_hint = max_level

    # Route raw deltas into the wrapped 3D circuit.
    circuit.edb_inner.push((outer_tick,), raw_edb_delta)
    by_level: dict[int, dict[Rule, int]] = {}
    for rule, weight in raw_program_delta.inner.items():
        head_pred = rule[0][0]
        level = level_map.get(head_pred, 0)
        bucket = by_level.setdefault(level, {})
        bucket[rule] = bucket.get(rule, 0) + weight
    for s in range(max_level + 1):
        circuit.program_inner.push((outer_tick, s), ZSet(by_level.get(s, {})))

    per_stratum_k: list[int] = []
    state = circuit.state
    for s in range(max_level + 1):
        k_final = -1
        for k in range(max_inner):
            diff_pair = circuit.body_at((outer_tick, s, k))
            is_zero = not diff_pair[0].inner and not diff_pair[1].inner
            state.push((outer_tick, s, k), diff_pair)
            if is_zero and k > 0:
                k_final = k
                break
        else:
            raise RuntimeError(
                f"dynamic level stratum {s} did not converge in {max_inner} inner ticks"
            )
        per_stratum_k.append(k_final)
    return per_stratum_k


__all__ = [
    "DynamicStratifiedCircuit",
    "IncrementalDatalogStratified",
    "head_predicates",
    "negative_dependencies",
    "positive_dependencies",
    "saturate_stratified",
]
