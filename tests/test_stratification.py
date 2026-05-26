"""Tests for the v1 stratification sidecars, re-expressed on top of v2's
:class:`IncrementalDatalogBody`.

The v1 stratified Datalog evaluator carried two sidecar circuits:

* ``level_circuit``: a positive Datalog program that computes
  ``level(P, S)``, the level of every head predicate. Positive
  dependencies preserve the level. Negative dependencies push the head
  one strictly higher.
* ``reach_circuit``: a positive Datalog program that computes the
  transitive closure of the positive-dependency graph. Combined with
  the level map, it identifies same-level positive cycles.

We exercise both circuits here as standalone positive Datalog programs.
The point is to confirm that the v2 body reproduces the dependency
analysis the v1 evaluator depended on, so that the eventual scope-B
driver can either reuse the analysis verbatim or replace it with
something Python-side.

Predicate-name constants below are written as plain strings (``"a"``,
``"b"``, …). The unifier discriminates Variables by the leading ``?``
prefix on a string, so a Constant without ``?`` is safe in atom-
argument position. Variables in the level and reach programs are
inlined as string literals (``"?P"``, ``"?S"``, …) rather than
declared via ``Variable``."""

from __future__ import annotations

from pydbsp import datalog as dlg
from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.core import Antichain, dbsp_time
from pydbsp.evaluate import Evaluator
from pydbsp.operator import Input, LiftStreamIntroduction
from pydbsp.progress import Time
from pydbsp.indexed_relational_operators import IndexedIncrementalDatalogBody
from pydbsp.storage import DictStorage
from pydbsp.zset import ZSet, ZSetAddition

# ---- Wiring -----------------------------------------------------------------


def _build():
    fg = ZSetAddition[dlg.Fact]()
    e = Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(2)),
        group=fg,
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
        fact_group=fg,
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
        fg,
    )


def _saturate(
    e,
    edb_1d,
    program_1d,
    sf,
    sr,
    seed_1d,
    body,
    fg,
    edb: ZSet[dlg.Fact],
    program: ZSet[dlg.Rule],
) -> ZSet[dlg.Fact]:
    e.push(edb_1d, edb)
    e.push(program_1d, program)
    e.push(seed_1d, ZSet({(0, dlg._rewrite_monoid.identity()): 1}))
    cumulative = fg.identity()
    for k, (df, dr) in e.saturate_inner(
        body,
        0,
        is_empty=lambda p: not p[0].inner and not p[1].inner,
    ):
        cumulative = fg.add(cumulative, df)
        e.push(sf, df, t=(0, k))
        e.push(sr, dr, t=(0, k))
    return cumulative


# ---- Program declarations ---------------------------------------------------

LEVEL_PROGRAM = ZSet(
    {
        # level(?P, 0) :- head(?P).
        (("level", ("?P", 0)), ("head", ("?P",))): 1,
        # level(?P, ?S) :- pos_dep(?P, ?Q), level(?Q, ?S).
        (("level", ("?P", "?S")), ("pos_dep", ("?P", "?Q")), ("level", ("?Q", "?S"))): 1,
        # level(?P, ?S1) :- neg_dep(?P, ?Q), level(?Q, ?S), succ(?S, ?S1).
        (
            ("level", ("?P", "?S1")),
            ("neg_dep", ("?P", "?Q")),
            ("level", ("?Q", "?S")),
            ("succ", ("?S", "?S1")),
        ): 1,
    }
)

REACH_PROGRAM = ZSet(
    {
        # reach(?P, ?Q) :- pos_dep(?P, ?Q).
        (("reach", ("?P", "?Q")), ("pos_dep", ("?P", "?Q"))): 1,
        # reach(?P, ?R) :- pos_dep(?P, ?Q), reach(?Q, ?R).
        (("reach", ("?P", "?R")), ("pos_dep", ("?P", "?Q")), ("reach", ("?Q", "?R"))): 1,
    }
)


def _succ_chain(n: int) -> dict[dlg.Fact, int]:
    """``succ(i, i+1)`` for ``i`` in ``0..n-1``. The level circuit's
    successor relation has to be supplied as EDB."""
    return {("succ", (i, i + 1)): 1 for i in range(n)}


def _level_assignment(derived: ZSet[dlg.Fact]) -> dict[str, int]:
    """Read out the ``level`` facts and keep the max level per predicate.
    The level circuit derives ``level_at_least`` (every level up to the
    predicate's actual stratum), and the assignment is its
    coordinate-wise max."""
    out: dict[str, int] = {}
    for (pred, args), weight in derived.inner.items():
        if pred != "level" or weight <= 0:
            continue
        p, s = args
        if p not in out or s > out[p]:
            out[p] = s
    return out


def _reach_pairs(derived: ZSet[dlg.Fact]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for (pred, args), weight in derived.inner.items():
        if pred != "reach" or weight <= 0:
            continue
        out.add(args)
    return out


# ---- Level circuit tests ----------------------------------------------------


def test_level_single_head_only_lands_at_zero() -> None:
    """One head predicate, no dependencies. Its level is 0."""
    e, edb_1d, program_1d, sf, sr, seed_1d, body, fg = _build()
    edb = ZSet({("head", ("a",)): 1})
    derived = _saturate(e, edb_1d, program_1d, sf, sr, seed_1d, body, fg, edb, LEVEL_PROGRAM)
    assert _level_assignment(derived) == {"a": 0}


def test_level_positive_chain_keeps_everyone_at_zero() -> None:
    """A linear positive chain. All three predicates land at level 0."""
    e, edb_1d, program_1d, sf, sr, seed_1d, body, fg = _build()
    edb = ZSet(
        {
            ("head", ("a",)): 1,
            ("head", ("b",)): 1,
            ("head", ("c",)): 1,
            ("pos_dep", ("b", "a")): 1,
            ("pos_dep", ("c", "b")): 1,
        }
    )
    derived = _saturate(e, edb_1d, program_1d, sf, sr, seed_1d, body, fg, edb, LEVEL_PROGRAM)
    assert _level_assignment(derived) == {"a": 0, "b": 0, "c": 0}


def test_level_one_negation_creates_two_strata() -> None:
    """A single negative dependency: ``q`` depends on ``!p``. The
    sink predicate ``p`` lands at level 0 and the source ``q`` is
    pushed one stratum up."""
    e, edb_1d, program_1d, sf, sr, seed_1d, body, fg = _build()
    edb = ZSet(
        {
            ("head", ("p",)): 1,
            ("head", ("q",)): 1,
            ("neg_dep", ("q", "p")): 1,
            **_succ_chain(2),
        }
    )
    derived = _saturate(e, edb_1d, program_1d, sf, sr, seed_1d, body, fg, edb, LEVEL_PROGRAM)
    assert _level_assignment(derived) == {"p": 0, "q": 1}


def test_level_two_negations_create_three_strata() -> None:
    """A chain of two negative dependencies stacks the predicates into
    three strata."""
    e, edb_1d, program_1d, sf, sr, seed_1d, body, fg = _build()
    edb = ZSet(
        {
            ("head", ("a",)): 1,
            ("head", ("b",)): 1,
            ("head", ("c",)): 1,
            ("neg_dep", ("b", "a")): 1,
            ("neg_dep", ("c", "b")): 1,
            **_succ_chain(3),
        }
    )
    derived = _saturate(e, edb_1d, program_1d, sf, sr, seed_1d, body, fg, edb, LEVEL_PROGRAM)
    assert _level_assignment(derived) == {"a": 0, "b": 1, "c": 2}


def test_level_mixes_positive_and_negative_paths() -> None:
    """Positive dependencies preserve the level, a single negative
    dependency raises it. The combination assigns four predicates
    across two strata."""
    e, edb_1d, program_1d, sf, sr, seed_1d, body, fg = _build()
    edb = ZSet(
        {
            ("head", ("a",)): 1,
            ("head", ("b",)): 1,
            ("head", ("c",)): 1,
            ("head", ("d",)): 1,
            ("pos_dep", ("b", "a")): 1,
            ("neg_dep", ("c", "b")): 1,
            ("pos_dep", ("d", "c")): 1,
            **_succ_chain(2),
        }
    )
    derived = _saturate(e, edb_1d, program_1d, sf, sr, seed_1d, body, fg, edb, LEVEL_PROGRAM)
    assert _level_assignment(derived) == {"a": 0, "b": 0, "c": 1, "d": 1}


def test_level_positive_cycle_does_not_raise_the_level() -> None:
    """Two predicates positively reference each other. Both stay at
    level 0. Same-stratum positive recursion of this shape is what the
    reach circuit identifies, and the inner fixpoint resolves it."""
    e, edb_1d, program_1d, sf, sr, seed_1d, body, fg = _build()
    edb = ZSet(
        {
            ("head", ("a",)): 1,
            ("head", ("b",)): 1,
            ("pos_dep", ("a", "b")): 1,
            ("pos_dep", ("b", "a")): 1,
        }
    )
    derived = _saturate(e, edb_1d, program_1d, sf, sr, seed_1d, body, fg, edb, LEVEL_PROGRAM)
    assert _level_assignment(derived) == {"a": 0, "b": 0}


# ---- Reach circuit tests ----------------------------------------------------


def test_reach_single_edge() -> None:
    e, edb_1d, program_1d, sf, sr, seed_1d, body, fg = _build()
    edb = ZSet({("pos_dep", ("a", "b")): 1})
    derived = _saturate(e, edb_1d, program_1d, sf, sr, seed_1d, body, fg, edb, REACH_PROGRAM)
    assert _reach_pairs(derived) == {("a", "b")}


def test_reach_linear_chain_yields_full_transitive_closure() -> None:
    e, edb_1d, program_1d, sf, sr, seed_1d, body, fg = _build()
    edb = ZSet(
        {
            ("pos_dep", ("a", "b")): 1,
            ("pos_dep", ("b", "c")): 1,
            ("pos_dep", ("c", "d")): 1,
        }
    )
    derived = _saturate(e, edb_1d, program_1d, sf, sr, seed_1d, body, fg, edb, REACH_PROGRAM)
    expected = {
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("a", "c"),
        ("b", "d"),
        ("a", "d"),
    }
    assert _reach_pairs(derived) == expected


def test_reach_positive_cycle_creates_self_loops() -> None:
    """A two-node positive cycle gives every node a self-loop in
    addition to reaching the other. The combined output is the basis
    for the same-level recursive pair extraction below."""
    e, edb_1d, program_1d, sf, sr, seed_1d, body, fg = _build()
    edb = ZSet(
        {
            ("pos_dep", ("a", "b")): 1,
            ("pos_dep", ("b", "a")): 1,
        }
    )
    derived = _saturate(e, edb_1d, program_1d, sf, sr, seed_1d, body, fg, edb, REACH_PROGRAM)
    assert _reach_pairs(derived) == {
        ("a", "b"),
        ("b", "a"),
        ("a", "a"),
        ("b", "b"),
    }


def test_reach_disjoint_components_stay_disjoint() -> None:
    e, edb_1d, program_1d, sf, sr, seed_1d, body, fg = _build()
    edb = ZSet(
        {
            ("pos_dep", ("a", "b")): 1,
            ("pos_dep", ("c", "d")): 1,
        }
    )
    derived = _saturate(e, edb_1d, program_1d, sf, sr, seed_1d, body, fg, edb, REACH_PROGRAM)
    assert _reach_pairs(derived) == {("a", "b"), ("c", "d")}


# ---- Combined: same-level recursive pair extraction -------------------------


def test_same_level_recursive_pairs_only_within_a_single_stratum() -> None:
    """Two predicates form a positive cycle at level 0. A third
    predicate that negatively depends on one of them is pushed to
    level 1. Reach reports the cycle pairs (plus self-loops), and the
    same-level filter retains only those whose endpoints share a
    level."""
    edb = ZSet(
        {
            ("head", ("a",)): 1,
            ("head", ("b",)): 1,
            ("head", ("c",)): 1,
            ("pos_dep", ("a", "b")): 1,
            ("pos_dep", ("b", "a")): 1,
            ("neg_dep", ("c", "a")): 1,
            **_succ_chain(2),
        }
    )

    e1, *args1, body1, fg1 = _build()
    levels = _level_assignment(_saturate(e1, *args1, body1, fg1, edb, LEVEL_PROGRAM))
    assert levels == {"a": 0, "b": 0, "c": 1}

    e2, *args2, body2, fg2 = _build()
    reach = _reach_pairs(_saturate(e2, *args2, body2, fg2, edb, REACH_PROGRAM))

    same_level = {(p, q) for (p, q) in reach if (q, p) in reach and levels.get(p) == levels.get(q)}
    assert same_level == {("a", "b"), ("b", "a"), ("a", "a"), ("b", "b")}
