"""Batch-vs-singletons invariant for :class:`IncrementalDatalogStratified`.

Mirrors the existing batch-vs-singletons clusters in
:mod:`tests.test_integration` (2-D bodies) and
:mod:`tests.test_3d_datalog_batch_vs_singletons` (3-D bodies),
extended to the stratified evaluator. For each ``(program, edb)`` pair
the cumulative derived-facts set must be invariant under

* pushing the whole EDB and saturating once (batch), or
* pushing one ``(fact, weight)`` at a time and saturating once at the
  end (singletons), or
* pushing one fact at a time and saturating after *each* push,
  taking the final saturate's output (intermediate saturates).

The stratified evaluator currently runs a single outer tick (each
saturate rebuilds the body circuit from the accumulated state), so
"singleton" here exercises ``push_facts`` accumulation and the
non-mutating nature of ``saturate``. The 2-D and 3-D body singleton
tests cover the multi-outer streaming bilinear directly.

All 9 stratified-Datalog acceptance tests in
``tests/test_datalog_stratified.py`` have a singleton variant here,
including the unstratifiable program rejection (which must reject in
all three modes)."""

from __future__ import annotations

import pytest

from pydbsp import datalog as dlg
from pydbsp.datalog_stratified import IncrementalDatalogStratified
from pydbsp.zset import ZSet


# ---- Helpers ---------------------------------------------------------------


def _run_batch(program: ZSet[dlg.Rule], edb: ZSet[dlg.Fact]) -> ZSet[dlg.Fact]:
    s = IncrementalDatalogStratified()
    s.push_rules(program)
    s.push_facts(edb)
    return s.saturate()


def _run_singletons_then_saturate(program: ZSet[dlg.Rule], edb: ZSet[dlg.Fact]) -> ZSet[dlg.Fact]:
    """Push every ``(fact, weight)`` separately, saturate once at the
    end. Tests that ``push_facts`` accumulates correctly."""
    s = IncrementalDatalogStratified()
    s.push_rules(program)
    for fact, weight in edb.inner.items():
        s.push_facts(ZSet({fact: weight}))
    return s.saturate()


def _run_singletons_with_intermediate_saturates(program: ZSet[dlg.Rule], edb: ZSet[dlg.Fact]) -> ZSet[dlg.Fact]:
    """Push one fact at a time and call ``saturate`` after each push.
    Returns the final saturate's output. Tests that ``saturate`` does
    not mutate accumulated state in a way that disrupts later
    saturates."""
    s = IncrementalDatalogStratified()
    s.push_rules(program)
    last_result: ZSet[dlg.Fact] = ZSet({})
    if not edb.inner:
        return s.saturate()
    for fact, weight in edb.inner.items():
        s.push_facts(ZSet({fact: weight}))
        last_result = s.saturate()
    return last_result


def _assert_three_modes_equal(program: ZSet[dlg.Rule], edb: ZSet[dlg.Fact]) -> None:
    batched = _run_batch(program, edb)
    singletons_end = _run_singletons_then_saturate(program, edb)
    singletons_intermediate = _run_singletons_with_intermediate_saturates(program, edb)
    assert batched == singletons_end, (
        f"batch vs singletons-then-saturate differ:\n"
        f"  batch:      {batched.inner}\n"
        f"  singletons: {singletons_end.inner}"
    )
    assert batched == singletons_intermediate, (
        f"batch vs intermediate-saturates differ:\n"
        f"  batch:        {batched.inner}\n"
        f"  intermediate: {singletons_intermediate.inner}"
    )


def _rules(*rules: tuple[dlg.Rule, int]) -> ZSet[dlg.Rule]:
    out: dict[dlg.Rule, int] = {}
    for r, w in rules:
        out[r] = out.get(r, 0) + w
    return ZSet({k: v for k, v in out.items() if v != 0})


def _facts(*facts: tuple[dlg.Fact, int]) -> ZSet[dlg.Fact]:
    out: dict[dlg.Fact, int] = {}
    for f, w in facts:
        out[f] = out.get(f, 0) + w
    return ZSet({k: v for k, v in out.items() if v != 0})


# ---- Singleton variants of every stratified test --------------------------


def test_stratified_batch_eq_singletons_negation_no_blocker() -> None:
    program = _rules(
        (
            (
                ("alive", ("?X",)),
                ("person", ("?X",)),
                ("!dead", ("?X",)),
            ),
            1,
        ),
    )
    edb = _facts(
        (("person", ("alice",)), 1),
        (("person", ("bob",)), 1),
    )
    _assert_three_modes_equal(program, edb)


def test_stratified_batch_eq_singletons_negation_blocker_filters() -> None:
    program = _rules(
        (
            (
                ("alive", ("?X",)),
                ("person", ("?X",)),
                ("!dead", ("?X",)),
            ),
            1,
        ),
    )
    edb = _facts(
        (("person", ("alice",)), 1),
        (("person", ("bob",)), 1),
        (("dead", ("bob",)), 1),
    )
    _assert_three_modes_equal(program, edb)


def test_stratified_batch_eq_singletons_two_strata_chain() -> None:
    program = _rules(
        (
            (
                ("mid", ("?X",)),
                ("bot", ("?X",)),
                ("!excluded", ("?X",)),
            ),
            1,
        ),
        (
            (
                ("top", ("?X",)),
                ("mid", ("?X",)),
                ("!blocked", ("?X",)),
            ),
            1,
        ),
    )
    edb = _facts(
        (("bot", (0,)), 1),
        (("bot", (1,)), 1),
        (("bot", (2,)), 1),
        (("excluded", (2,)), 1),
        (("blocked", (1,)), 1),
    )
    _assert_three_modes_equal(program, edb)


def test_stratified_batch_eq_singletons_freezes_lower_before_upper_cycle() -> None:
    program = _rules(
        ((("tc", ("?X", "?Y")), ("e", ("?X", "?Y"))), 1),
        ((("tc", ("?X", "?Y")), ("tc", ("?X", "?Z")), ("e", ("?Z", "?Y"))), 1),
        ((("node", ("?X",)), ("e", ("?X", "?U"))), 1),
        ((("node", ("?Y",)), ("e", ("?V", "?Y"))), 1),
        ((("a", ("?X",)), ("node", ("?X",)), ("!tc", (0, "?X"))), 1),
        ((("b", ("?X",)), ("a", ("?X",))), 1),
        ((("a", ("?X",)), ("b", ("?X",)), ("node", ("?X",))), 1),
    )
    edb = _facts(
        (("e", (0, 1)), 1),
        (("e", (1, 2)), 1),
        (("e", (2, 3)), 1),
    )
    _assert_three_modes_equal(program, edb)


def test_stratified_batch_eq_singletons_canonical_counterexample_full() -> None:
    """Same as the freezes-lower test but with the entire derived set
    asserted in :mod:`tests.test_datalog_stratified`. Here the
    invariant being checked is batch-vs-singletons; equality with the
    expected full set is left to the acceptance test."""
    program = _rules(
        ((("tc", ("?X", "?Y")), ("e", ("?X", "?Y"))), 1),
        ((("tc", ("?X", "?Y")), ("tc", ("?X", "?Z")), ("e", ("?Z", "?Y"))), 1),
        ((("node", ("?X",)), ("e", ("?X", "?U"))), 1),
        ((("node", ("?Y",)), ("e", ("?V", "?Y"))), 1),
        ((("a", ("?X",)), ("node", ("?X",)), ("!tc", (0, "?X"))), 1),
        ((("b", ("?X",)), ("a", ("?X",))), 1),
        ((("a", ("?X",)), ("b", ("?X",)), ("node", ("?X",))), 1),
    )
    edb = _facts(
        (("e", (0, 1)), 1),
        (("e", (1, 2)), 1),
        (("e", (2, 3)), 1),
    )
    _assert_three_modes_equal(program, edb)


def test_stratified_batch_eq_singletons_empty_program() -> None:
    edb = _facts(
        (("a", (0,)), 1),
        (("b", (1,)), 1),
    )
    _assert_three_modes_equal(_rules(), edb)


def test_stratified_batch_eq_singletons_empty_edb() -> None:
    program = _rules(
        ((("a", ("?X",)), ("b", ("?X",))), 1),
    )
    _assert_three_modes_equal(program, _facts())


def test_stratified_batch_eq_singletons_pure_positive_program() -> None:
    program = _rules(
        ((("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))), 1),
        (
            (
                ("reach", ("?X", "?Z")),
                ("edge", ("?X", "?Y")),
                ("reach", ("?Y", "?Z")),
            ),
            1,
        ),
    )
    edb = _facts(
        (("edge", (0, 1)), 1),
        (("edge", (1, 2)), 1),
        (("edge", (2, 3)), 1),
    )
    _assert_three_modes_equal(program, edb)


def test_stratified_batch_eq_singletons_unstratifiable_program_rejected_in_all_modes() -> None:
    """A program with a cycle through negation must be rejected by
    ``push_rules`` in every mode. ``push_facts`` and ``saturate`` are
    never called because rule validation is up-front."""
    program = _rules(
        ((("a", ("?X",)), ("!b", ("?X",))), 1),
        ((("b", ("?X",)), ("!a", ("?X",))), 1),
    )
    for build_one in (
        IncrementalDatalogStratified,
        IncrementalDatalogStratified,
        IncrementalDatalogStratified,
    ):
        with pytest.raises((ValueError, RuntimeError)):
            s = build_one()
            s.push_rules(program)
