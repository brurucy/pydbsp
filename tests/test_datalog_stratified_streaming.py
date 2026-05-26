"""Multi-outer streaming tests for :class:`IncrementalDatalogStratified`.

The stratified evaluator is incremental across outer ticks w.r.t. both
facts AND rules. Each :meth:`saturate` call advances one outer tick;
the running level subgraph can change rule strata, and the body turns
those changes into guarded-rule value deltas.

These tests pin two invariants:

1. **Facts-streaming batch vs singletons**. Pushing all facts at once
   then ``saturate()`` once produces the same cumulative as pushing
   facts one at a time with a ``saturate()`` between each.
2. **Rules-streaming batch vs singletons**. Same, but for rule
   deltas. A rule added at outer ``N`` updates running levels at ``N``
   and the guarded body program changes incrementally via ordinary
   Z-set deltas.

Each test runs the same program/EDB through up to three modes and
asserts the cumulative results match."""

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


def _run_facts_singletons(program: ZSet[dlg.Rule], edb: ZSet[dlg.Fact]) -> ZSet[dlg.Fact]:
    """Push rules at outer 0. Push one fact at a time across outer
    ticks; ``saturate()`` between each push. Returns the final
    cumulative."""
    s = IncrementalDatalogStratified()
    s.push_rules(program)
    last: ZSet[dlg.Fact] = ZSet({})
    if not edb.inner:
        return s.saturate()
    for fact, weight in edb.inner.items():
        s.push_facts(ZSet({fact: weight}))
        last = s.saturate()
    return last


def _run_rules_singletons(program: ZSet[dlg.Rule], edb: ZSet[dlg.Fact]) -> ZSet[dlg.Fact]:
    """Push EDB at outer 0. Push one rule at a time across outer
    ticks; ``saturate()`` between each push. Returns the final
    cumulative. Skips intermediate programs that are unstratifiable
    (they would raise during ``push_rules``)."""
    s = IncrementalDatalogStratified()
    s.push_facts(edb)
    last: ZSet[dlg.Fact] = ZSet({})
    if not program.inner:
        return s.saturate()
    for rule, weight in program.inner.items():
        s.push_rules(ZSet({rule: weight}))
        last = s.saturate()
    return last


def _run_interleaved(program: ZSet[dlg.Rule], edb: ZSet[dlg.Fact]) -> ZSet[dlg.Fact]:
    """Interleave one rule and one fact per outer tick. Whichever
    runs out first, the remaining items go in subsequent ticks."""
    s = IncrementalDatalogStratified()
    rules = list(program.inner.items())
    facts = list(edb.inner.items())
    n = max(len(rules), len(facts))
    last: ZSet[dlg.Fact] = ZSet({})
    if n == 0:
        return s.saturate()
    for i in range(n):
        if i < len(rules):
            r, w = rules[i]
            s.push_rules(ZSet({r: w}))
        if i < len(facts):
            f, w = facts[i]
            s.push_facts(ZSet({f: w}))
        last = s.saturate()
    return last


def _assert_all_modes_equal(
    label: str,
    program: ZSet[dlg.Rule],
    edb: ZSet[dlg.Fact],
) -> None:
    """Assert that batch, facts-singletons, interleaved, and
    rules-singletons all return the same cumulative."""
    batched = _run_batch(program, edb)
    facts_singletons = _run_facts_singletons(program, edb)
    assert batched == facts_singletons, (
        f"{label}: batch vs facts-singletons differ:\n  batch:  {batched.inner}\n  facts:  {facts_singletons.inner}"
    )
    interleaved = _run_interleaved(program, edb)
    assert batched == interleaved, (
        f"{label}: batch vs interleaved differ:\n  batch:       {batched.inner}\n  interleaved: {interleaved.inner}"
    )
    rules_singletons = _run_rules_singletons(program, edb)
    assert batched == rules_singletons, (
        f"{label}: batch vs rules-singletons differ:\n  batch: {batched.inner}\n  rules: {rules_singletons.inner}"
    )


# ---- Test programs --------------------------------------------------------


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


# ---- Multi-outer streaming parity tests -----------------------------------


def test_streaming_negation_no_blocker() -> None:
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
    _assert_all_modes_equal("negation_no_blocker", program, edb)


def test_streaming_negation_blocker_filters() -> None:
    """A late-arriving ``dead(bob)`` must retract ``alive(bob)``
    derived at an earlier outer tick. Tests cross-outer bilinear
    retraction through the anti-product."""
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
    _assert_all_modes_equal("negation_with_blocker", program, edb)


def test_streaming_two_strata_chain() -> None:
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
    _assert_all_modes_equal("two_strata_chain", program, edb)


def test_streaming_canonical_counterexample() -> None:
    """The canonical counterexample (positive cycle at stratum 1 with
    negation against stratum-0 ``tc``). All four modes — batch,
    facts-singletons, rules-singletons, interleaved — must produce
    exactly ``a(0), b(0)`` plus the stratum-0 derivations.

    Pushing the cycle-closing rule ``a(?X) :- b(?X), node(?X)`` at
    an outer tick after ``b(0)`` is already derived exercises the
    bilinear's cross-outer cube together with the guarded-program
    mechanism. The guard atoms turn the new rule's level into a
    signed value delta, and the driver revisits prior strata so the
    cycle's mutual derivations settle within stratum 1."""
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
    _assert_all_modes_equal("canonical_counterexample", program, edb)


def test_streaming_pure_positive_program() -> None:
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
    _assert_all_modes_equal("pure_positive", program, edb)


def test_streaming_late_arriving_negation_target_retracts() -> None:
    """``alive(?X) :- node(?X), !dead(?X)``. Push nodes first
    (alives derive), then ``dead(alice)`` at a later outer. Expected:
    ``alive(alice)`` retracts. Stresses cross-outer anti-product
    retraction explicitly via the streaming order."""
    program = _rules(
        (
            (
                ("alive", ("?X",)),
                ("node", ("?X",)),
                ("!dead", ("?X",)),
            ),
            1,
        ),
    )
    edb = _facts(
        (("node", ("alice",)), 1),
        (("node", ("bob",)), 1),
        (("dead", ("alice",)), 1),
    )
    _assert_all_modes_equal("late_arriving_dead", program, edb)


def test_streaming_rules_grow_program_height_late() -> None:
    """Push the level-0 rule first, then the level-1 rule (which
    raises the program's stratum count). The running level subgraph must
    pick up the new max-level at the second outer; the body must
    activate the guarded rule at its correct stratum."""
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
        (("dead", ("alice",)), 1),
        (("person", ("bob",)), 1),
    )
    _assert_all_modes_equal("rule_added_late", program, edb)


def test_late_lower_stratum_fact_retracts_upper_negation() -> None:
    """A later level-0 rule/fact can invalidate a level-1 negated
    derivation from a prior outer. This pins batch-vs-staged
    equivalence for the original stratified routing semantic gap."""
    r_a_p = (("a", ("?X",)), ("p", ("?X",)))
    r_c_not_a = (
        ("c", ("?X",)),
        ("domain", ("?X",)),
        ("!a", ("?X",)),
    )
    r_a_q = (("a", ("?X",)), ("q", ("?X",)))
    f0 = ZSet(
        {
            ("p", (1,)): 1,
            ("domain", (1,)): 1,
            ("domain", (2,)): 1,
        }
    )
    f1 = ZSet({("q", (2,)): 1})

    batch = IncrementalDatalogStratified()
    batch.push_rules(ZSet({r_a_p: 1, r_c_not_a: 1, r_a_q: 1}))
    batch.push_facts(ZSet({**f0.inner, **f1.inner}))
    expected = batch.saturate()

    staged = IncrementalDatalogStratified()
    staged.push_rules(ZSet({r_a_p: 1, r_c_not_a: 1}))
    staged.push_facts(f0)
    staged.saturate()
    staged.push_rules(ZSet({r_a_q: 1}))
    staged.push_facts(f1)
    actual = staged.saturate()

    assert actual == expected
    assert actual.inner.get(("c", (2,)), 0) == 0


def test_upper_rule_retraction_after_max_level_drop_revisits_stratum() -> None:
    """When the current program height drops, historical upper strata
    still have to be clocked so their old derivations can retract."""
    r_a_p = (("a", ("?X",)), ("p", ("?X",)))
    r_c_not_a = (
        ("c", ("?X",)),
        ("domain", ("?X",)),
        ("!a", ("?X",)),
    )
    edb = ZSet({("domain", (1,)): 1})

    staged = IncrementalDatalogStratified()
    staged.push_rules(ZSet({r_a_p: 1, r_c_not_a: 1}))
    staged.push_facts(edb)
    staged.saturate()
    staged.push_rules(ZSet({r_c_not_a: -1}))
    actual = staged.saturate()

    truth = IncrementalDatalogStratified()
    truth.push_rules(ZSet({r_a_p: 1}))
    truth.push_facts(edb)
    expected = truth.saturate()

    assert actual == expected
    assert actual.inner.get(("c", (1,)), 0) == 0


# ---- Programs with multiple weighted facts (exercise the retraction path
#     when the same fact arrives via different singleton orderings) --------


def test_streaming_weighted_facts() -> None:
    program = _rules(
        ((("derived", ("?X",)), ("base", ("?X",))), 1),
    )
    edb = _facts(
        (("base", (0,)), 2),
        (("base", (1,)), 1),
    )
    _assert_all_modes_equal("weighted_facts", program, edb)


@pytest.mark.parametrize("trial", range(3))
def test_streaming_random_permutation_invariance(trial: int) -> None:
    """For a small program / EDB, permuting the singleton push order
    should not change the cumulative. Three trials with the program's
    rules in different orders inside the singleton driver."""
    import random

    rng = random.Random(trial)

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
        ((("hub", ("?X",)), ("reach", ("?X", "?Y")), ("!dead", ("?Y",))), 1),
    )
    edb = _facts(
        (("edge", (0, 1)), 1),
        (("edge", (1, 2)), 1),
        (("edge", (2, 3)), 1),
        (("dead", (3,)), 1),
    )
    batched = _run_batch(program, edb)

    rules_list = list(program.inner.items())
    facts_list = list(edb.inner.items())
    rng.shuffle(rules_list)
    rng.shuffle(facts_list)

    s = IncrementalDatalogStratified()
    n = max(len(rules_list), len(facts_list))
    last: ZSet[dlg.Fact] = ZSet({})
    for i in range(n):
        if i < len(rules_list):
            r, w = rules_list[i]
            s.push_rules(ZSet({r: w}))
        if i < len(facts_list):
            f, w = facts_list[i]
            s.push_facts(ZSet({f: w}))
        last = s.saturate()
    assert batched == last, (
        f"trial {trial}: batched != shuffled-stream:\n  batched: {batched.inner}\n  stream:  {last.inner}"
    )


def test_rule_retraction_should_unsay_derived_fact() -> None:
    """At outer 0, push a rule and EDB that derive `b(1)`. At outer 1,
    retract the rule. The cumulative derived facts must match what a
    fresh evaluator on the empty program (just the EDB) would produce.
    """
    r = (("b", ("?X",)), ("p", ("?X",)))
    edb = ZSet({("p", (1,)): 1})

    s = IncrementalDatalogStratified()
    s.push_rules(ZSet({r: 1}))
    s.push_facts(edb)
    s.saturate()  # outer 0 — derives b(1)

    s.push_rules(ZSet({r: -1}))
    streamed = s.saturate()  # outer 1 — should drop b(1)

    truth = IncrementalDatalogStratified()
    truth.push_facts(edb)
    expected = truth.saturate()

    assert streamed == expected, (
        f"rule retraction did not unsay derived fact:\n"
        f"  streamed: {dict(streamed.inner)}\n"
        f"  expected: {dict(expected.inner)}"
    )


def test_fact_retraction_should_unsay_derived_fact() -> None:
    """At outer 0, EDB ``p(1)`` and rule ``b :- p`` derive ``b(1)``.
    At outer 1, retract ``p(1)``. The cumulative should match a fresh
    evaluator on the empty EDB (no derivations)."""
    program = ZSet({(("b", ("?X",)), ("p", ("?X",))): 1})

    s = IncrementalDatalogStratified()
    s.push_rules(program)
    s.push_facts(ZSet({("p", (1,)): 1}))
    s.saturate()  # outer 0 — derives b(1)

    s.push_facts(ZSet({("p", (1,)): -1}))
    streamed = s.saturate()  # outer 1 — should drop p(1) and b(1)

    truth = IncrementalDatalogStratified()
    truth.push_rules(program)
    # empty EDB
    expected = truth.saturate()

    assert streamed == expected, (
        f"stratified fact retraction did not unsay derived fact:\n"
        f"  streamed: {dict(streamed.inner)}\n"
        f"  expected: {dict(expected.inner)}"
    )
