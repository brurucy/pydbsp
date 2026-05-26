"""End-to-end tests for :class:`IncrementalDatalogStratified`.

Ported from the v1 ``test/test_datalog.py`` stratified test cluster.
Each test pushes a (program, edb) pair, drives the saturate loop, and
inspects the derived facts.

The class lives in :mod:`pydbsp.datalog_stratified` — a separate
module from the helpers in :mod:`pydbsp.datalog_stratified` because
the 3-D body is being built up incrementally and we want to insulate
the helpers from the in-progress 3-D primitives.

Variables are inline ``"?X"`` literals. Predicate-name constants are
plain strings."""

from __future__ import annotations

import pytest

from pydbsp import datalog as dlg
from pydbsp.datalog_stratified import IncrementalDatalogStratified
from pydbsp.zset import ZSet


# ---- Helpers ----------------------------------------------------------------


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


def _run(program: ZSet[dlg.Rule], edb: ZSet[dlg.Fact]) -> ZSet[dlg.Fact]:
    """Build a fresh evaluator, push the program and EDB at outer 0,
    saturate, and return the cumulative derived facts."""
    s = IncrementalDatalogStratified()
    s.push_rules(program)
    s.push_facts(edb)
    return s.saturate()


# ---- Negation smoke tests ---------------------------------------------------


def test_negation_no_blocker_passes() -> None:
    """``alive(?X) :- person(?X), !dead(?X)`` with no ``dead`` facts.
    Every ``person`` derives ``alive``."""
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
    expected = ZSet(
        {
            ("person", ("alice",)): 1,
            ("person", ("bob",)): 1,
            ("alive", ("alice",)): 1,
            ("alive", ("bob",)): 1,
        }
    )
    assert _run(program, edb) == expected


def test_negation_blocker_filters() -> None:
    """``alive(?X) :- person(?X), !dead(?X)`` with ``dead(bob)``.
    ``alive(alice)`` derives, ``alive(bob)`` does not."""
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
    expected = ZSet(
        {
            ("person", ("alice",)): 1,
            ("person", ("bob",)): 1,
            ("dead", ("bob",)): 1,
            ("alive", ("alice",)): 1,
        }
    )
    assert _run(program, edb) == expected


# ---- Multi-stratum tests ----------------------------------------------------


def test_two_strata_chain_through_negation() -> None:
    """A two-step stratification.

    ``mid(?X) :- bot(?X), !excluded(?X)``    -- stratum 1
    ``top(?X) :- mid(?X), !blocked(?X)``     -- stratum 2

    With ``bot(0..2)`` and ``excluded(2)``, ``mid`` covers ``{0, 1}``.
    With ``blocked(1)``, ``top`` covers ``{0}``."""
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
    expected = ZSet(
        {
            ("bot", (0,)): 1,
            ("bot", (1,)): 1,
            ("bot", (2,)): 1,
            ("excluded", (2,)): 1,
            ("blocked", (1,)): 1,
            ("mid", (0,)): 1,
            ("mid", (1,)): 1,
            ("top", (0,)): 1,
        }
    )
    assert _run(program, edb) == expected


def test_stratified_datalog_freezes_lower_stratum_before_upper_positive_cycle() -> None:
    """Regression for the v1 notebook counterexample.

    .. code-block:: text

        tc(?X, ?Y)  :- e(?X, ?Y).
        tc(?X, ?Y)  :- tc(?X, ?Z), e(?Z, ?Y).
        node(?X)    :- e(?X, ?U).
        node(?Y)    :- e(?V, ?Y).
        a(?X)       :- node(?X), !tc(0, ?X).
        b(?X)       :- a(?X).
        a(?X)       :- b(?X), node(?X).

    ``tc`` / ``node`` sit at stratum 0. The upper ``a <-> b`` SCC sits
    at stratum 1 and sees the fully-settled lower stratum. So only
    node ``0`` (which is not in ``tc(0, _)``) is in ``a`` / ``b``."""
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
    result = _run(program, edb)
    for pair in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        assert result.inner.get(("tc", pair), 0) == 1, f"missing tc{pair}"
    for n in [0, 1, 2, 3]:
        assert result.inner.get(("node", (n,)), 0) == 1, f"missing node({n})"
    assert result.inner.get(("a", (0,)), 0) == 1
    assert result.inner.get(("b", (0,)), 0) == 1
    for bad in [1, 2, 3]:
        assert result.inner.get(("a", (bad,)), 0) == 0, f"spurious a({bad})"
        assert result.inner.get(("b", (bad,)), 0) == 0, f"spurious b({bad})"


def test_canonical_stratified_matches_counterexample_in_full() -> None:
    """Same program as
    :func:`test_stratified_datalog_freezes_lower_stratum_before_upper_positive_cycle`,
    but asserts the entire derived set rather than a per-fact spot
    check."""
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
    expected = _facts(
        (("e", (0, 1)), 1),
        (("e", (1, 2)), 1),
        (("e", (2, 3)), 1),
        (("tc", (0, 1)), 1),
        (("tc", (1, 2)), 1),
        (("tc", (2, 3)), 1),
        (("tc", (0, 2)), 1),
        (("tc", (1, 3)), 1),
        (("tc", (0, 3)), 1),
        (("node", (0,)), 1),
        (("node", (1,)), 1),
        (("node", (2,)), 1),
        (("node", (3,)), 1),
        (("a", (0,)), 1),
        (("b", (0,)), 1),
    )
    assert _run(program, edb) == expected


# ---- Edge cases -------------------------------------------------------------


def test_empty_program_returns_edb() -> None:
    """No rules. The result equals the EDB."""
    edb = _facts(
        (("a", (0,)), 1),
        (("b", (1,)), 1),
    )
    assert _run(_rules(), edb) == edb


def test_empty_edb_with_positive_rules_returns_empty() -> None:
    """Rules but no facts. Nothing derives. Result is empty."""
    program = _rules(
        ((("a", ("?X",)), ("b", ("?X",))), 1),
    )
    assert _run(program, _facts()) == _facts()


def test_pure_positive_program_unaffected_by_stratification() -> None:
    """Without negation, every predicate sits at stratum 0. The
    stratified evaluator should produce the same answers a plain
    positive Datalog body would."""
    program = _rules(
        ((("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))), 1),
        ((("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))), 1),
    )
    edb = _facts(
        (("edge", (0, 1)), 1),
        (("edge", (1, 2)), 1),
        (("edge", (2, 3)), 1),
    )
    expected = ZSet(
        {
            ("edge", (0, 1)): 1,
            ("edge", (1, 2)): 1,
            ("edge", (2, 3)): 1,
            ("reach", (0, 1)): 1,
            ("reach", (1, 2)): 1,
            ("reach", (2, 3)): 1,
            ("reach", (0, 2)): 1,
            ("reach", (1, 3)): 1,
            ("reach", (0, 3)): 1,
        }
    )
    assert _run(program, edb) == expected


def test_unstratifiable_program_is_rejected() -> None:
    """A rule cycle through negation has no valid stratification. The
    class should reject the program at push or at saturate, not return
    garbage."""
    program = _rules(
        ((("a", ("?X",)), ("!b", ("?X",))), 1),
        ((("b", ("?X",)), ("!a", ("?X",))), 1),
    )
    edb = _facts((("a", (0,)), 1))
    with pytest.raises((ValueError, RuntimeError)):
        _run(program, edb)
