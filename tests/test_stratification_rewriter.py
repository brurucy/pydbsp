"""Tests for :class:`StratificationRewriter`.

The rewriter is the operator that joins a user rule stream with the
level circuit's ``level(P, S)`` fact stream, and emits the same rules
with a ``("stratum_ready", (s - 1,))`` guard prepended to every body.

Here we exercise the rewriter in isolation: feed it a synthetic level
fact stream alongside a tiny program, and check the output. The full
stratified driver (level circuit + rewriter + body, all wired
together) is left to later tests."""

from __future__ import annotations

from pydbsp import datalog as dlg
from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.core import Antichain, dbsp_time
from pydbsp.evaluate import Evaluator
from pydbsp.indexed_relational_operators import StratificationRewriter
from pydbsp.operator import Input, LiftStreamIntroduction
from pydbsp.progress import Time
from pydbsp.storage import DictStorage
from pydbsp.zset import ZSet, ZSetAddition


def _build():
    """Wire a rewriter on top of two 1-D Inputs lifted to 2-D. The
    test feeds rule deltas on the program input and level-fact deltas
    on the level input. Reads the rewriter's output at ``(0, k)`` to
    obtain the guarded-rule delta for that inner tick."""
    rule_group = ZSetAddition[dlg.Rule]()
    fact_group = ZSetAddition[dlg.Fact]()
    e = Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(2)),
        group=rule_group,
    )
    program_1d = Input[ZSet[dlg.Rule]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    level_facts_1d = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    program_2d = LiftStreamIntroduction[ZSet[dlg.Rule]](group=rule_group).connect(e.circuit, (program_1d,))
    level_facts_2d = LiftStreamIntroduction[ZSet[dlg.Fact]](group=fact_group).connect(e.circuit, (level_facts_1d,))
    guarded = StratificationRewriter(
        rule_group=rule_group,
        fact_group=fact_group,
    ).connect(e.circuit, (program_2d, level_facts_2d))
    return e, program_1d, level_facts_1d, guarded


def _accumulate_through_inner(e, guarded, outer: int = 0, max_inner: int = 16):
    """Sum the rewriter's per-inner-tick output deltas at ``outer``,
    yielding the cumulative guarded-rule set the body would have seen
    by the end of inner saturation."""
    group = ZSetAddition[dlg.Rule]()
    cumulative = group.identity()
    for k in range(max_inner):
        delta = e.read(guarded, (outer, k))
        if not delta.inner:
            break
        cumulative = group.add(cumulative, delta)
    return cumulative


def test_rewriter_prepends_guard_for_stratum_zero_predicate() -> None:
    """A predicate at stratum 0 has its rule guarded by
    ``stratum_ready(-1)``."""
    e, program_1d, level_1d, guarded = _build()
    rule = (("p", ("?X",)), ("q", ("?X",)))
    e.push(program_1d, ZSet({rule: 1}))
    e.push(level_1d, ZSet({("level", ("p", 0)): 1}))
    out = _accumulate_through_inner(e, guarded)
    assert out == ZSet(
        {
            (("p", ("?X",)), ("stratum_ready", (-1,)), ("q", ("?X",))): 1,
        }
    )


def test_rewriter_picks_max_stratum_when_multiple_at_least_facts() -> None:
    """The level circuit emits ``level_at_least(P, S)`` for every
    ``S`` up to the predicate's actual stratum. The rewriter must
    take the max."""
    e, program_1d, level_1d, guarded = _build()
    rule = (("p", ("?X",)), ("q", ("?X",)))
    e.push(program_1d, ZSet({rule: 1}))
    e.push(
        level_1d,
        ZSet(
            {
                ("level", ("p", 0)): 1,
                ("level", ("p", 1)): 1,
                ("level", ("p", 2)): 1,
            }
        ),
    )
    out = _accumulate_through_inner(e, guarded)
    assert out == ZSet(
        {
            (("p", ("?X",)), ("stratum_ready", (1,)), ("q", ("?X",))): 1,
        }
    )


def test_rewriter_emits_one_guarded_rule_per_input_rule() -> None:
    """Two rules with the same head get the same guard. Two rules
    with different heads get guards matching their own levels."""
    e, program_1d, level_1d, guarded = _build()
    rule_p_at_zero = (("p", ("?X",)), ("a", ("?X",)))
    rule_p_at_zero_via_b = (("p", ("?X",)), ("b", ("?X",)))
    rule_q_at_one = (("q", ("?X",)), ("!p", ("?X",)))
    e.push(
        program_1d,
        ZSet(
            {
                rule_p_at_zero: 1,
                rule_p_at_zero_via_b: 1,
                rule_q_at_one: 1,
            }
        ),
    )
    e.push(
        level_1d,
        ZSet(
            {
                ("level", ("p", 0)): 1,
                ("level", ("q", 0)): 1,
                ("level", ("q", 1)): 1,
            }
        ),
    )
    out = _accumulate_through_inner(e, guarded)
    expected = ZSet(
        {
            (("p", ("?X",)), ("stratum_ready", (-1,)), ("a", ("?X",))): 1,
            (("p", ("?X",)), ("stratum_ready", (-1,)), ("b", ("?X",))): 1,
            (("q", ("?X",)), ("stratum_ready", (0,)), ("!p", ("?X",))): 1,
        }
    )
    assert out == expected


def test_rewriter_emits_nothing_until_levels_arrive() -> None:
    """If the level circuit has not yet derived a level for the
    head, the bilinear join has no matching row and the rewriter
    emits no guarded rule. Once a level arrives at a later outer
    tick, the bilinear formula's cross-tick term picks up the rule
    that was pushed earlier."""
    e, program_1d, level_1d, guarded = _build()
    rule = (("p", ("?X",)), ("q", ("?X",)))
    e.push(program_1d, ZSet({rule: 1}))  # outer 0
    # No level fact pushed yet. Read at the first inner cell: empty.
    assert e.read(guarded, (0, 0)).inner == {}
    # Push the level fact at outer 1 explicitly. The bilinear join's
    # Iᵒ a × b term pairs the rule (cumulative through outer 1) with
    # the newly arrived level fact.
    e.push(level_1d, ZSet({("level", ("p", 0)): 1}), t=(1,))
    assert e.read(guarded, (1, 0)).inner == {
        (("p", ("?X",)), ("stratum_ready", (-1,)), ("q", ("?X",))): 1,
    }
