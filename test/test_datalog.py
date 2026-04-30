"""Tests for the Datalog circuits —
``pydbsp.algorithms.IncrementalDatalog`` and
``IncrementalDatalogWithNegation``.

Iteration is driven via ``History`` + ``run_to_fixpoint``; each
circuit returns a ``DatalogCircuit`` handle whose ``feedback`` is
seeded with ``seed`` and whose ``body_out`` is pushed back on every
observer tick until the observable converges.
"""

from pydbsp.algorithms import (
    EDB,
    IncrementalDatalog,
    IncrementalDatalogWithNegation,
    IncrementalDatalogStratified,
    Program,
    Variable,
    saturate,
    saturate_stratified,
)
from pydbsp.history import History
from pydbsp.zset import ZSet


def _run(program: Program, edb: EDB) -> EDB:
    circuit = IncrementalDatalog()
    circuit.edb.push((0,), edb)
    circuit.program.push((0,), program)
    saturate(circuit)
    h: History[EDB, tuple[int]] = History(circuit.observable)
    assert h.try_step()
    return h.at((0,))


def _run_singletons(program: Program, edb: EDB) -> EDB:
    """Feed one fact per t0 tick (empty program-delta after t0=0),
    saturating at each tick. Sum the outer-diff observations across
    ticks to get cumulative derived facts — that sum should equal
    the batched ``_run`` result.
    """
    from pydbsp.zset import ZSetAddition

    addn: ZSetAddition = ZSetAddition()

    circuit = IncrementalDatalog()
    fact_list = list(edb.inner.items())
    if not fact_list:
        circuit.edb.push((0,), edb)
        circuit.program.push((0,), program)
        saturate(circuit)
        h: History[EDB, tuple[int]] = History(circuit.observable)
        h.try_step()
        return h.at((0,))

    h = History(circuit.observable)
    acc: EDB = addn.identity()
    for t0, (fact, weight) in enumerate(fact_list):
        circuit.edb.push((t0,), ZSet({fact: weight}))
        circuit.program.push((t0,), program if t0 == 0 else ZSet({}))
        saturate(circuit, outer_tick=t0)
        assert h.try_step()
        acc = addn.add(acc, h.at((t0,)))
    return acc


def _run_with_negation(program: Program, edb: EDB) -> EDB:
    circuit = IncrementalDatalogWithNegation()
    circuit.edb.push((0,), edb)
    circuit.program.push((0,), program)
    saturate(circuit)
    h: History[EDB, tuple[int]] = History(circuit.observable)
    assert h.try_step()
    return h.at((0,))


def _cumulative(batches: list[EDB]) -> list[EDB]:
    """Produce cumulative EDBs from incremental batches."""
    from pydbsp.zset import ZSetAddition

    addn: ZSetAddition = ZSetAddition()
    acc: EDB = addn.identity()
    out: list[EDB] = []
    for b in batches:
        acc = addn.add(acc, b)
        out.append(acc)
    return out


def _run_negation_batches(program: Program, batches: list[EDB]) -> list[EDB]:
    return [_run_with_negation(program, cum) for cum in _cumulative(batches)]


def _run_stratified(program: Program, edb: EDB) -> EDB:
    circuit = IncrementalDatalogStratified()
    circuit.edb.push((0,), edb)
    circuit.program.push((0,), program)
    saturate_stratified(circuit)
    h: History[EDB, tuple[int]] = History(circuit.observable)
    assert h.try_step()
    return h.at((0,))


def _run_stratified_batches(program: Program, batches: list[EDB]) -> list[EDB]:
    return [_run_stratified(program, cum) for cum in _cumulative(batches)]


def _run_stratified_incremental_batches(program: Program, batches: list[EDB]) -> list[EDB]:
    from pydbsp.zset import ZSetAddition

    addn: ZSetAddition = ZSetAddition()
    circuit = IncrementalDatalogStratified()
    h: History[EDB, tuple[int]] = History(circuit.observable)
    outputs: list[EDB] = []
    for t0, batch in enumerate(batches):
        circuit.edb.push((t0,), batch)
        circuit.program.push((t0,), program if t0 == 0 else ZSet({}))
        saturate_stratified(circuit, outer_tick=t0)
        assert h.try_step()
        outputs.append(h.at((t0,)))
    # Convert outer deltas to cumulative outputs for parity with
    # ``_run_stratified_batches``.
    acc: EDB = addn.identity()
    cumulative: list[EDB] = []
    for delta in outputs:
        acc = addn.add(acc, delta)
        cumulative.append(acc)
    return cumulative


def _rules(*rules: tuple[tuple, int]) -> Program:
    z = ZSet({})
    for rule, weight in rules:
        z.inner[rule] = z.inner.get(rule, 0) + weight
    return ZSet({k: v for k, v in z.inner.items() if v != 0})


def _facts(*facts: tuple[tuple, int]) -> EDB:
    z = ZSet({})
    for fact, weight in facts:
        z.inner[fact] = z.inner.get(fact, 0) + weight
    return ZSet({k: v for k, v in z.inner.items() if v != 0})


# ---- Batch vs singleton equivalence ----------------------------------------


def _assert_batch_eq_singletons(program: Program, edb: EDB) -> None:
    """Verify that feeding ``edb`` as a single batch at t0=0 and as
    per-fact singletons across successive t0 ticks yields the same
    cumulative derived-fact set."""
    batched = _run(program, edb)
    singleton = _run_singletons(program, edb)
    assert batched == singleton, f"batch != singletons: {batched.inner} vs {singleton.inner}"


def test_batch_eq_singletons_empty_program() -> None:
    edb = _facts((("E", (0,)), 1), (("E", (1,)), 1))
    _assert_batch_eq_singletons(_rules(), edb)


def test_batch_eq_singletons_single_rule() -> None:
    program = _rules(
        ((("A", (Variable("X"),)), ("E", (Variable("X"),))), 1),
    )
    edb = _facts((("E", (0,)), 1), (("E", (1,)), 1))
    _assert_batch_eq_singletons(program, edb)


def test_batch_eq_singletons_unification() -> None:
    program = _rules(
        (
            (
                ("P", (Variable("X"), Variable("Y"))),
                ("E", (Variable("X"), Variable("Y"))),
                ("N", (Variable("Y"),)),
            ),
            1,
        ),
    )
    edb = _facts(
        (("E", (0, 1)), 1),
        (("E", (2, 3)), 1),
        (("N", (1,)), 1),
    )
    _assert_batch_eq_singletons(program, edb)


def test_batch_eq_singletons_transitive_closure() -> None:
    program = _rules(
        (
            (
                ("T", (Variable("X"), Variable("Y"))),
                ("E", (Variable("X"), Variable("Y"))),
            ),
            1,
        ),
        (
            (
                ("T", (Variable("X"), Variable("Z"))),
                ("E", (Variable("X"), Variable("Y"))),
                ("T", (Variable("Y"), Variable("Z"))),
            ),
            1,
        ),
    )
    edb = _facts(
        (("E", (0, 1)), 1),
        (("E", (1, 2)), 1),
        (("E", (2, 3)), 1),
    )
    _assert_batch_eq_singletons(program, edb)


def test_batch_eq_singletons_two_rules_union() -> None:
    program = _rules(
        ((("Good", (Variable("X"),)), ("B", (Variable("X"),))), 1),
        ((("Good", (Variable("X"),)), ("C", (Variable("X"),))), 1),
    )
    edb = _facts((("B", (0,)), 1), (("C", (1,)), 1))
    _assert_batch_eq_singletons(program, edb)


def test_batch_eq_singletons_constant_filter() -> None:
    program = _rules(
        ((("OnlyTwo", (Variable("X"),)), ("E", (Variable("X"), 2))), 1),
    )
    edb = _facts(
        (("E", (0, 2)), 1),
        (("E", (1, 3)), 1),
        (("E", (5, 2)), 1),
    )
    _assert_batch_eq_singletons(program, edb)


# ---- Positive Datalog ------------------------------------------------------


def test_empty_program_returns_edb() -> None:
    edb = _facts((("E", (0,)), 1), (("E", (1,)), 1))
    program = _rules()
    assert _run(program, edb) == edb


def test_empty_edb_returns_empty() -> None:
    program = _rules(
        ((("A", (Variable("X"),)), ("E", (Variable("X"),))), 1),
    )
    assert _run(program, _facts()) == ZSet({})


def test_single_rule_projects_edb_into_head() -> None:
    # A(X) :- E(X).
    program = _rules(
        ((("A", (Variable("X"),)), ("E", (Variable("X"),))), 1),
    )
    edb = _facts((("E", (0,)), 1), (("E", (1,)), 1))
    assert _run(program, edb) == _facts(
        (("E", (0,)), 1),
        (("E", (1,)), 1),
        (("A", (0,)), 1),
        (("A", (1,)), 1),
    )


def test_unification_binds_variables_across_body() -> None:
    # P(X, Y) :- E(X, Y), N(Y).
    program = _rules(
        (
            (
                ("P", (Variable("X"), Variable("Y"))),
                ("E", (Variable("X"), Variable("Y"))),
                ("N", (Variable("Y"),)),
            ),
            1,
        ),
    )
    edb = _facts(
        (("E", (0, 1)), 1),
        (("E", (2, 3)), 1),
        (("N", (1,)), 1),
    )
    result = _run(program, edb)
    # Only (0, 1) survives — Y=3 has no N(3) match.
    assert result.inner.get(("P", (0, 1)), 0) == 1
    assert result.inner.get(("P", (2, 3)), 0) == 0


def test_recursive_transitive_closure() -> None:
    # T(X, Y) :- E(X, Y).
    # T(X, Z) :- E(X, Y), T(Y, Z).
    program = _rules(
        (
            (
                ("T", (Variable("X"), Variable("Y"))),
                ("E", (Variable("X"), Variable("Y"))),
            ),
            1,
        ),
        (
            (
                ("T", (Variable("X"), Variable("Z"))),
                ("E", (Variable("X"), Variable("Y"))),
                ("T", (Variable("Y"), Variable("Z"))),
            ),
            1,
        ),
    )
    edb = _facts(
        (("E", (0, 1)), 1),
        (("E", (1, 2)), 1),
        (("E", (2, 3)), 1),
    )
    result = _run(program, edb)
    for pair in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        assert result.inner.get(("T", pair), 0) == 1, f"missing T{pair}"


def test_two_rules_into_same_head_are_unioned() -> None:
    # Good(X) :- B(X).
    # Good(X) :- C(X).
    program = _rules(
        ((("Good", (Variable("X"),)), ("B", (Variable("X"),))), 1),
        ((("Good", (Variable("X"),)), ("C", (Variable("X"),))), 1),
    )
    edb = _facts((("B", (0,)), 1), (("C", (1,)), 1))
    result = _run(program, edb)
    assert result.inner.get(("Good", (0,)), 0) == 1
    assert result.inner.get(("Good", (1,)), 0) == 1


def test_constant_term_in_body_filters() -> None:
    # OnlyTwo(X) :- E(X, 2).
    program = _rules(
        ((("OnlyTwo", (Variable("X"),)), ("E", (Variable("X"), 2))), 1),
    )
    edb = _facts(
        (("E", (0, 2)), 1),
        (("E", (1, 3)), 1),
        (("E", (5, 2)), 1),
    )
    result = _run(program, edb)
    assert result.inner.get(("OnlyTwo", (0,)), 0) == 1
    assert result.inner.get(("OnlyTwo", (5,)), 0) == 1
    assert result.inner.get(("OnlyTwo", (1,)), 0) == 0


# ---- Datalog with negation -------------------------------------------------


def test_negation_transient_fact_then_blocker() -> None:
    """``A(X) :- E(X), !B(X).`` — A(0) holds while B is absent, then
    is retracted once B(0) is introduced.
    """
    program = _rules(
        ((("A", (Variable("X"),)), ("A", (Variable("X"),))), 1),
        ((("A", (Variable("X"),)), ("E", (Variable("X"),)), ("!B", (Variable("X"),))), 1),
    )
    outputs = _run_negation_batches(
        program,
        [
            _facts((("E", (0,)), 1)),
            _facts((("B", (0,)), 1)),
        ],
    )
    assert outputs[0].inner.get(("A", (0,)), 0) == 1, f"batch 0: {outputs[0].inner}"
    assert outputs[1].inner.get(("A", (0,)), 0) == 0, f"batch 1: {outputs[1].inner}"


def test_negation_dual_support_survives_single_support_retraction() -> None:
    """``A`` derived via two rules; ``Good(X) :- A(X), !Bad(X)``
    disappears when Bad fires, re-emerges when Bad is retracted.
    """
    program = _rules(
        ((("A", (Variable("X"),)), ("B", (Variable("X"),))), 1),
        ((("A", (Variable("X"),)), ("C", (Variable("X"),))), 1),
        ((("Good", (Variable("X"),)), ("A", (Variable("X"),)), ("!Bad", (Variable("X"),))), 1),
    )
    outputs = _run_negation_batches(
        program,
        [
            _facts((("B", (0,)), 1)),
            _facts((("C", (0,)), 1)),
            _facts((("Bad", (0,)), 1)),
            _facts((("C", (0,)), -1)),
            _facts((("Bad", (0,)), -1)),
        ],
    )
    assert outputs[0].inner.get(("A", (0,)), 0) == 1
    assert outputs[1].inner.get(("A", (0,)), 0) == 1
    assert outputs[2].inner.get(("Good", (0,)), 0) == 0
    assert outputs[3].inner.get(("A", (0,)), 0) == 1
    assert outputs[4].inner.get(("Good", (0,)), 0) == 1


def test_negation_recursive_blocker_retracts_and_restores_good() -> None:
    """Recursive ``T`` (transitive closure) plus a ``Bad`` predicate
    that depends on ``T``; adding ``Block`` turns on ``Bad`` for
    upstream nodes, retracting it restores ``Good``.
    """
    program = _rules(
        (
            (
                ("T", (Variable("X"), Variable("Y"))),
                ("E", (Variable("X"), Variable("Y"))),
            ),
            1,
        ),
        (
            (
                ("T", (Variable("X"), Variable("Z"))),
                ("E", (Variable("X"), Variable("Y"))),
                ("T", (Variable("Y"), Variable("Z"))),
            ),
            1,
        ),
        (
            (
                ("Bad", (Variable("X"),)),
                ("T", (Variable("X"), Variable("Y"))),
                ("Block", (Variable("Y"),)),
            ),
            1,
        ),
        (
            (
                ("Good", (Variable("X"),)),
                ("Node", (Variable("X"),)),
                ("!Bad", (Variable("X"),)),
            ),
            1,
        ),
    )
    outputs = _run_negation_batches(
        program,
        [
            _facts(
                (("Node", (0,)), 1),
                (("Node", (1,)), 1),
                (("Node", (2,)), 1),
                (("Node", (3,)), 1),
            ),
            _facts((("E", (0, 1)), 1)),
            _facts((("E", (1, 2)), 1)),
            _facts((("E", (2, 3)), 1)),
            _facts((("Block", (3,)), 1)),
            _facts((("Block", (3,)), -1)),
        ],
    )
    assert outputs[3].inner.get(("Good", (0,)), 0) == 1
    assert outputs[4].inner.get(("Good", (0,)), 0) == 0
    assert outputs[5].inner.get(("Good", (0,)), 0) == 1


def test_negation_same_batch_cancellation() -> None:
    """``A(X) :- E(X), !B(X)`` with E and B appearing and
    cancelling within the same cumulative EDB — A should reflect
    only the surviving weights.
    """
    program = _rules(
        ((("A", (Variable("X"),)), ("E", (Variable("X"),)), ("!B", (Variable("X"),))), 1),
    )
    outputs = _run_negation_batches(
        program,
        [
            _facts((("E", (0,)), 1), (("B", (0,)), 1), (("B", (0,)), -1)),
            _facts(
                (("E", (1,)), 1),
                (("E", (1,)), -1),
                (("B", (1,)), 1),
                (("B", (1,)), -1),
            ),
        ],
    )
    assert outputs[0].inner.get(("A", (0,)), 0) == 1
    assert outputs[1].inner.get(("A", (1,)), 0) == 0


def test_stratified_datalog_freezes_lower_stratum_before_upper_positive_cycle() -> None:
    """Regression for the notebook counterexample:

    tc(X,Y) :- e(X,Y).
    tc(X,Y) :- tc(X,Z), e(Z,Y).
    node(X) :- e(X,U).
    node(Y) :- e(V,Y).
    a(X) :- node(X), !tc(0,X).
    b(X) :- a(X).
    a(X) :- b(X), node(X).

    In the correct stratified semantics, ``tc/node`` are a lower stratum
    and the upper ``a<->b`` SCC sees the fully-settled lower stratum. So
    only ``0`` remains outside ``tc(0,_)``.
    """
    program = _rules(
        ((("tc", (Variable("X"), Variable("Y"))), ("e", (Variable("X"), Variable("Y")))), 1),
        ((("tc", (Variable("X"), Variable("Y"))), ("tc", (Variable("X"), Variable("Z"))), ("e", (Variable("Z"), Variable("Y")))), 1),
        ((("node", (Variable("X"),)), ("e", (Variable("X"), Variable("U")))), 1),
        ((("node", (Variable("Y"),)), ("e", (Variable("V"), Variable("Y")))), 1),
        ((("a", (Variable("X"),)), ("node", (Variable("X"),)), ("!tc", (0, Variable("X")))), 1),
        ((("b", (Variable("X"),)), ("a", (Variable("X"),))), 1),
        ((("a", (Variable("X"),)), ("b", (Variable("X"),)), ("node", (Variable("X"),))), 1),
    )
    edb = _facts(
        (("e", (0, 1)), 1),
        (("e", (1, 2)), 1),
        (("e", (2, 3)), 1),
    )
    result = _run_stratified(program, edb)
    for pair in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        assert result.inner.get(("tc", pair), 0) == 1, f"missing tc{pair}"
    for node in [0, 1, 2, 3]:
        assert result.inner.get(("node", (node,)), 0) == 1, f"missing node({node})"
    assert result.inner.get(("a", (0,)), 0) == 1
    assert result.inner.get(("b", (0,)), 0) == 1
    for bad in [1, 2, 3]:
        assert result.inner.get(("a", (bad,)), 0) == 0, f"spurious a({bad})"
        assert result.inner.get(("b", (bad,)), 0) == 0, f"spurious b({bad})"


def test_stratified_datalog_prefixes_do_not_latch_stale_negation() -> None:
    program = _rules(
        ((("tc", (Variable("X"), Variable("Y"))), ("e", (Variable("X"), Variable("Y")))), 1),
        ((("tc", (Variable("X"), Variable("Y"))), ("tc", (Variable("X"), Variable("Z"))), ("e", (Variable("Z"), Variable("Y")))), 1),
        ((("node", (Variable("X"),)), ("e", (Variable("X"), Variable("U")))), 1),
        ((("node", (Variable("Y"),)), ("e", (Variable("V"), Variable("Y")))), 1),
        ((("a", (Variable("X"),)), ("node", (Variable("X"),)), ("!tc", (0, Variable("X")))), 1),
        ((("b", (Variable("X"),)), ("a", (Variable("X"),))), 1),
        ((("a", (Variable("X"),)), ("b", (Variable("X"),)), ("node", (Variable("X"),))), 1),
    )
    outputs = _run_stratified_batches(
        program,
        [
            _facts((("e", (0, 1)), 1)),
            _facts((("e", (1, 2)), 1)),
            _facts((("e", (2, 3)), 1)),
        ],
    )
    assert outputs[0].inner.get(("a", (0,)), 0) == 1
    assert outputs[0].inner.get(("b", (0,)), 0) == 1
    assert outputs[1].inner.get(("a", (2,)), 0) == 0
    assert outputs[1].inner.get(("b", (2,)), 0) == 0
    assert outputs[2].inner.get(("a", (3,)), 0) == 0
    assert outputs[2].inner.get(("b", (3,)), 0) == 0


def test_dynamic_levels_skeleton_exposes_program_dependency_sidecar() -> None:
    program = _rules(
        ((("a", (Variable("X"),)), ("b", (Variable("X"),))), 1),
        ((("a", (Variable("X"),)), ("!c", (Variable("X"),))), 1),
    )
    circuit = IncrementalDatalogStratified()
    circuit.program.push((0,), program)

    heads = circuit.analysis_sidecar.head_predicates_3d.at((0, 0, 0))
    pos = circuit.analysis_sidecar.positive_dependencies_3d.at((0, 0, 0))
    neg = circuit.analysis_sidecar.negative_dependencies_3d.at((0, 0, 0))

    assert heads.inner == {"a": 1}
    assert pos.inner == {("a", "b"): 1}
    assert neg.inner == {("a", "c"): 1}

def test_dynamic_levels_sidecar_tracks_same_level_positive_recursion() -> None:
    program = _rules(
        ((("tc", (Variable("X"), Variable("Y"))), ("e", (Variable("X"), Variable("Y")))), 1),
        ((("tc", (Variable("X"), Variable("Y"))), ("tc", (Variable("X"), Variable("Z"))), ("e", (Variable("Z"), Variable("Y")))), 1),
        ((("node", (Variable("X"),)), ("e", (Variable("X"), Variable("U")))), 1),
        ((("node", (Variable("Y"),)), ("e", (Variable("V"), Variable("Y")))), 1),
        ((("a", (Variable("X"),)), ("node", (Variable("X"),)), ("!tc", (0, Variable("X")))), 1),
        ((("b", (Variable("X"),)), ("a", (Variable("X"),))), 1),
        ((("a", (Variable("X"),)), ("b", (Variable("X"),)), ("node", (Variable("X"),))), 1),
    )
    edb = _facts(
        (("e", (0, 1)), 1),
        (("e", (1, 2)), 1),
        (("e", (2, 3)), 1),
    )
    circuit = IncrementalDatalogStratified()
    circuit.edb.push((0,), edb)
    circuit.program.push((0,), program)
    saturate_stratified(circuit, outer_tick=0)

    same_level_recursive = circuit.analysis_sidecar.same_level_recursive_3d.at((0, 0, 0))
    pairs = set(same_level_recursive.inner.keys())
    assert ("tc", "tc") in pairs
    assert ("a", "a") in pairs
    assert ("a", "b") in pairs
    assert ("b", "a") in pairs
    assert ("b", "b") in pairs
    assert ("node", "node") not in pairs
    assert circuit._recursive_predicates == {"tc", "a", "b"}


def test_canonical_stratified_matches_counterexample() -> None:
    program = _rules(
        ((("tc", (Variable("X"), Variable("Y"))), ("e", (Variable("X"), Variable("Y")))), 1),
        ((("tc", (Variable("X"), Variable("Y"))), ("tc", (Variable("X"), Variable("Z"))), ("e", (Variable("Z"), Variable("Y")))), 1),
        ((("node", (Variable("X"),)), ("e", (Variable("X"), Variable("U")))), 1),
        ((("node", (Variable("Y"),)), ("e", (Variable("V"), Variable("Y")))), 1),
        ((("a", (Variable("X"),)), ("node", (Variable("X"),)), ("!tc", (0, Variable("X")))), 1),
        ((("b", (Variable("X"),)), ("a", (Variable("X"),))), 1),
        ((("a", (Variable("X"),)), ("b", (Variable("X"),)), ("node", (Variable("X"),))), 1),
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
    assert _run_stratified(program, edb) == expected


def test_canonical_stratified_prefixes_on_counterexample() -> None:
    program = _rules(
        ((("tc", (Variable("X"), Variable("Y"))), ("e", (Variable("X"), Variable("Y")))), 1),
        ((("tc", (Variable("X"), Variable("Y"))), ("tc", (Variable("X"), Variable("Z"))), ("e", (Variable("Z"), Variable("Y")))), 1),
        ((("node", (Variable("X"),)), ("e", (Variable("X"), Variable("U")))), 1),
        ((("node", (Variable("Y"),)), ("e", (Variable("V"), Variable("Y")))), 1),
        ((("a", (Variable("X"),)), ("node", (Variable("X"),)), ("!tc", (0, Variable("X")))), 1),
        ((("b", (Variable("X"),)), ("a", (Variable("X"),))), 1),
        ((("a", (Variable("X"),)), ("b", (Variable("X"),)), ("node", (Variable("X"),))), 1),
    )
    batches = [
        _facts((("e", (0, 1)), 1)),
        _facts((("e", (1, 2)), 1)),
        _facts((("e", (2, 3)), 1)),
    ]
    assert _run_stratified_incremental_batches(program, batches) == _run_stratified_batches(program, batches)
