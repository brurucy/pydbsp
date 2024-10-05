from pydbsp.algorithms.datalog import (
    EDB,
    Fact,
    IncrementalDatalog,
    IncrementalDatalogWithIndexing,
    Program,
    Rule,
    Variable,
)
from pydbsp.stream import Stream, StreamHandle, step_until_fixpoint
from pydbsp.stream.functions.linear import stream_elimination
from pydbsp.zset import ZSetAddition

from test.test_zset import create_test_zset_graph


def create_test_edb(n: int) -> EDB:
    test_graph = create_test_zset_graph(n)
    group: ZSetAddition[Fact] = ZSetAddition()

    test_edb = group.identity()
    for k, v in test_graph.items():
        test_edb.inner[("E", (k[0], k[1]))] = v

    return test_edb


def from_fact_into_zset(fact: Fact, weight: int) -> EDB:
    edb: EDB = ZSetAddition().identity()
    edb.inner[fact] = weight

    return edb


def from_rule_into_zset(rule: Rule, weight: int) -> Program:
    program: Program = ZSetAddition().identity()
    program.inner[rule] = weight

    return program


def test_reachability() -> None:
    program_group: ZSetAddition[Rule] = ZSetAddition()
    program_stream = Stream(program_group)
    program_stream_h = StreamHandle(lambda: program_stream)

    # T(X, Y) <- E(X, Y)
    seed: Rule = (("T", (Variable("X"), Variable("Y"))), ("E", (Variable("X"), Variable("Y"))))
    # T(X, Z) <- E(X, Y), T(Y, Z)
    transitivity: Rule = (
        ("T", (Variable("X"), Variable("Z"))),
        ("E", (Variable("X"), Variable("Y"))),
        ("T", (Variable("Y"), Variable("Z"))),
    )
    reachability: Program = program_group.identity()
    reachability.inner[seed] = 1
    reachability.inner[transitivity] = 1

    program_stream.send(reachability)

    edb_group: ZSetAddition[Fact] = ZSetAddition()
    edb_stream = Stream(edb_group)
    edb_stream_h = StreamHandle(lambda: edb_stream)

    # (0, 1), (1, 2), (2, 3), (3, 4)
    test_edb = create_test_edb(4)
    edb_stream.send(test_edb)

    incremental_datalog = IncrementalDatalog(edb_stream_h, program_stream_h, None)
    incremental_indexed_datalog = IncrementalDatalogWithIndexing(edb_stream_h, program_stream_h, None)
    step_until_fixpoint(incremental_datalog)
    step_until_fixpoint(incremental_indexed_datalog)

    output_stream = incremental_datalog.output_handle().get()
    indexed_output_stream = incremental_indexed_datalog.output_handle().get()
    actual_output = stream_elimination(output_stream)
    indexed_actual_output = stream_elimination(indexed_output_stream)

    expected_output = edb_group.identity()
    for fact, weight in test_edb.items():
        expected_output.inner[fact] = weight
        expected_output.inner[("T", fact[1])] = weight

    expected_output.inner[("T", (0, 2))] = 1
    expected_output.inner[("T", (0, 3))] = 1
    expected_output.inner[("T", (0, 4))] = 1

    expected_output.inner[("T", (1, 3))] = 1
    expected_output.inner[("T", (1, 4))] = 1

    expected_output.inner[("T", (2, 4))] = 1

    assert actual_output == expected_output
    assert indexed_actual_output == expected_output


def test_triangle() -> None:
    program_group: ZSetAddition[Rule] = ZSetAddition()
    program_stream = Stream(program_group)
    program_stream_h = StreamHandle(lambda: program_stream)

    # T(X, Y) <- E(E, Y)
    triangle: Rule = (
        ("T", (Variable("A"), Variable("B"), Variable("C"))),
        ("E", (Variable("A"), Variable("B"))),
        ("E", (Variable("B"), Variable("C"))),
        ("E", (Variable("C"), Variable("A"))),
    )
    triangle_program: Program = program_group.identity()
    triangle_program.inner[triangle] = 1

    program_stream.send(triangle_program)

    edb_group: ZSetAddition[Fact] = ZSetAddition()
    edb_stream = Stream(edb_group)
    edb_stream_h = StreamHandle(lambda: edb_stream)

    # (0, 1), (1, 2), (2, 3), (3, 4)
    test_edb = edb_group.identity()
    test_edb.inner[("E", (1, 2))] = 1
    test_edb.inner[("E", (2, 3))] = 1
    test_edb.inner[("E", (3, 1))] = 1
    test_edb.inner[("E", (4, 5))] = 1
    test_edb.inner[("E", (5, 6))] = 1
    edb_stream.send(test_edb)

    incremental_datalog = IncrementalDatalog(edb_stream_h, program_stream_h, None)
    incremental_indexed_datalog = IncrementalDatalogWithIndexing(edb_stream_h, program_stream_h, None)
    step_until_fixpoint(incremental_datalog)
    step_until_fixpoint(incremental_indexed_datalog)
    output_stream = incremental_datalog.output_handle().get()
    indexed_output_stream = incremental_indexed_datalog.output_handle().get()
    actual_output = stream_elimination(output_stream)
    indexed_actual_output = stream_elimination(indexed_output_stream)

    expected_output = edb_group.identity()
    for fact, weight in test_edb.items():
        expected_output.inner[fact] = weight

    expected_output.inner[("T", (1, 2, 3))] = 1
    expected_output.inner[("T", (2, 3, 1))] = 1
    expected_output.inner[("T", (3, 1, 2))] = 1

    assert actual_output == expected_output
    assert indexed_actual_output == expected_output
