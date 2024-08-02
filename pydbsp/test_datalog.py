from datalog import Fact, IncrementalDatalog, Program, Rule, Variable, EDB
from stream_operators import stream_elimination
from stream import Stream, StreamHandle
from test_zset import create_test_zset_graph
from zset_operators import ZSetAddition


def create_test_edb(n: int) -> EDB:
    test_graph = create_test_zset_graph(n)
    group: ZSetAddition[Fact] = ZSetAddition()

    test_edb = group.identity()
    for k, v in test_graph.items():
        test_edb.inner[("E", (k[0], k[1]))] = v

    return test_edb


def test_reachability() -> None:
    program_group: ZSetAddition[Rule] = ZSetAddition()
    program_stream = Stream(program_group)
    program_stream_h = StreamHandle(lambda: program_stream)

    # T(X, Y) <- E(E, Y)
    seed: Rule = (("T", (Variable("X"), Variable("Y"))), ("E", (Variable("X"), Variable("Y"))))
    transitivity: Rule = (
    ("T", (Variable("X"), Variable("Z"))), ("E", (Variable("X"), Variable("Y"))), ("T", (Variable("Y"), Variable("Z"))))
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
    print("INPUT STREAM")
    print(edb_stream)
    print("ITERATION 1 --")
    incremental_datalog.step()
    print("ITERATION 1 - OUT")
    output_stream = incremental_datalog.output_handle().get()
    print(output_stream)
    print(stream_elimination(output_stream))

    assert True == False

def test_triangle() -> None:
    program_group: ZSetAddition[Rule] = ZSetAddition()
    program_stream = Stream(program_group)
    program_stream_h = StreamHandle(lambda: program_stream)

    # T(X, Y) <- E(E, Y)
    triangle: Rule = (
        ("T", (Variable("A"), Variable("B"), Variable("C"))), ("E", (Variable("A"), Variable("B"))),
        ("E", (Variable("B"), Variable("C"))), ("E", (Variable("C"), Variable("A"))))
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
    print("INPUT STREAM")
    print(edb_stream)
    print("ITERATION 1 --")
    incremental_datalog.step()
    print("ITERATION 1 - OUT")
    output_stream = incremental_datalog.output_handle().get()
    print(output_stream)
    print(stream_elimination(output_stream))

    assert True == False
