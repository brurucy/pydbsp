from pydbsp import (
    DOMAIN,
    TYPE,
    Datalog,
    RDFS,
    Reachability,
    StratifiedDatalog,
    V,
    atom,
    fact,
    facts,
    rule,
    rules,
)
from pydbsp.algorithms import (
    IncrementalDatalogWithIndexing,
    IncrementalReachabilityWithIndexing,
)


def _tc_program():
    x = V("X")
    y = V("Y")
    z = V("Z")
    return rules(
        rule(atom("tc", x, y), atom("e", x, y)),
        rule(atom("tc", x, z), atom("tc", x, y), atom("e", y, z)),
    )


def test_indexed_variants_are_exported_from_algorithms() -> None:
    assert IncrementalDatalogWithIndexing.__name__ == "IncrementalDatalogWithIndexing"
    assert IncrementalReachabilityWithIndexing.__name__ == "IncrementalReachabilityWithIndexing"


def test_datalog_facade_owns_ticks_and_materialization() -> None:
    db = Datalog(indexed=True)
    db.step(
        facts=facts(
            fact("e", 0, 1),
            fact("e", 1, 2),
            fact("e", 2, 3),
        ),
        program=_tc_program(),
    )

    assert db.tick == 1
    assert db.relation("tc").inner == {
        (0, 1): 1,
        (0, 2): 1,
        (0, 3): 1,
        (1, 2): 1,
        (1, 3): 1,
        (2, 3): 1,
    }


def test_stratified_datalog_facade_uses_same_user_shape() -> None:
    db = StratifiedDatalog()
    db.step(
        facts=facts(
            fact("e", 0, 1),
            fact("e", 1, 2),
            fact("e", 2, 3),
        ),
        program=_tc_program(),
    )

    assert db.relation("tc").inner[(0, 3)] == 1
    assert db.materialized().inner[("tc", (1, 3))] == 1


def test_reachability_facade_materializes_incrementally() -> None:
    reach = Reachability(indexed=True)
    reach.step([(0, 1), (1, 2)])

    assert reach.materialized().inner == {
        (0, 1): 1,
        (0, 2): 1,
        (1, 2): 1,
    }


def test_rdfs_facade_materializes_domain_rule() -> None:
    rdfs = RDFS(indexed=True)
    rdfs.step(
        abox=[("alice", "parent", "bob")],
        tbox=[("parent", DOMAIN, "Person")],
    )

    assert rdfs.materialized().inner[("alice", TYPE, "Person")] == 1
