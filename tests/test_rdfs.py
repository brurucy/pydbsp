"""Integration tests for :mod:`pydbsp.rdfs`.

Each test wires the indexed RDFS body, pushes a small TBox + ABox,
drives the inner fixpoint via ``e.saturate_inner``, and checks the
accumulated derivation matches the expected RDFS closure.
"""

from __future__ import annotations

from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.core import Antichain, dbsp_time
from pydbsp.evaluate import Evaluator
from pydbsp.operator import Input
from pydbsp.progress import Time
from pydbsp.rdfs import (
    DOMAIN,
    RANGE,
    SCO,
    SPO,
    TYPE,
    IndexedIncrementalRDFSBody,
    RDFTuple,
)
from pydbsp.storage import DictStorage
from pydbsp.zset import ZSet, ZSetAddition


def _build():
    tg: ZSetAddition[RDFTuple] = ZSetAddition()
    e = Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(2)),
        group=tg,
    )
    abox_1d = Input[ZSet[RDFTuple]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    tbox_1d = Input[ZSet[RDFTuple]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    state_2d = Input[ZSet[RDFTuple]](frontier=Antichain(dbsp_time(2))).connect(e.circuit, ())
    body = IndexedIncrementalRDFSBody(tuple_group=tg).connect(
        e.circuit,
        (abox_1d, tbox_1d, state_2d),
    )
    return e, abox_1d, tbox_1d, state_2d, body, tg


def _saturate(
    e,
    abox_1d,
    tbox_1d,
    state_2d,
    body,
    tg,
    abox: ZSet[RDFTuple],
    tbox: ZSet[RDFTuple],
) -> ZSet[RDFTuple]:
    e.push(abox_1d, abox)
    e.push(tbox_1d, tbox)
    cumulative = tg.identity()
    for k, diff in e.saturate_inner(body, 0, is_empty=lambda d: not d.inner):
        cumulative = tg.add(cumulative, diff)
        e.push(state_2d, diff, t=(0, k))
    return cumulative


def test_sco_chain_propagates_type() -> None:
    """Cat ⊑ Mammal ⊑ Animal; Mittens : Cat.

    Expected closure pulls in the SCO TC ``(Cat, SCO, Animal)`` and the
    Class-TC derivations ``(Mittens, TYPE, Mammal)`` and ``(Mittens,
    TYPE, Animal)``."""
    e, abox_1d, tbox_1d, state_2d, body, tg = _build()
    tbox = ZSet(
        {
            ("Cat", SCO, "Mammal"): 1,
            ("Mammal", SCO, "Animal"): 1,
        }
    )
    abox = ZSet({("Mittens", TYPE, "Cat"): 1})

    closure = _saturate(
        e,
        abox_1d,
        tbox_1d,
        state_2d,
        body,
        tg,
        abox,
        tbox,
    )

    expected = ZSet(
        {
            ("Cat", SCO, "Mammal"): 1,
            ("Mammal", SCO, "Animal"): 1,
            ("Cat", SCO, "Animal"): 1,  # SCO TC
            ("Mittens", TYPE, "Cat"): 1,
            ("Mittens", TYPE, "Mammal"): 1,  # Class TC (Cat ⊑ Mammal)
            ("Mittens", TYPE, "Animal"): 1,  # Class TC (Cat ⊑ Animal)
        }
    )
    assert closure == expected


def test_spo_propagates_through_property_rule() -> None:
    """parentOf ⊑ ancestorOf in the TBox; an ABox parentOf fact yields
    the corresponding ancestorOf fact via the Property rule."""
    e, abox_1d, tbox_1d, state_2d, body, tg = _build()
    tbox = ZSet({("parentOf", SPO, "ancestorOf"): 1})
    abox = ZSet({("alice", "parentOf", "bob"): 1})

    closure = _saturate(
        e,
        abox_1d,
        tbox_1d,
        state_2d,
        body,
        tg,
        abox,
        tbox,
    )

    expected = ZSet(
        {
            ("parentOf", SPO, "ancestorOf"): 1,
            ("alice", "parentOf", "bob"): 1,
            ("alice", "ancestorOf", "bob"): 1,
        }
    )
    assert closure == expected


def test_domain_rule_assigns_type_to_subject() -> None:
    """``parentOf`` has domain ``Person``; ``(alice, parentOf, bob)``
    triggers the Domain rule → ``(alice, TYPE, Person)``."""
    e, abox_1d, tbox_1d, state_2d, body, tg = _build()
    tbox = ZSet({("parentOf", DOMAIN, "Person"): 1})
    abox = ZSet({("alice", "parentOf", "bob"): 1})

    closure = _saturate(
        e,
        abox_1d,
        tbox_1d,
        state_2d,
        body,
        tg,
        abox,
        tbox,
    )

    expected = ZSet(
        {
            ("parentOf", DOMAIN, "Person"): 1,
            ("alice", "parentOf", "bob"): 1,
            ("alice", TYPE, "Person"): 1,
        }
    )
    assert closure == expected


def test_range_rule_assigns_type_to_object() -> None:
    """``parentOf`` has range ``Person``; ``(alice, parentOf, bob)``
    triggers the Range rule → ``(bob, TYPE, Person)``."""
    e, abox_1d, tbox_1d, state_2d, body, tg = _build()
    tbox = ZSet({("parentOf", RANGE, "Person"): 1})
    abox = ZSet({("alice", "parentOf", "bob"): 1})

    closure = _saturate(
        e,
        abox_1d,
        tbox_1d,
        state_2d,
        body,
        tg,
        abox,
        tbox,
    )

    expected = ZSet(
        {
            ("parentOf", RANGE, "Person"): 1,
            ("alice", "parentOf", "bob"): 1,
            ("bob", TYPE, "Person"): 1,
        }
    )
    assert closure == expected


def test_spo_transitive_closure() -> None:
    """Three-element SPO chain — ``a ⊑ b ⊑ c`` produces the TC pair
    ``(a, SPO, c)``, then ``(alice, c, bob)`` follows from
    ``(alice, a, bob)`` via two Property rule applications."""
    e, abox_1d, tbox_1d, state_2d, body, tg = _build()
    tbox = ZSet(
        {
            ("a", SPO, "b"): 1,
            ("b", SPO, "c"): 1,
        }
    )
    abox = ZSet({("alice", "a", "bob"): 1})

    closure = _saturate(
        e,
        abox_1d,
        tbox_1d,
        state_2d,
        body,
        tg,
        abox,
        tbox,
    )

    expected = ZSet(
        {
            ("a", SPO, "b"): 1,
            ("b", SPO, "c"): 1,
            ("a", SPO, "c"): 1,  # SPO TC
            ("alice", "a", "bob"): 1,
            ("alice", "b", "bob"): 1,  # Property via a ⊑ b
            ("alice", "c", "bob"): 1,  # Property via a ⊑ c
        }
    )
    assert closure == expected


def test_domain_composes_with_class_tc() -> None:
    """Combine domain + SCO: ``parentOf`` has domain ``Parent``;
    ``Parent ⊑ Person``. A parentOf fact yields TYPE ``Parent`` (domain
    rule) AND TYPE ``Person`` (class TC via SCO)."""
    e, abox_1d, tbox_1d, state_2d, body, tg = _build()
    tbox = ZSet(
        {
            ("parentOf", DOMAIN, "Parent"): 1,
            ("Parent", SCO, "Person"): 1,
        }
    )
    abox = ZSet({("alice", "parentOf", "bob"): 1})

    closure = _saturate(
        e,
        abox_1d,
        tbox_1d,
        state_2d,
        body,
        tg,
        abox,
        tbox,
    )

    expected = ZSet(
        {
            ("parentOf", DOMAIN, "Parent"): 1,
            ("Parent", SCO, "Person"): 1,
            ("alice", "parentOf", "bob"): 1,
            ("alice", TYPE, "Parent"): 1,  # Domain rule
            ("alice", TYPE, "Person"): 1,  # Class TC composes
        }
    )
    assert closure == expected
