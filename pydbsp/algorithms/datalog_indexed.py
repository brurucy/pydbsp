"""Indexed-join variant of ``IncrementalDatalog`` (positive only),
ported to the flat 2-D product lattice.

Same topology as ``datalog.py``; the difference is the product join
— gatekeep entries carry a ``col_ref`` (bound-column indices from
the rule's join order), and facts are indexed under
``(pred, extracted_values_at_col_ref)``. Both sides go through
``Index`` and ``DLDSortMergeJoin`` so matching pairs are found via
sort-merge instead of all-pairs filtering.

Negation is omitted here — the structure is mechanical to add.
"""

from typing import cast

from pydbsp.core import ProductGroup, Time1D, Time2D
from pydbsp.indexed_zset.operators.bilinear import DLDSortMergeJoin
from pydbsp.indexed_zset.operators.linear import Index
from pydbsp.evaluator import Evaluator
from pydbsp.stream import Input, Lift1, Lift2
from pydbsp.stream.functions.linear import (
    TimeAxisElimination,
    TimeAxisIntroduction,
    StreamIntroduction,
)
from pydbsp.stream.operators.linear import Delay
from pydbsp.stream.zset.operators.binary import DLDDistinct
from pydbsp.zset import ZSet, ZSetAddition

from pydbsp.algorithms.datalog import (
    EDB,
    Atom,
    DatalogCircuit,
    Fact,
    Program,
    Provenance,
    ProvenanceIndexedRewrite,
    Rewrite,
    Rule,
    Signal,
    Variable,
    _Variable,
    _rewrite_monoid,
    sig,
    unify,
)


# ---- Indexed types ---------------------------------------------------------


ColumnReference = tuple[int, ...]
ExtendedDirection = tuple[Provenance, Provenance, Atom | None, ColumnReference]
ExtendedProvenanceChain = ZSet[ExtendedDirection]

IndexedGatekeepEntry = tuple[Provenance, Atom | None, Rewrite, ColumnReference]
JoinKey = tuple[str, tuple]
IndexedFact = tuple[JoinKey, Fact]

Time = tuple[int, int]


# ---- Program-derived functions --------------------------------------------


def _rule_col_refs(rule: Rule) -> list[tuple[Atom, ColumnReference]]:
    variables: set[Variable] = set()
    out: list[tuple[Atom, ColumnReference]] = []
    for body_atom in rule[1:]:
        cols: list[int] = []
        fresh: set[Variable] = set()
        for idx, term in enumerate(body_atom[1]):
            if isinstance(term, _Variable) and term not in variables:
                fresh.add(cast(Variable, term))
            else:
                cols.append(idx)
        out.append((body_atom, tuple(cols)))
        variables |= fresh
    return out


def ext_dir(program: Program) -> ExtendedProvenanceChain:
    """Direction chain with per-body-atom ``col_ref`` attached."""
    entries: dict[ExtendedDirection, int] = {}
    for rule, weight in program.inner.items():
        if weight <= 0:
            continue
        running = 0
        for body_atom, col_ref in _rule_col_refs(rule):
            prev = running
            running += hash(body_atom)
            entries[(prev, running, body_atom, col_ref)] = weight
    return ZSet(entries)


def jorder(program: Program) -> ZSet[tuple[str, ColumnReference]]:
    """Unique ``(pred, col_ref)`` pairs across all rule body atoms."""
    entries: dict[tuple[str, ColumnReference], int] = {}
    for rule, weight in program.inner.items():
        if weight <= 0:
            continue
        for body_atom, col_ref in _rule_col_refs(rule):
            entries[(body_atom[0].lstrip("!"), col_ref)] = 1
    return ZSet(entries)


def _index_fact(col_ref: ColumnReference, fact: Fact) -> IndexedFact:
    values = tuple(fact[1][i] for i in col_ref)
    return ((fact[0], values), fact)


def _gatekeep_join_key(gk: IndexedGatekeepEntry) -> JoinKey:
    _prov, atom, rewrite, col_ref = gk
    if atom is None:
        return ("", ())
    applied = rewrite.apply(atom)
    return (applied[0].lstrip("!"), tuple(applied[1][i] for i in col_ref))


def _indexed_product_proj(
    _key: JoinKey,
    gk: IndexedGatekeepEntry,
    ifact: IndexedFact,
) -> ProvenanceIndexedRewrite | None:
    prov, atom, rewrite, _col_ref = gk
    fact = ifact[1]
    if atom is None:
        return prov, rewrite
    fresh = unify(rewrite.apply(atom), fact)
    if fresh is None:
        return None
    return prov, _rewrite_monoid.add(rewrite, fresh)


# ---- Circuit ---------------------------------------------------------------


def IncrementalDatalogWithIndexing(parallelism: int = 1) -> DatalogCircuit:
    """Builds an incremental Datalog interpreter with sort-merge-join
    indexing. Same usage as ``IncrementalDatalog``; faster on programs
    with many joins on repeated columns.
    """
    fact_group: ZSetAddition[Fact] = ZSetAddition()
    rule_group: ZSetAddition[Rule] = ZSetAddition()
    rewrite_group: ZSetAddition[ProvenanceIndexedRewrite] = ZSetAddition()
    ext_dir_group: ZSetAddition[ExtendedDirection] = ZSetAddition()
    jorder_group: ZSetAddition[tuple[str, ColumnReference]] = ZSetAddition()
    signal_group: ZSetAddition[Signal] = ZSetAddition()
    gatekeep_group: ZSetAddition[IndexedGatekeepEntry] = ZSetAddition()
    indexed_fact_group: ZSetAddition[IndexedFact] = ZSetAddition()
    product_group: ProductGroup[EDB, ZSet[ProvenanceIndexedRewrite]] = ProductGroup(
        fact_group, rewrite_group
    )

    lattice = Time2D

    edb = Input(fact_group, Time1D)
    program = Input(rule_group, Time1D)

    edb_2d = TimeAxisIntroduction(edb, fact_group, lattice, axis=1)
    sig_stream = TimeAxisIntroduction(
        Lift1(program, sig, signal_group), signal_group, lattice, axis=1,
    )
    ext_dir_stream = TimeAxisIntroduction(
        Lift1(program, ext_dir, ext_dir_group), ext_dir_group, lattice, axis=1,
    )
    jorder_stream = TimeAxisIntroduction(
        Lift1(program, jorder, jorder_group), jorder_group, lattice, axis=1,
    )

    rewrites_seed = StreamIntroduction(
        ZSet({(0, _rewrite_monoid.identity()): 1}),
        rewrite_group, lattice,
    )

    state = Input(product_group, lattice)
    distinct_facts = Delay(Lift1(state, lambda p: p[0], fact_group), lattice, axis=1)
    distinct_rewrites = Delay(Lift1(state, lambda p: p[1], rewrite_group), lattice, axis=1)

    # gatekeep: sort-merge on the provenance id (first field of both
    # sides). Rule counts > ~10 make provenance fan-out large enough
    # that nested-loop's O(|A|×|B|) dominates — sort-merge is O(|A|+|B|).
    gatekeep = DLDSortMergeJoin(
        Index(distinct_rewrites, lambda rw: rw[0], rewrite_group),
        Index(ext_dir_stream, lambda ed: ed[0], ext_dir_group),
        lambda _k, l, r: (r[1], r[2], l[1], r[3]),
        gatekeep_group, lattice,
    )

    # facts × jorder on predicate — sort-merge.
    indexed_facts_raw = DLDSortMergeJoin(
        Index(distinct_facts, lambda f: f[0], fact_group),
        Index(jorder_stream, lambda j: j[0], jorder_group),
        lambda _k, fact, jo: _index_fact(jo[1], fact),
        indexed_fact_group, lattice,
    )
    distinct_indexed_facts = DLDDistinct(indexed_facts_raw, indexed_fact_group, lattice)

    # Product: gatekeep ⋈ indexed-facts on (pred, extracted values) — sort-merge.
    product = DLDSortMergeJoin(
        Index(gatekeep, _gatekeep_join_key, gatekeep_group),
        Index(distinct_indexed_facts, lambda ifact: ifact[0], indexed_fact_group),
        _indexed_product_proj,
        rewrite_group, lattice,
    )

    # ground: sort-merge on the provenance id joining a rewrite with
    # the signal (head atom pattern) of the rule it satisfies.
    ground = DLDSortMergeJoin(
        Index(product, lambda r: r[0], rewrite_group),
        Index(sig_stream, lambda s: s[0], signal_group),
        lambda _k, l, r: l[1].apply(r[1]),
        fact_group, lattice,
    )

    next_facts = DLDDistinct(ground + edb_2d, fact_group, lattice)
    next_rewrites = DLDDistinct(product + rewrites_seed, rewrite_group, lattice)

    body_out = Lift2(next_facts, next_rewrites, lambda a, b: (a, b), product_group)
    observable = TimeAxisElimination(
        Lift1(body_out, lambda p: p[0], fact_group),
    )

    evaluator = Evaluator(observable, parallelism=parallelism)
    setattr(observable, "_evaluator", evaluator)
    return DatalogCircuit(
        observable=observable,
        body_out=body_out,
        evaluator=evaluator,
        state=state,
        edb=edb,
        program=program,
        product_group=product_group,
        lattice=lattice,
    )


__all__ = ["IncrementalDatalogWithIndexing"]
