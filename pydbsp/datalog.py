"""Datalog value-layer terms: facts, rules, rewrites, unification, and
the indexed-join helpers.

This module is pure data and pure functions. No Circuit / Evaluator /
Stream dependencies. It is the substrate over which the v2 circuit
operators in :mod:`pydbsp.relational_operators` and
:mod:`pydbsp.indexed_relational_operators` (and the algorithm modules
that follow them) construct their bodies.

Two layers of types stack here:

* **Plain (positive) Datalog** . ``Constant``, ``Variable``, ``Atom``,
  ``Fact``, ``Rule``, ``Program``, ``EDB``, plus the provenance /
  signal / direction types that drive the gatekeep machinery.
* **Indexed extensions** . ``ColumnReference``, ``IndexedFact``,
  ``IndexedGatekeepEntry``, ``JoinKey``, ``ExtendedDirection``, plus
  the ``ext_dir`` / ``jorder`` / ``_index_fact`` / ``_gatekeep_join_key``
  helpers that the sort-merge-join variant uses.

The :class:`Rewrite` value type and its monoid (``_rewrite_monoid``)
are shared by both layers. So is ``unify``.

Originally split across ``pydbsp.algorithms.datalog`` and
``pydbsp.algorithms.datalog_indexed`` in v1. Consolidated here for the
v2 layout, with the v1 evaluator/stream imports stripped.
"""

from typing import Any, NewType, TypeAlias, cast

from pydbsp.zset import ZSet


# ---- Datalog types ----------------------------------------------------------

Constant = Any
_Variable: TypeAlias = str
Variable = NewType("Variable", _Variable)
Term = Constant | Variable


def is_variable(term: object) -> bool:
    """Return ``True`` iff ``term`` is a Variable. Variables are strings
    beginning with ``?``. Any other value (ints, tuples, lists, strings
    without the ``?`` prefix) is a Constant. The prefix is the only
    discriminator the unifier and :meth:`Rewrite.apply` consult, so
    callers must construct Variables as ``Variable("?X")``, never as
    a bare ``"X"``."""
    return isinstance(term, str) and term.startswith("?")


Predicate: TypeAlias = str
Atom: TypeAlias = tuple[Predicate, tuple[Term, ...]]
Fact: TypeAlias = tuple[Predicate, tuple[Constant, ...]]
Rule: TypeAlias = tuple[Atom, ...]

Program: TypeAlias = ZSet[Rule]
EDB: TypeAlias = ZSet[Fact]

Provenance: TypeAlias = int
Direction: TypeAlias = tuple[Provenance, Provenance, Atom | None]
ProvenanceChain: TypeAlias = ZSet[Direction]
Signal: TypeAlias = tuple[Provenance, Atom]
GroundingSignals: TypeAlias = ZSet[Signal]

ProvenanceIndexedRewrite: TypeAlias = tuple[Provenance, "Rewrite"]
AtomWithSourceRewriteAndProvenance: TypeAlias = tuple[Provenance, Atom | None, "Rewrite"]


# ---- Rewrites ---------------------------------------------------------------


class Rewrite:
    inner: dict[Variable, Constant]
    _hash: int | None

    def __init__(self, substitutions: dict[Variable, Constant]) -> None:
        self.inner = substitutions
        self._hash = None

    def __repr__(self) -> str:
        return self.inner.__repr__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rewrite):
            return False
        return self.inner == other.inner

    def __contains__(self, variable: Variable) -> bool:
        return variable in self.inner

    def __getitem__(self, variable: Variable) -> Constant | None:
        return self.inner.get(variable)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self.inner.items()))
        return self._hash

    def apply(self, atom: Atom) -> Atom:
        inner = self.inner
        if not inner:
            return atom
        return (
            atom[0],
            tuple(inner.get(t, t) if is_variable(t) else t for t in atom[1]),
        )


class RewriteMonoid:
    def add(self, a: Rewrite, b: Rewrite) -> Rewrite:
        # ``a`` takes precedence on key collision (matches the old
        # "first seen wins" loop). ``{**b, **a}`` is one C-implemented
        # dict merge instead of two Python loops with a redundant
        # ``Variable(var)`` no-op wrapper per iteration.
        if not a.inner:
            return b
        if not b.inner:
            return a
        return Rewrite({**b.inner, **a.inner})

    def identity(self) -> Rewrite:
        return Rewrite({})


_rewrite_monoid = RewriteMonoid()


_UNBOUND = object()  # sentinel so ``None`` stays a valid constant value


def unify(atom: Atom, fact: Fact) -> Rewrite | None:
    # Build the binding dict in-place; skip the per-term
    # ``Rewrite({term: constant})`` + ``RewriteMonoid.add`` intermediates
    # the original loop allocated on every iteration.
    atom_args = atom[1]
    fact_args = fact[1]
    if len(atom_args) != len(fact_args):
        return _rewrite_monoid.identity()
    bindings: dict[Variable, Constant] = {}
    for term, constant in zip(atom_args, fact_args):
        if is_variable(term):
            existing = bindings.get(term, _UNBOUND)
            if existing is _UNBOUND:
                bindings[term] = constant
            elif existing != constant:
                return None
        elif term != constant:
            return None
    return Rewrite(bindings)


# ---- Program-derived functions ---------------------------------------------


def sig(program: Program) -> GroundingSignals:
    signals: dict[Signal, int] = {}
    for rule, weight in program.inner.items():
        if weight == 0:
            continue
        running = 0
        if len(rule) > 1:
            for dep in rule[1:]:
                running += hash(dep)
        signals[(running, rule[0])] = weight
    return ZSet(signals)


def dir(program: Program) -> ProvenanceChain:
    directions: dict[Direction, int] = {}
    for rule, weight in program.inner.items():
        if weight == 0:
            continue
        running = 0
        if len(rule) > 1:
            for dep in rule[1:]:
                prev = running
                running += hash(dep)
                directions[(prev, running, dep)] = weight
    return ZSet(directions)


def rewrite_product_projection(
    gatekeep: AtomWithSourceRewriteAndProvenance,
    fact: Fact,
) -> ProvenanceIndexedRewrite:
    provenance = gatekeep[0]
    possible_atom = gatekeep[1]
    rewrite = gatekeep[2]

    fresh = _rewrite_monoid.identity()
    if possible_atom is not None:
        fresh = cast(Rewrite, unify(rewrite.apply(possible_atom), fact))
    return provenance, _rewrite_monoid.add(rewrite, fresh)


# ---- Indexed extensions ----------------------------------------------------
#
# The sort-merge-join variant of incremental Datalog needs three
# additional pieces of bookkeeping on top of the plain Rule / Atom
# vocabulary above:
#
# * A ``ColumnReference`` per body atom — the indices of arguments that
#   were already bound by an earlier body atom in the rule. These
#   columns define the join key.
# * An ``ExtendedDirection`` chain — the standard provenance chain with
#   each body atom's ``col_ref`` attached.
# * An ``IndexedFact`` — a fact paired with its ``(pred, values_at_col_ref)``
#   join key, so both sides of the join are pre-indexed.

ColumnReference = tuple[int, ...]
ExtendedDirection = tuple[Provenance, Provenance, Atom | None, ColumnReference]
ExtendedProvenanceChain = ZSet[ExtendedDirection]

IndexedGatekeepEntry = tuple[Provenance, Atom | None, Rewrite, ColumnReference]
JoinKey = tuple[str, tuple]
IndexedFact = tuple[JoinKey, Fact]


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
        if weight == 0:
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
