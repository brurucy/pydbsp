"""Incremental Datalog as a stream-to-stream circuit on a flat 2-D
product lattice.

``edb`` and ``program`` are ``Input`` streams on a single
``Time2D`` lattice (axis 0 = outer/input progression,
axis 1 = inner/fixpoint iteration). Each push at ``(t0, 0)``
contributes a batch at outer tick ``t0``; identity everywhere else.
``state`` is a 2-D ``Input[pair, (int, int)]`` — a cell per
(outer, inner) step. ``saturate`` drives inner iteration at a
chosen outer tick until ``body_out``'s inner-diff is zero past the
max prior inner depth.

The public ``observable`` is a 1-D stream (collapsed inner axis via
``TimeAxisElimination``) — reading ``observable.at((t0,))`` yields the
outer-delta of the cumulative EDB fixpoint at that tick.
"""

from dataclasses import dataclass
from typing import Any, NewType, TypeAlias, cast

from pydbsp.core import DBSPTime, ProductGroup, Time1D, Time2D
from pydbsp.evaluator import Evaluator
from pydbsp.stream import Input, Lift1, Lift2, Stream
from pydbsp.stream.functions.linear import (
    TimeAxisElimination,
    TimeAxisIntroduction,
    StreamIntroduction,
)
from pydbsp.stream.operators.linear import Delay
from pydbsp.stream.zset.operators.bilinear import DLDJoin
from pydbsp.stream.zset.operators.binary import DLDDistinct
from pydbsp.stream.zset.operators.linear import Select, Project
from pydbsp.zset import ZSet, ZSetAddition


# ---- Datalog types ----------------------------------------------------------

Constant = Any
_Variable: TypeAlias = str
Variable = NewType("Variable", _Variable)
Term = Constant | Variable

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

Time: TypeAlias = tuple[int, int]


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
        return (atom[0], tuple(inner.get(t, t) for t in atom[1]))


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
        if isinstance(term, _Variable):
            existing = bindings.get(term, _UNBOUND)
            if existing is _UNBOUND:
                bindings[term] = constant
            elif existing != constant:
                return None
        elif term != constant:
            return None
    return Rewrite(bindings)


_UNBOUND = object()  # sentinel so ``None`` stays a valid constant value


# ---- Program-derived functions ---------------------------------------------


def sig(program: Program) -> GroundingSignals:
    signals: dict[Signal, int] = {}
    for rule, weight in program.inner.items():
        if weight <= 0:
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
        if weight <= 0:
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


# ---- DatalogCircuit handle -------------------------------------------------


@dataclass
class DatalogCircuit:
    """Handle for a built Datalog circuit.

    ``edb``, ``program``: ``Input`` streams on the outer axis. Each
    push at ``(t0,)`` contributes a batch.

    ``state``: 2-D ``Input`` of (facts, rewrites) pairs. The
    ``saturate`` driver pushes per-cell deltas ``(o, k)`` as inner
    iteration progresses.

    ``observable``: 1-D stream — reading ``observable_at((t0,))``
    yields the outer-delta of the cumulative fixpoint at that tick.

    ``body_out``: the 2-D body output; ``saturate`` reads its inner
    column via ``body_at((o, k))`` to discover iteration diffs.

    ``evaluator``: single shared cache for all reads; do not touch
    directly — call ``body_at`` / ``observable_at``.
    """

    observable: Stream[EDB, tuple[int]]
    body_out: Stream[tuple[EDB, ZSet[ProvenanceIndexedRewrite]], Time]
    state: Input[tuple[EDB, ZSet[ProvenanceIndexedRewrite]], Time]
    edb: Input[EDB, tuple[int]]
    program: Input[Program, tuple[int]]
    product_group: ProductGroup[EDB, ZSet[ProvenanceIndexedRewrite]]
    lattice: DBSPTime[Time]
    evaluator: Evaluator

    def body_at(self, t: Time) -> tuple[EDB, ZSet[ProvenanceIndexedRewrite]]:
        """Returns the ``(facts, rewrites)`` diff pair at 2-D timestamp ``t``."""
        return self.evaluator.at_op(self.body_out, t)

    def observable_at(self, t: tuple[int]) -> EDB:
        """Returns the derived facts at outer timestamp ``t``."""
        return self.evaluator.at(t)


def saturate(
    circuit: DatalogCircuit,
    outer_tick: int = 0,
    *,
    max_inner: int = 1 << 16,
) -> int:
    """Drives the Datalog inner fixpoint at ``outer_tick`` to convergence.
    Returns the final inner tick.
    """
    state = circuit.state
    prior_max = -1
    for (o, k) in state._values.keys():
        if o < outer_tick and k > prior_max:
            prior_max = k

    k_final = -1
    for k in range(max_inner):
        diff_pair = circuit.body_at((outer_tick, k))
        is_zero = diff_pair[0].inner == {} and diff_pair[1].inner == {}
        state.push((outer_tick, k), diff_pair)
        if is_zero and k > prior_max:
            k_final = k
            break
    else:
        raise RuntimeError(f"iteration did not converge in {max_inner} inner ticks")
    return k_final


# ---- IncrementalDatalog (positive) -----------------------------------------


def IncrementalDatalog(parallelism: int = 1) -> DatalogCircuit:
    """Builds a positive incremental Datalog interpreter. Push ``edb``
    facts and ``program`` rules at outer ticks, call ``saturate``, read
    derived facts via ``observable_at``. For stratified negation use
    ``IncrementalDatalogWithNegation``; for faster joins use
    ``IncrementalDatalogWithIndexing``.
    """
    fact_group: ZSetAddition[Fact] = ZSetAddition()
    rule_group: ZSetAddition[Rule] = ZSetAddition()
    rewrite_group: ZSetAddition[ProvenanceIndexedRewrite] = ZSetAddition()
    dir_group: ZSetAddition[Direction] = ZSetAddition()
    signal_group: ZSetAddition[Signal] = ZSetAddition()
    gatekeep_group: ZSetAddition[AtomWithSourceRewriteAndProvenance] = ZSetAddition()
    product_group = ProductGroup(fact_group, rewrite_group)

    lattice = Time2D

    edb = Input(fact_group, Time1D)
    program = Input(rule_group, Time1D)

    edb_2d = TimeAxisIntroduction(edb, fact_group, lattice, axis=1)
    sig_stream = TimeAxisIntroduction(
        Lift1(program, sig, signal_group), signal_group, lattice, axis=1,
    )
    dir_stream = TimeAxisIntroduction(
        Lift1(program, dir, dir_group), dir_group, lattice, axis=1,
    )

    rewrites_seed = StreamIntroduction(
        ZSet({(0, _rewrite_monoid.identity()): 1}),
        rewrite_group, lattice,
    )

    state = Input(product_group, lattice)
    distinct_facts = Delay(Lift1(state, lambda p: p[0], fact_group), lattice, axis=1)
    distinct_rewrites = Delay(Lift1(state, lambda p: p[1], rewrite_group), lattice, axis=1)

    gatekeep = DLDJoin(
        distinct_rewrites, dir_stream,
        lambda l, r: l[0] == r[0],
        lambda l, r: (r[1], r[2], l[1]),
        gatekeep_group, lattice,
    )
    product = DLDJoin(
        gatekeep, distinct_facts,
        lambda l, r: l[1] is None or (l[1][0] == r[0] and unify(l[2].apply(l[1]), r) is not None),
        rewrite_product_projection,
        rewrite_group, lattice,
    )
    ground = DLDJoin(
        product, sig_stream,
        lambda l, r: l[0] == r[0],
        lambda l, r: l[1].apply(r[1]),
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


# ---- IncrementalDatalogWithNegation ----------------------------------------


def IncrementalDatalogWithNegation(parallelism: int = 1) -> DatalogCircuit:
    """Builds an incremental Datalog interpreter with stratified
    negation. Same usage as ``IncrementalDatalog``; body atoms whose
    predicate starts with ``!`` are treated as negations (e.g.
    ``!parent(X, Y)`` matches the absence of ``parent(X, Y)``).
    """
    fact_group: ZSetAddition[Fact] = ZSetAddition()
    rule_group: ZSetAddition[Rule] = ZSetAddition()
    rewrite_group: ZSetAddition[ProvenanceIndexedRewrite] = ZSetAddition()
    dir_group: ZSetAddition[Direction] = ZSetAddition()
    signal_group: ZSetAddition[Signal] = ZSetAddition()
    gatekeep_group: ZSetAddition[AtomWithSourceRewriteAndProvenance] = ZSetAddition()
    product_group = ProductGroup(fact_group, rewrite_group)

    lattice = Time2D

    edb = Input(fact_group, Time1D)
    program = Input(rule_group, Time1D)

    edb_2d = TimeAxisIntroduction(edb, fact_group, lattice, axis=1)
    sig_stream = TimeAxisIntroduction(
        Lift1(program, sig, signal_group), signal_group, lattice, axis=1,
    )
    dir_stream = TimeAxisIntroduction(
        Lift1(program, dir, dir_group), dir_group, lattice, axis=1,
    )

    rewrites_seed = StreamIntroduction(
        ZSet({(0, _rewrite_monoid.identity()): 1}),
        rewrite_group, lattice,
    )

    state = Input(product_group, lattice)
    distinct_facts = Delay(Lift1(state, lambda p: p[0], fact_group), lattice, axis=1)
    distinct_rewrites = Delay(Lift1(state, lambda p: p[1], rewrite_group), lattice, axis=1)

    gatekeep = DLDJoin(
        distinct_rewrites, dir_stream,
        lambda l, r: l[0] == r[0],
        lambda l, r: (r[1], r[2], l[1]),
        gatekeep_group, lattice,
    )

    positive_atoms = Select(gatekeep, lambda gk: gk[1] is None or ("!" not in gk[1][0]))
    negative_atoms = Select(gatekeep, lambda gk: not (gk[1] is None or ("!" not in gk[1][0])))

    product = DLDJoin(
        positive_atoms, distinct_facts,
        lambda l, r: l[1] is None or (l[1][0] == r[0] and unify(l[2].apply(l[1]), r) is not None),
        rewrite_product_projection,
        rewrite_group, lattice,
    )

    proj = Project(negative_atoms, lambda gk: (gk[0], gk[2]), rewrite_group)

    anti_product = DLDJoin(
        negative_atoms, distinct_facts,
        lambda l, r: l[1] is None or (l[1][0].strip("!") == r[0] and unify(l[2].apply(l[1]), r) is not None),
        lambda l, _: (l[0], l[2]),
        rewrite_group, lattice,
    )

    final_product = product + proj - anti_product

    ground = DLDJoin(
        final_product, sig_stream,
        lambda l, r: l[0] == r[0],
        lambda l, r: l[1].apply(r[1]),
        fact_group, lattice,
    )

    next_facts = DLDDistinct(ground + edb_2d, fact_group, lattice)
    next_rewrites = DLDDistinct(final_product + rewrites_seed, rewrite_group, lattice)

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
