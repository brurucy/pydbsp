from typing import Any, Dict, List, NewType, Optional, Set, Tuple, TypeAlias, TypeVar, cast

from pydbsp.indexed_zset.operators.linear import LiftedLiftedIndex
from pydbsp.stream import (
    BinaryOperator,
    Lift1,
    LiftedGroupAdd,
    LiftedGroupNegate,
    Stream,
    StreamHandle,
    step_until_fixpoint_and_return,
)
from pydbsp.stream.operators.linear import (
    Delay,
    LiftedStreamElimination,
    LiftedStreamIntroduction,
)
from pydbsp.zset import ZSet, ZSetAddition
from pydbsp.zset.operators.bilinear import DeltaLiftedDeltaLiftedJoin
from pydbsp.zset.operators.linear import LiftedLiftedProject, LiftedLiftedSelect
from pydbsp.zset.operators.unary import DeltaLiftedDeltaLiftedDistinct

Constant = Any
_Variable: TypeAlias = str
Variable = NewType("Variable", _Variable)
Term = Constant | Variable

Predicate = str
Atom = Tuple[Predicate, Tuple[Term, ...]]
Fact = Tuple[Predicate, Tuple[Constant, ...]]
Rule = Tuple[Atom, ...]

Program = ZSet[Rule]
EDB = ZSet[Fact]


class Rewrite:
    inner: Dict[Variable, Constant]

    def __init__(self, substitutions: Dict[Variable, Constant]) -> None:
        self.inner = substitutions

    def __repr__(self) -> str:
        return self.inner.__repr__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rewrite):
            return False

        return self.inner == other.inner  # type: ignore

    def __contains__(self, variable: Variable) -> bool:
        return self.inner.__contains__(variable)

    def __getitem__(self, variable: Variable) -> Constant | None:
        if variable not in self.inner:
            return None

        return self.inner[variable]

    def __hash__(self) -> int:
        items = sorted(list(self.inner.items()), key=lambda x: x[0])

        return hash(tuple(items))

    def apply(self, atom: Atom) -> Atom:
        terms: List[Term] = list(atom[1])

        for idx, term in enumerate(atom[1]):
            if term in self:
                terms[idx] = self[term]

        new_atom: Atom = (atom[0], tuple(terms))

        return new_atom


class RewriteMonoid:
    def add(self, a: Rewrite, b: Rewrite) -> Rewrite:
        merge = self.identity()
        for var, constant in a.inner.items():
            key = Variable(var)

            if key not in merge:
                merge.inner[key] = constant

        for var, constant in b.inner.items():
            key = Variable(var)

            if key not in merge:
                merge.inner[key] = constant

        return merge

    def identity(self) -> Rewrite:
        return Rewrite({})

    def is_associative(self, a: Rewrite, b: Rewrite, c: Rewrite) -> bool:
        return self.add(self.add(a, b), c) == self.add(a, self.add(b, c))


rewrite_monoid = RewriteMonoid()

RewriteSet = ZSet[Rewrite]


def unify(atom: Atom, fact: Fact) -> Rewrite | None:
    unification_result = rewrite_monoid.identity()

    atom_term_count = len(atom[1])
    fact_constant_count = len(fact[1])
    if atom_term_count != fact_constant_count:
        return rewrite_monoid.identity()

    for left_term, right_constant in zip(atom[1], fact[1]):
        if isinstance(left_term, _Variable):
            if left_term not in unification_result:
                substitution: Dict[Variable, Constant] = {left_term: right_constant}

                unification_result = rewrite_monoid.add(unification_result, Rewrite(substitution))
            else:
                left_variable_bound_constant = unification_result[left_term]
                cast(Constant, left_variable_bound_constant)

                if left_variable_bound_constant != right_constant:
                    return None
        else:
            if left_term != right_constant:
                return None

    return unification_result


T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")
Q = TypeVar("Q")

Provenance: TypeAlias = int
Direction: TypeAlias = tuple[Provenance, Provenance, Atom | None]
ProvenanceChain: TypeAlias = ZSet[Direction]

Head: TypeAlias = Atom
Signal = tuple[Provenance, Head]
GroundingSignals = ZSet[Signal]

EmptyGroundingCanaries: GroundingSignals = ZSetAddition().identity()


def sig(program: Program) -> GroundingSignals:
    signals: GroundingSignals = ZSetAddition().identity()

    for rule, weight in program.items():
        running_provenance = 0

        if len(rule) > 1:
            for dependency in rule[1:]:
                running_provenance += hash(dependency)

        full_provenance = running_provenance
        signals.inner[(full_provenance, rule[0])] = weight  # type: ignore

    return signals


class LiftedSig(Lift1[Program, GroundingSignals]):
    def __init__(self, stream: Optional[StreamHandle[Program]]):
        group: ZSetAddition[Signal] = ZSetAddition()

        super().__init__(stream, sig, group)


def dir(program: Program) -> ProvenanceChain:
    directions: ProvenanceChain = ZSetAddition().identity()

    for rule, weight in program.items():
        running_provenance = 0

        if len(rule) > 1:
            for dependency in rule[1:]:
                previous_atom_provenance = running_provenance
                running_provenance += hash(dependency)

                directions.inner[(previous_atom_provenance, running_provenance, dependency)] = weight

    return directions


class LiftedDir(Lift1[Program, ProvenanceChain]):
    def __init__(self, stream: Optional[StreamHandle[Program]]):
        group: ZSetAddition[Direction] = ZSetAddition()

        super().__init__(stream, dir, group)


AtomSet: TypeAlias = ZSet[Atom]
ProvenanceIndexedRewrite = Tuple[Provenance, Rewrite]
AtomWithSourceRewriteAndProvenance = Tuple[Provenance, Atom | None, Rewrite]


def rewrite_product_projection(
    atom_rewrite_provenance: AtomWithSourceRewriteAndProvenance, fact: Fact
) -> Tuple[Provenance, Rewrite]:
    provenance = atom_rewrite_provenance[0]
    possible_atom = atom_rewrite_provenance[1]
    rewrite = atom_rewrite_provenance[2]

    monoid = RewriteMonoid()
    fresh_rewrite = monoid.identity()

    if possible_atom is not None:
        fresh_rewrite = cast(Rewrite, unify(rewrite.apply(possible_atom), fact))

    new_rewrite = monoid.add(rewrite, fresh_rewrite)

    return provenance, new_rewrite


edb_identity: EDB = ZSetAddition().identity()
provenance_indexed_rewrite_identity: ProvenanceChain = ZSetAddition().identity()

class IncrementalDatalog(BinaryOperator[EDB, Program, EDB]):
    # Program transformations
    lifted_sig: LiftedSig
    lifted_intro_lifted_sig: LiftedStreamIntroduction[GroundingSignals]
    lifted_dir: LiftedDir
    lifted_intro_lifted_dir: LiftedStreamIntroduction[ProvenanceChain]

    # EDB transformations
    lifted_intro_edb: LiftedStreamIntroduction[EDB]

    # Rewrite transformations
    rewrites: StreamHandle[ZSet[ProvenanceIndexedRewrite]]
    lifted_rewrites: LiftedStreamIntroduction[ZSet[ProvenanceIndexedRewrite]]

    # Joins
    gatekeep: DeltaLiftedDeltaLiftedJoin[ProvenanceIndexedRewrite, Direction, AtomWithSourceRewriteAndProvenance]
    product: DeltaLiftedDeltaLiftedJoin[AtomWithSourceRewriteAndProvenance, Fact, ProvenanceIndexedRewrite]
    ground: DeltaLiftedDeltaLiftedJoin[ProvenanceIndexedRewrite, Signal, Fact]

    # Distincts
    distinct_rewrites: DeltaLiftedDeltaLiftedDistinct[ProvenanceIndexedRewrite]
    distinct_facts: DeltaLiftedDeltaLiftedDistinct[Fact]

    # Delays
    delay_distinct_facts: Delay[Stream[ZSet[Fact]]]
    delay_distinct_rewrites: Delay[Stream[ZSet[ProvenanceIndexedRewrite]]]

    # Pluses
    fresh_facts_plus_edb: LiftedGroupAdd[Stream[EDB]]
    rewrite_product_plus_rewrites: LiftedGroupAdd[Stream[ZSet[ProvenanceIndexedRewrite]]]

    # Stream elimination
    lifted_elim_fresh_facts: LiftedStreamElimination[EDB]

    def set_input_a(self, stream_handle_a: StreamHandle[EDB]) -> None:
        self.input_stream_handle_a = stream_handle_a
        self.lifted_intro_edb = LiftedStreamIntroduction(self.input_stream_handle_a)

        provenance_indexed_rewrite_group: ZSetAddition[ProvenanceIndexedRewrite] = ZSetAddition()
        rewrite_stream: Stream[ZSet[ProvenanceIndexedRewrite]] = Stream(provenance_indexed_rewrite_group)
        self.rewrites = StreamHandle(lambda: rewrite_stream)
        empty_rewrite_set = rewrite_stream.group().identity()
        empty_rewrite_set.inner[(0, RewriteMonoid().identity())] = 1

        self.rewrites.get().send(empty_rewrite_set)
        self.lifted_rewrites = LiftedStreamIntroduction(self.rewrites)

    def set_input_b(self, stream_handle_b: StreamHandle[Program]) -> None:
        self.input_stream_handle_b = stream_handle_b

        self.lifted_sig = LiftedSig(self.input_stream_handle_b)
        self.lifted_intro_lifted_sig = LiftedStreamIntroduction(self.lifted_sig.output_handle())

        self.lifted_dir = LiftedDir(self.input_stream_handle_b)
        self.lifted_intro_lifted_dir = LiftedStreamIntroduction(self.lifted_dir.output_handle())

        self.gatekeep = DeltaLiftedDeltaLiftedJoin(
            None, None, lambda left, right: left[0] == right[0], lambda left, right: (right[1], right[2], left[1])
        )

        self.product = DeltaLiftedDeltaLiftedJoin(
            self.gatekeep.output_handle(),
            None,
            lambda left, right: left[1] is None
            or (left[1][0] == right[0] and unify(left[2].apply(left[1]), right) is not None),
            rewrite_product_projection,
        )

        self.ground = DeltaLiftedDeltaLiftedJoin(
            self.product.output_handle(),
            self.lifted_intro_lifted_sig.output_handle(),
            lambda left, right: left[0] == right[0],
            lambda left, right: left[1].apply(right[1]),
        )

        self.fresh_facts_plus_edb = LiftedGroupAdd(self.ground.output_handle(), self.lifted_intro_edb.output_handle())
        self.distinct_facts = DeltaLiftedDeltaLiftedDistinct(self.fresh_facts_plus_edb.output_handle())

        self.rewrite_product_plus_rewrites = LiftedGroupAdd(
            self.product.output_handle(), self.lifted_rewrites.output_handle()
        )
        self.distinct_rewrites = DeltaLiftedDeltaLiftedDistinct(self.rewrite_product_plus_rewrites.output_handle())

        self.delay_distinct_facts = Delay(self.distinct_facts.output_handle())
        self.delay_distinct_rewrites = Delay(self.distinct_rewrites.output_handle())
        self.gatekeep.set_input_a(self.delay_distinct_rewrites.output_handle())
        self.gatekeep.set_input_b(self.lifted_intro_lifted_dir.output_handle())
        self.product.set_input_b(self.delay_distinct_facts.output_handle())

        self.lifted_elim_fresh_facts = LiftedStreamElimination(self.distinct_facts.output_handle())
        self.output_stream_handle = self.lifted_elim_fresh_facts.output_handle()

    def step(self) -> bool:
        self.lifted_intro_edb.step()
        self.lifted_rewrites.step()
        self.lifted_sig.step()
        self.lifted_dir.step()
        self.lifted_intro_lifted_sig.step()
        self.lifted_intro_lifted_dir.step()
        self.gatekeep.step()
        self.product.step()
        self.ground.step()
        self.fresh_facts_plus_edb.step()
        self.distinct_facts.step()
        self.rewrite_product_plus_rewrites.step()
        self.distinct_rewrites.step()
        self.delay_distinct_facts.step()
        self.delay_distinct_rewrites.step()
        self.lifted_elim_fresh_facts.step()

        new_facts_stream = self.distinct_facts.output().latest()
        new_rewrites_stream = self.distinct_rewrites.output().latest()
        return new_facts_stream.is_identity() and new_rewrites_stream.is_identity()

class IncrementalDatalogWithNegation(BinaryOperator[EDB, Program, EDB]):
    # Program transformations
    lifted_sig: LiftedSig
    lifted_intro_lifted_sig: LiftedStreamIntroduction[GroundingSignals]
    lifted_dir: LiftedDir
    lifted_intro_lifted_dir: LiftedStreamIntroduction[ProvenanceChain]

    # EDB transformations
    lifted_intro_edb: LiftedStreamIntroduction[EDB]

    # Rewrite transformations
    rewrites: StreamHandle[ZSet[ProvenanceIndexedRewrite]]
    lifted_rewrites: LiftedStreamIntroduction[ZSet[ProvenanceIndexedRewrite]]

    # Joins
    gatekeep: DeltaLiftedDeltaLiftedJoin[ProvenanceIndexedRewrite, Direction, AtomWithSourceRewriteAndProvenance]
    positive_atoms: LiftedLiftedSelect[AtomWithSourceRewriteAndProvenance]
    product: DeltaLiftedDeltaLiftedJoin[AtomWithSourceRewriteAndProvenance, Fact, ProvenanceIndexedRewrite]

    negative_atoms: LiftedLiftedSelect[AtomWithSourceRewriteAndProvenance]
    proj: LiftedLiftedProject[AtomWithSourceRewriteAndProvenance, ProvenanceIndexedRewrite]

    anti_product: DeltaLiftedDeltaLiftedJoin[AtomWithSourceRewriteAndProvenance, Fact, ProvenanceIndexedRewrite]
    negated_product: LiftedGroupNegate[Stream[ZSet[ProvenanceIndexedRewrite]]]

    final_product_0: LiftedGroupAdd[Stream[ZSet[ProvenanceIndexedRewrite]]]
    final_product: LiftedGroupAdd[Stream[ZSet[ProvenanceIndexedRewrite]]]

    ground: DeltaLiftedDeltaLiftedJoin[ProvenanceIndexedRewrite, Signal, Fact]

    # Distincts
    distinct_rewrites: DeltaLiftedDeltaLiftedDistinct[ProvenanceIndexedRewrite]
    distinct_facts: DeltaLiftedDeltaLiftedDistinct[Fact]

    # Delays
    delay_distinct_facts: Delay[Stream[ZSet[Fact]]]
    delay_distinct_rewrites: Delay[Stream[ZSet[ProvenanceIndexedRewrite]]]

    # Pluses
    fresh_facts_plus_edb: LiftedGroupAdd[Stream[EDB]]
    rewrite_product_plus_rewrites: LiftedGroupAdd[Stream[ZSet[ProvenanceIndexedRewrite]]]

    # Stream elimination
    lifted_elim_fresh_facts: LiftedStreamElimination[EDB]

    def set_input_a(self, stream_handle_a: StreamHandle[EDB]) -> None:
        self.input_stream_handle_a = stream_handle_a
        self.lifted_intro_edb = LiftedStreamIntroduction(self.input_stream_handle_a)

        provenance_indexed_rewrite_group: ZSetAddition[ProvenanceIndexedRewrite] = ZSetAddition()
        rewrite_stream: Stream[ZSet[ProvenanceIndexedRewrite]] = Stream(provenance_indexed_rewrite_group)
        self.rewrites = StreamHandle(lambda: rewrite_stream)
        empty_rewrite_set = rewrite_stream.group().identity()
        empty_rewrite_set.inner[(0, RewriteMonoid().identity())] = 1

        self.rewrites.get().send(empty_rewrite_set)
        self.lifted_rewrites = LiftedStreamIntroduction(self.rewrites)

    def set_input_b(self, stream_handle_b: StreamHandle[Program]) -> None:
        self.input_stream_handle_b = stream_handle_b

        self.lifted_sig = LiftedSig(self.input_stream_handle_b)
        self.lifted_intro_lifted_sig = LiftedStreamIntroduction(self.lifted_sig.output_handle())

        self.lifted_dir = LiftedDir(self.input_stream_handle_b)
        self.lifted_intro_lifted_dir = LiftedStreamIntroduction(self.lifted_dir.output_handle())

        self.gatekeep = DeltaLiftedDeltaLiftedJoin(
            None, None, lambda left, right: left[0] == right[0], lambda left, right: (right[1], right[2], left[1])
        )

        self.positive_atoms = LiftedLiftedSelect(self.gatekeep.output_handle(), lambda gkeep: gkeep[1] is None or ("!" not in gkeep[1][0]))

        self.product = DeltaLiftedDeltaLiftedJoin(
            self.positive_atoms.output_handle(),
            None,
            lambda left, right: left[1] is None
            or (left[1][0] == right[0] and unify(left[2].apply(left[1]), right) is not None),
            rewrite_product_projection,
        )

        self.negative_atoms = LiftedLiftedSelect(self.gatekeep.output_handle(), lambda gkeep: not (gkeep[1] is None or ("!" not in gkeep[1][0])))

        # Bypass rewrites around product so that we can apply the anti join kills by retracting anti join matches in product from the bypassed rewrites
        self.proj = LiftedLiftedProject(self.negative_atoms.output_handle(), lambda gkeep: (gkeep[0], gkeep[2]))

        self.anti_product = DeltaLiftedDeltaLiftedJoin(
            self.negative_atoms.output_handle(),
            None,
            lambda left, right: left[1] is None
            or (left[1][0].strip("!") == right[0] and unify(left[2].apply(left[1]), right) is not None),
            lambda left, _: (left[0], left[2]),
        )

        self.negated_product = LiftedGroupNegate(self.anti_product.output_handle())

        self.final_product_0 = LiftedGroupAdd(self.negated_product.output_handle(), self.proj.output_handle())
        self.final_product = LiftedGroupAdd(self.final_product_0.output_handle(), self.product.output_handle())

        self.ground = DeltaLiftedDeltaLiftedJoin(
            self.final_product.output_handle(),
            self.lifted_intro_lifted_sig.output_handle(),
            lambda left, right: left[0] == right[0],
            lambda left, right: left[1].apply(right[1]),
        )

        self.fresh_facts_plus_edb = LiftedGroupAdd(self.ground.output_handle(), self.lifted_intro_edb.output_handle())
        self.distinct_facts = DeltaLiftedDeltaLiftedDistinct(self.fresh_facts_plus_edb.output_handle())

        self.rewrite_product_plus_rewrites = LiftedGroupAdd(
            self.final_product.output_handle(), self.lifted_rewrites.output_handle()
        )
        self.distinct_rewrites = DeltaLiftedDeltaLiftedDistinct(self.rewrite_product_plus_rewrites.output_handle())

        self.delay_distinct_facts = Delay(self.distinct_facts.output_handle())
        self.delay_distinct_rewrites = Delay(self.distinct_rewrites.output_handle())
        self.gatekeep.set_input_a(self.delay_distinct_rewrites.output_handle())
        self.gatekeep.set_input_b(self.lifted_intro_lifted_dir.output_handle())
        self.product.set_input_b(self.delay_distinct_facts.output_handle())
        self.anti_product.set_input_b(self.delay_distinct_facts.output_handle())

        self.lifted_elim_fresh_facts = LiftedStreamElimination(self.distinct_facts.output_handle())
        self.output_stream_handle = self.lifted_elim_fresh_facts.output_handle()

    def step(self) -> bool:
        self.lifted_intro_edb.step()
        self.lifted_rewrites.step()
        self.lifted_sig.step()
        self.lifted_dir.step()
        self.lifted_intro_lifted_sig.step()
        self.lifted_intro_lifted_dir.step()
        self.gatekeep.step()
        self.positive_atoms.step()
        self.product.step()
        self.negative_atoms.step()
        self.proj.step()
        self.anti_product.step()
        self.negated_product.step()
        self.final_product_0.step()
        self.final_product.step()
        self.ground.step()
        self.fresh_facts_plus_edb.step()
        self.distinct_facts.step()
        self.rewrite_product_plus_rewrites.step()
        self.distinct_rewrites.step()
        self.delay_distinct_facts.step()
        self.delay_distinct_rewrites.step()
        self.lifted_elim_fresh_facts.step()

        new_facts_stream = self.distinct_facts.output().latest()
        new_rewrites_stream = self.distinct_rewrites.output().latest()
        return new_facts_stream.is_identity() and new_rewrites_stream.is_identity()


ColumnReference = Tuple[int, ...]
JoinIndex = Tuple[Predicate, ColumnReference]
NaiveJoinOrder = ZSet[JoinIndex]


def compute_rule_index(rule: Rule) -> NaiveJoinOrder:
    group: ZSetAddition[JoinIndex] = ZSetAddition()
    column_index = group.identity()

    variables: Set[Variable] = set()
    fresh_variables: Set[Variable] = set()

    for body_atom in rule[1:]:
        predicate = body_atom[0]
        columns: List[int] = []

        for idx, term in enumerate(body_atom[1]):
            if isinstance(term, _Variable) and term not in variables:
                fresh_variables.add(term)

                continue

            columns.append(idx)

        column_index.inner[(predicate, tuple(columns))] = 1
        for variable in fresh_variables:
            variables.add(variable)

        fresh_variables.clear()

    return column_index


def jorder(program: Program) -> NaiveJoinOrder:
    group: ZSetAddition[JoinIndex] = ZSetAddition()
    column_index = group.identity()

    for rule, _weight in program.items():
        rule_index = compute_rule_index(rule)

        for index_schema, _weight in rule_index.items():
            column_index.inner[index_schema] = 1

    return column_index


class LiftedJorder(Lift1[Program, NaiveJoinOrder]):
    def __init__(self, stream: Optional[StreamHandle[Program]]):
        super().__init__(stream, lambda p: jorder(p), None)


class LiftedLiftedJorder(Lift1[Stream[Program], Stream[NaiveJoinOrder]]):
    def __init__(
        self,
        stream: Optional[StreamHandle[Stream[Program]]],
    ):
        super().__init__(
            stream, lambda sp: step_until_fixpoint_and_return(LiftedJorder(StreamHandle(lambda: sp))), None
        )


IndexedFact = Tuple[Fact, Fact]


def index_fact(column_reference: ColumnReference, fact: Fact) -> IndexedFact:
    indexed_fact_terms: List[int] = []
    for column in column_reference:
        indexed_fact_terms.append(fact[1][column])

    frozen_indexed_fact_terms = tuple(indexed_fact_terms)
    if len(frozen_indexed_fact_terms) == 0:
        frozen_indexed_fact_terms = fact[1]

    return ((fact[0], frozen_indexed_fact_terms), fact)


ConstantTerms = Tuple[Constant, ...]


def get_constant_terms(atom: Atom) -> ConstantTerms:
    constant_terms: List[Constant] = []
    for term in atom[1]:
        if not isinstance(term, _Variable):
            constant_terms.append(term)

    return tuple(constant_terms)


ExtendedProvenanceDirection: TypeAlias = tuple[Provenance, Provenance, Atom | None, ColumnReference]
ExtendedProvenanceChain: TypeAlias = ZSet[ExtendedProvenanceDirection]

ColumnReferenceSequence = List[ColumnReference]


AtomWithSourceRewriteAndExtendedProvenance = Tuple[Provenance, Atom | None, Rewrite, ColumnReference]


class IncrementalDatalogWithIndexing(BinaryOperator[EDB, Program, EDB]):
    # Program transformations
    lifted_sig: LiftedSig
    lifted_intro_lifted_sig: LiftedStreamIntroduction[GroundingSignals]
    lifted_dir: LiftedDir
    lifted_intro_lifted_dir: LiftedStreamIntroduction[ProvenanceChain]
    lifted_jorder: LiftedJorder
    lifted_intro_lifted_jorder: LiftedStreamIntroduction[NaiveJoinOrder]

    # EDB transformations
    lift_intro_edb: LiftedStreamIntroduction[EDB]

    # Rewrite transformations
    rewrites: StreamHandle[ZSet[ProvenanceIndexedRewrite]]
    lift_rewrites: LiftedStreamIntroduction[ZSet[ProvenanceIndexedRewrite]]

    # Joins
    gatekeep: DeltaLiftedDeltaLiftedJoin[ProvenanceIndexedRewrite, Direction, AtomWithSourceRewriteAndProvenance]
    product: DeltaLiftedDeltaLiftedJoin[AtomWithSourceRewriteAndProvenance, IndexedFact, ProvenanceIndexedRewrite]
    ground: DeltaLiftedDeltaLiftedJoin[ProvenanceIndexedRewrite, Signal, Fact]
    index: DeltaLiftedDeltaLiftedJoin[Fact, JoinIndex, IndexedFact]

    # Distincts
    distinct_rewrites: DeltaLiftedDeltaLiftedDistinct[ProvenanceIndexedRewrite]
    distinct_facts: DeltaLiftedDeltaLiftedDistinct[Fact]
    distinct_indexed_facts: DeltaLiftedDeltaLiftedDistinct[IndexedFact]

    # Delays
    delay_distinct_indexed_facts: Delay[Stream[ZSet[IndexedFact]]]
    delay_distinct_rewrites: Delay[Stream[ZSet[ProvenanceIndexedRewrite]]]

    # Pluses
    fresh_facts_plus_edb: LiftedGroupAdd[Stream[EDB]]
    rewrite_product_plus_rewrites: LiftedGroupAdd[Stream[ZSet[ProvenanceIndexedRewrite]]]

    # Stream elimination
    lift_elim_fresh_facts: LiftedStreamElimination[EDB]

    def set_input_a(self, stream_handle_a: StreamHandle[EDB]) -> None:
        self.input_stream_handle_a = stream_handle_a
        self.lift_intro_edb = LiftedStreamIntroduction(self.input_stream_handle_a)

        provenance_indexed_rewrite_group: ZSetAddition[ProvenanceIndexedRewrite] = ZSetAddition()
        rewrite_stream: Stream[ZSet[ProvenanceIndexedRewrite]] = Stream(provenance_indexed_rewrite_group)
        self.rewrites = StreamHandle(lambda: rewrite_stream)
        empty_rewrite_set = rewrite_stream.group().identity()
        empty_rewrite_set.inner[(0, RewriteMonoid().identity())] = 1

        self.rewrites.get().send(empty_rewrite_set)
        self.lift_rewrites = LiftedStreamIntroduction(self.rewrites)

    def set_input_b(self, stream_handle_b: StreamHandle[Program]) -> None:
        self.input_stream_handle_b = stream_handle_b

        self.lifted_sig = LiftedSig(self.input_stream_handle_b)
        self.lifted_intro_lifted_sig = LiftedStreamIntroduction(self.lifted_sig.output_handle())

        self.lifted_dir = LiftedDir(self.input_stream_handle_b)
        self.lifted_intro_lifted_dir = LiftedStreamIntroduction(self.lifted_dir.output_handle())

        self.lifted_jorder = LiftedJorder(self.input_stream_handle_b)
        self.lifted_intro_lifted_jorder = LiftedStreamIntroduction(self.lifted_jorder.output_handle())

        self.gatekeep = DeltaLiftedDeltaLiftedJoin(
            None,
            None,
            lambda left, right: left[0] == right[0],
            lambda left, right: (right[1], right[2], left[1]),
        )
        self.indexed_gatekeep = LiftedLiftedIndex(self.gatekeep.output_handle(), lambda s: s[0])

        self.product = DeltaLiftedDeltaLiftedJoin(
            self.gatekeep.output_handle(),
            None,
            lambda left, right: left[1] is None
            or (left[1][0] == right[1][0] and left[2] == RewriteMonoid().identity())
            or (left[1][0] == right[1][0] and (get_constant_terms(left[2].apply(left[1]))) == right[0][1]),
            lambda left, right: rewrite_product_projection((left[0], left[1], left[2]), right[1]),
        )

        self.ground = DeltaLiftedDeltaLiftedJoin(
            self.product.output_handle(),
            self.lifted_intro_lifted_sig.output_handle(),
            lambda left, right: left[0] == right[0],
            lambda left, right: left[1].apply(right[1]),
        )

        self.fresh_facts_plus_edb = LiftedGroupAdd(self.ground.output_handle(), self.lift_intro_edb.output_handle())
        self.distinct_facts = DeltaLiftedDeltaLiftedDistinct(self.fresh_facts_plus_edb.output_handle())
        self.index = DeltaLiftedDeltaLiftedJoin(
            self.distinct_facts.output_handle(),
            self.lifted_intro_lifted_jorder.output_handle(),
            lambda left, right: left[0] == right[0],
            lambda left, right: index_fact(right[1], left),
        )
        self.distinct_indexed_facts = DeltaLiftedDeltaLiftedDistinct(self.index.output_handle())

        self.rewrite_product_plus_rewrites = LiftedGroupAdd(
            self.product.output_handle(), self.lift_rewrites.output_handle()
        )
        self.distinct_rewrites = DeltaLiftedDeltaLiftedDistinct(self.rewrite_product_plus_rewrites.output_handle())

        self.delay_distinct_rewrites = Delay(self.distinct_rewrites.output_handle())
        self.delay_distinct_indexed_facts = Delay(self.distinct_indexed_facts.output_handle())

        self.gatekeep.set_input_a(self.delay_distinct_rewrites.output_handle())
        self.gatekeep.set_input_b(self.lifted_intro_lifted_dir.output_handle())
        self.product.set_input_b(self.delay_distinct_indexed_facts.output_handle())

        self.lift_elim_fresh_facts = LiftedStreamElimination(self.distinct_facts.output_handle())
        self.output_stream_handle = self.lift_elim_fresh_facts.output_handle()

    def step(self) -> bool:
        self.lift_intro_edb.step()
        self.lift_rewrites.step()
        self.lifted_sig.step()
        self.lifted_dir.step()
        self.lifted_jorder.step()
        self.lifted_intro_lifted_sig.step()
        self.lifted_intro_lifted_dir.step()
        self.lifted_intro_lifted_jorder.step()
        self.gatekeep.step()
        self.product.step()
        self.ground.step()
        self.fresh_facts_plus_edb.step()
        self.distinct_facts.step()
        self.index.step()
        self.distinct_indexed_facts.step()
        self.rewrite_product_plus_rewrites.step()
        self.distinct_rewrites.step()
        self.delay_distinct_indexed_facts.step()
        self.delay_distinct_rewrites.step()
        self.lift_elim_fresh_facts.step()

        new_facts_stream = self.distinct_facts.output().latest()
        new_rewrites_stream = self.distinct_rewrites.output().latest()
        return new_facts_stream.is_identity() and new_rewrites_stream.is_identity()
