from typing import Any, Dict, List, NewType, Tuple, TypeAlias, TypeVar, cast

from stream import Stream, StreamHandle
from stream_operator import BinaryOperator
from stream_operators import (
    Delay,
    LiftedGroupAdd,
    LiftedStreamElimination,
    LiftedStreamIntroduction,
    step_until_timestamp,
)
from zset import ZSet
from zset_operators import DeltaLiftedDeltaLiftedDistinct, LiftedLiftedDeltaJoin, ZSetAddition

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
ProvenanceDirection: TypeAlias = tuple[Provenance, Provenance, Atom | None]
ProvenanceChain: TypeAlias = ZSet[ProvenanceDirection]

Head: TypeAlias = Atom
Canary = tuple[Provenance, Head]
GroundingCanaries = ZSet[Canary]

EmptyGroundingCanaries: GroundingCanaries = ZSetAddition().identity()


def derive_rule_provenance(rule: Rule) -> tuple[GroundingCanaries, ProvenanceChain]:
    provenance_chain: ProvenanceChain = ZSetAddition().identity()
    running_provenance = 0

    if len(rule) > 1:
        for dependency in rule[1:]:
            previous_atom_provenance = running_provenance
            running_provenance += hash(dependency)

            provenance_chain.inner[(previous_atom_provenance, running_provenance, dependency)] = 1

    full_provenance = running_provenance
    grounding_canaries: GroundingCanaries = ZSetAddition().identity()
    grounding_canaries.inner[(full_provenance, rule[0])] = 1  # type: ignore

    return (grounding_canaries, provenance_chain)


def derive_program_provenance(program: Program) -> tuple[GroundingCanaries, ProvenanceChain]:
    grounding_group: ZSetAddition[Canary] = ZSetAddition()
    grounding_canaries: GroundingCanaries = ZSetAddition().identity()

    provenance_group: ZSetAddition[ProvenanceDirection] = ZSetAddition()
    provenance_chain: ProvenanceChain = ZSetAddition().identity()
    for rule in program.inner.keys():
        (canaries, chain) = derive_rule_provenance(rule)
        grounding_canaries = grounding_group.add(grounding_canaries, canaries)
        provenance_chain = provenance_group.add(provenance_chain, chain)

    return grounding_canaries, provenance_chain


class LiftedDeriveProgramProvenance:
    input_a: StreamHandle[Program]
    grounding_stream_handle: StreamHandle[GroundingCanaries]
    provenance_stream_handle: StreamHandle[ProvenanceChain]

    def __init__(self, stream_handle: StreamHandle[Program]) -> None:
        self.input_a = stream_handle
        grounding_group: ZSetAddition[Canary] = ZSetAddition()
        grounding_stream: Stream[GroundingCanaries] = Stream(grounding_group)

        provenance_group: ZSetAddition[ProvenanceDirection] = ZSetAddition()
        provenance_stream: Stream[ProvenanceChain] = Stream(provenance_group)

        self.grounding_stream_handle = StreamHandle(lambda: grounding_stream)
        self.provenance_stream_handle = StreamHandle(lambda: provenance_stream)

    def output_grounding(self) -> Stream[GroundingCanaries]:
        return self.grounding_stream_handle.get()

    def output_provenance(self) -> Stream[ProvenanceChain]:
        return self.provenance_stream_handle.get()

    def step(self) -> bool:
        output_timestamp = self.output_grounding().current_time() + 1
        latest_program_provenance = derive_program_provenance(self.input_a.get()[output_timestamp])

        self.grounding_stream_handle.get().send(latest_program_provenance[0])
        self.provenance_stream_handle.get().send(latest_program_provenance[1])

        return True


AtomSet: TypeAlias = ZSet[Atom]
ProvenanceIndexedRewrite = tuple[Provenance, Rewrite]
AtomWithSourceRewriteAndProvenance = tuple[Provenance, Atom | None, Rewrite]


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
    lift_derive_program_provenance: LiftedDeriveProgramProvenance
    grounding: StreamHandle[GroundingCanaries]
    lifted_intro_grounding: LiftedStreamIntroduction[GroundingCanaries]
    provenance: StreamHandle[ProvenanceChain]
    lifted_intro_provenance: LiftedStreamIntroduction[ProvenanceChain]

    # EDB transformations
    lifted_intro_edb: LiftedStreamIntroduction[EDB]

    # Rewrite transformations
    rewrites: StreamHandle[ZSet[ProvenanceIndexedRewrite]]
    lifted_rewrites: LiftedStreamIntroduction[ZSet[ProvenanceIndexedRewrite]]

    iteration: LiftedLiftedDeltaJoin[ProvenanceIndexedRewrite, ProvenanceDirection, AtomWithSourceRewriteAndProvenance]
    rewrite_product: LiftedLiftedDeltaJoin[AtomWithSourceRewriteAndProvenance, Fact, ProvenanceIndexedRewrite]
    fresh_facts: LiftedLiftedDeltaJoin[ProvenanceIndexedRewrite, Canary, Fact]

    distinct_facts: DeltaLiftedDeltaLiftedDistinct[Fact]
    distinct_rewrites: DeltaLiftedDeltaLiftedDistinct[ProvenanceIndexedRewrite]

    lift_delay_distinct_facts: Delay[Stream[ZSet[Fact]]]
    lift_delay_distinct_rewrites: Delay[Stream[ZSet[ProvenanceIndexedRewrite]]]

    fresh_facts_plus_edb: LiftedGroupAdd[Stream[EDB]]
    rewrite_product_plus_rewrites: LiftedGroupAdd[Stream[ZSet[ProvenanceIndexedRewrite]]]

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
        self.lift_derive_program_provenance = LiftedDeriveProgramProvenance(self.input_stream_handle_b)

        self.grounding = self.lift_derive_program_provenance.grounding_stream_handle
        self.lifted_intro_grounding = LiftedStreamIntroduction(self.grounding)

        self.provenance = self.lift_derive_program_provenance.provenance_stream_handle
        self.lifted_intro_provenance = LiftedStreamIntroduction(self.provenance)

        self.iteration = LiftedLiftedDeltaJoin(
            None, None, lambda left, right: left[0] == right[0], lambda left, right: (right[1], right[2], left[1])
        )

        self.rewrite_product = LiftedLiftedDeltaJoin(
            self.iteration.output_handle(),
            None,
            lambda left, right: left[1] is None
            or (left[1][0] == right[0] and unify(left[2].apply(left[1]), right) is not None),
            rewrite_product_projection,
        )

        self.fresh_facts = LiftedLiftedDeltaJoin(
            self.rewrite_product.output_handle(),
            self.lifted_intro_grounding.output_handle(),
            lambda left, right: left[0] == right[0],
            lambda left, right: left[1].apply(right[1]),
        )

        self.fresh_facts_plus_edb = LiftedGroupAdd(
            self.fresh_facts.output_handle(), self.lifted_intro_edb.output_handle()
        )

        self.distinct_facts = DeltaLiftedDeltaLiftedDistinct(self.fresh_facts_plus_edb.output_handle())
        self.rewrite_product_plus_rewrites = LiftedGroupAdd(
            self.rewrite_product.output_handle(), self.lifted_rewrites.output_handle()
        )
        self.distinct_rewrites = DeltaLiftedDeltaLiftedDistinct(self.rewrite_product_plus_rewrites.output_handle())

        self.lift_delay_distinct_facts = Delay(self.distinct_facts.output_handle())
        self.lift_delay_distinct_rewrites = Delay(self.distinct_rewrites.output_handle())
        self.iteration.set_input_a(self.lift_delay_distinct_rewrites.output_handle())
        self.iteration.set_input_b(self.lifted_intro_provenance.output_handle())
        self.rewrite_product.set_input_b(self.lift_delay_distinct_facts.output_handle())

        self.lifted_elim_fresh_facts = LiftedStreamElimination(self.distinct_facts.output_handle())
        self.output_stream_handle = self.lifted_elim_fresh_facts.output_handle()

    def step(self) -> bool:
        self.lifted_intro_edb.step()
        self.lifted_rewrites.step()
        self.lift_derive_program_provenance.step()
        self.lifted_intro_grounding.step()
        self.lifted_intro_provenance.step()

        while True:
            self.iteration.step()
            self.rewrite_product.step()
            self.fresh_facts.step()

            self.fresh_facts_plus_edb.step()
            self.distinct_facts.step()
            new_facts = self.distinct_facts.output().latest().latest()

            self.rewrite_product_plus_rewrites.step()
            self.distinct_rewrites.step()
            new_rewrites = self.distinct_rewrites.output().latest().latest()

            if new_facts == edb_identity and new_rewrites == provenance_indexed_rewrite_identity:
                break

            self.lift_delay_distinct_facts.step()
            self.lift_delay_distinct_rewrites.step()

        step_until_timestamp(self.lifted_elim_fresh_facts, self.lifted_elim_fresh_facts.input_a().current_time())

        return True
