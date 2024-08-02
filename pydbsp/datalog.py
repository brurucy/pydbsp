from typing import Any, Dict, List, NewType, Tuple, TypeAlias, TypeVar, cast 
from stream import Stream, StreamHandle
from stream_operators import Delay, LiftedDelay, LiftedGroupAdd, LiftedStreamElimination, LiftedStreamIntroduction
from zset_operators import DeltaLiftedDeltaLiftedDistinct, LiftedLiftedDeltaJoin, ZSetAddition
from stream_operator import BinaryOperator 
from zset import ZSet

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

        return self.inner == other.inner # type: ignore
    
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
            if not left_term in unification_result:
                substitution: Dict[Variable, Constant] = { left_term: right_constant }
                
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

# apply - rewrite x atom -> atom
# merge - rewrite x rewrite -> rewrite
# unify - atom x fact -> rewrite

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

    full_provenance = running_provenance #+ hash(rule[0]) # type: ignore
    #provenance_chain.inner[(running_provenance, full_provenance, None)] = 1

    grounding_canaries: GroundingCanaries = ZSetAddition().identity()
    grounding_canaries.inner[(full_provenance, rule[0])] = 1 # type: ignore
    
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
        grounding_stream: Stream[GroundingCanaries] = Stream(ZSetAddition())
        provenance_stream: Stream[ProvenanceChain] = Stream(ZSetAddition())
        
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

def rewrite_product_projection(l, r):
    fst = l[0]
    snd = unify((l[2]).apply(l[1]), r) if l[1] is not None else l[2]
    new_rewrite = RewriteMonoid().add(l[2], snd)
    return fst, new_rewrite

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

        rewrite_stream: Stream[ZSet[ProvenanceIndexedRewrite]] = Stream(ZSetAddition())
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
            None,
            None,
            lambda l, r: l[0] == r[0],
            lambda l, r: (r[1], r[2], l[1]))
        
        self.rewrite_product = LiftedLiftedDeltaJoin(
            self.iteration.output_handle(),
            None,
            lambda l, r: l[1] is None or (l[1][0] == r[0] and unify(l[2].apply(l[1]), r) is not None),
            rewrite_product_projection)
        
        self.fresh_facts = LiftedLiftedDeltaJoin(
            self.rewrite_product.output_handle(),
            self.lifted_intro_grounding.output_handle(),
            lambda l, r: l[0] == r[0],
            lambda l, r: l[1].apply(r[1]))
        
        self.fresh_facts_plus_edb = LiftedGroupAdd(self.fresh_facts.output_handle(), self.lifted_intro_edb.output_handle())

        self.distinct_facts = DeltaLiftedDeltaLiftedDistinct(self.fresh_facts_plus_edb.output_handle())
        self.distinct_rewrites = DeltaLiftedDeltaLiftedDistinct(self.rewrite_product.output_handle())

        self.lift_delay_distinct_facts = Delay(self.distinct_facts.output_handle())
        self.lift_delay_distinct_rewrites = Delay(self.distinct_rewrites.output_handle())
        self.rewrite_product_plus_rewrites = LiftedGroupAdd(self.lift_delay_distinct_rewrites.output_handle(), self.lifted_rewrites.output_handle())

        self.iteration.set_input_a(self.rewrite_product_plus_rewrites.output_handle())
        self.iteration.set_input_b(self.lifted_intro_provenance.output_handle())
        self.rewrite_product.set_input_b(self.lift_delay_distinct_facts.output_handle())

        self.lifted_elim_fresh_facts = LiftedStreamElimination(self.distinct_facts.output_handle())
        self.output_stream_handle = self.lifted_elim_fresh_facts.output_handle()

    def rewrite_product_projection(self, l, r):
        return lambda l, r: (l[0], unify((l[2]).apply(l[1]), r) if l[1] is not None else l[2])

    def step(self) -> bool:
        self.lifted_intro_edb.step()
        print("lifted_intro")

        self.lifted_rewrites.step()
        print("lifted_rewrites")

        self.lift_derive_program_provenance.step()

        print("intro_grounding")
        self.lifted_intro_grounding.step()

        print("intro_provenance")
        self.lifted_intro_provenance.step()

        while True:
            steps = [not b for b in [self.lift_delay_distinct_facts.step(),
                     self.lift_delay_distinct_rewrites.step(),
                     self.rewrite_product_plus_rewrites.step(),
                     self.iteration.step(),
                     self.rewrite_product.step(),
                     self.fresh_facts.step(),
                     self.fresh_facts_plus_edb.step(),
                     self.distinct_facts.step(),
                     self.distinct_rewrites.step(),
                     self.lifted_elim_fresh_facts.step()
                    ]]
            
            if any(steps):
                continue
            else:
                break

        return True
