"""User-facing algorithm wrappers and Datalog construction helpers.

The low-level circuit API stays available for research work. This module
provides the small stateful layer most callers want: own the outer tick,
push deltas, run the right saturation driver, and keep cumulative output.
"""

from collections.abc import Iterable, Mapping
from typing import TypeVar

from pydbsp.algorithms.datalog import (
    Constant,
    EDB,
    Atom,
    DatalogCircuit,
    Fact,
    IncrementalDatalog,
    IncrementalDatalogWithNegation,
    Program,
    Rule,
    Term,
    Variable,
    saturate,
)
from pydbsp.algorithms.datalog_indexed import IncrementalDatalogWithIndexing
from pydbsp.algorithms.datalog_stratified import (
    DynamicStratifiedCircuit,
    IncrementalDatalogStratified,
    saturate_stratified,
)
from pydbsp.algorithms.rdfs import (
    RDFGraph,
    RDFSCircuit,
    RDFTuple,
    IncrementalRDFSMaterialization,
    saturate_rdfs,
)
from pydbsp.algorithms.rdfs_indexed import IncrementalRDFSMaterializationWithIndexing
from pydbsp.algorithms.reachability import (
    Edge,
    ReachabilityCircuit,
    IncrementalReachability,
    saturate_reach,
)
from pydbsp.algorithms.reachability_indexed import IncrementalReachabilityWithIndexing
from pydbsp.zset import ZSet, ZSetAddition


T = TypeVar("T")


def V(name: str) -> Variable:
    """Create a Datalog variable."""
    return Variable(name)


def atom(predicate: str, *terms: Term) -> Atom:
    """Create a Datalog atom, e.g. ``atom("tc", X, Y)``."""
    return (predicate, tuple(terms))


def not_(body_atom: Atom) -> Atom:
    """Negate a body atom using the interpreter's ``!predicate`` encoding."""
    predicate, terms = body_atom
    if predicate.startswith("!"):
        raise ValueError(f"atom is already negated: {predicate}")
    return (f"!{predicate}", terms)


def fact(predicate: str, *values: Constant) -> Fact:
    """Create a Datalog fact, e.g. ``fact("e", 0, 1)``."""
    return (predicate, tuple(values))


def rule(head: Atom, *body: Atom) -> Rule:
    """Create a Datalog rule from a head atom and zero or more body atoms."""
    return (head, *body)


def _weighted_zset(entries: tuple[T | tuple[T, int], ...]) -> ZSet[T]:
    values: dict[T, int] = {}
    for entry in entries:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], int):
            item, weight = entry
        else:
            item, weight = entry, 1
        values[item] = values.get(item, 0) + weight
        if values[item] == 0:
            del values[item]
    return ZSet(values)


def facts(*entries: Fact | tuple[Fact, int]) -> EDB:
    """Create an EDB ``ZSet`` from facts, optionally with explicit weights."""
    return _weighted_zset(entries)


def rules(*entries: Rule | tuple[Rule, int]) -> Program:
    """Create a program ``ZSet`` from rules, optionally with explicit weights."""
    return _weighted_zset(entries)


def _as_zset(value: ZSet[T] | Mapping[T, int] | Iterable[T] | None) -> ZSet[T]:
    if value is None:
        return ZSet({})
    if isinstance(value, ZSet):
        return value
    if isinstance(value, Mapping):
        return ZSet({k: v for k, v in value.items() if v != 0})
    return ZSet({item: 1 for item in value})


def _copy_zset(value: ZSet[T]) -> ZSet[T]:
    return ZSet(dict(value.inner))


class Datalog:
    """Stateful wrapper around the flat 2-D Datalog circuits.

    ``step`` accepts fact and program deltas, advances one outer tick,
    returns the output delta, and updates ``materialized``.
    """

    def __init__(
        self,
        *,
        indexed: bool = False,
        negation: bool = False,
        parallelism: int = 1,
    ) -> None:
        if indexed and negation:
            raise ValueError(
                "indexed Datalog currently supports positive programs only; "
                "use StratifiedDatalog for indexed stratified negation"
            )
        if negation:
            self.circuit: DatalogCircuit = IncrementalDatalogWithNegation(parallelism=parallelism)
        elif indexed:
            self.circuit = IncrementalDatalogWithIndexing(parallelism=parallelism)
        else:
            self.circuit = IncrementalDatalog(parallelism=parallelism)
        self.tick = 0
        self.last_inner_tick: int | None = None
        self._addition: ZSetAddition[Fact] = ZSetAddition()
        self._materialized: EDB = self._addition.identity()

    def step(
        self,
        *,
        facts: EDB | Mapping[Fact, int] | Iterable[Fact] | None = None,
        program: Program | Mapping[Rule, int] | Iterable[Rule] | None = None,
        max_inner: int = 1 << 16,
    ) -> EDB:
        outer_tick = self.tick
        self.circuit.edb.push((outer_tick,), _as_zset(facts))
        self.circuit.program.push((outer_tick,), _as_zset(program))
        self.last_inner_tick = saturate(
            self.circuit,
            outer_tick=outer_tick,
            max_inner=max_inner,
        )
        delta = self.delta_at(outer_tick)
        self._materialized = self._addition.add(self._materialized, delta)
        self.tick += 1
        return delta

    def delta_at(self, tick: int) -> EDB:
        return _copy_zset(self.circuit.observable_at((tick,)))

    def materialized(self) -> EDB:
        return _copy_zset(self._materialized)

    def relation(self, predicate: str) -> ZSet[tuple[Constant, ...]]:
        return ZSet(
            {
                args: weight
                for (pred, args), weight in self._materialized.inner.items()
                if pred == predicate and weight != 0
            }
        )


class StratifiedDatalog:
    """Stateful wrapper around dynamic 3-D stratified Datalog.

    Note: the 3-D path is currently unrefined — on equivalent
    workloads it is roughly an order of magnitude slower than the 2-D
    ``Datalog`` (semipositive negation only) and parallel scaling tops
    out around 1.4× regardless of ``parallelism``. Use it when you need
    full multi-stratum negation; expect a perf cliff vs ``Datalog``.
    """

    def __init__(self, *, parallelism: int = 1) -> None:
        self.circuit: DynamicStratifiedCircuit = IncrementalDatalogStratified(parallelism=parallelism)
        self.tick = 0
        self.last_inner_ticks: list[int] | None = None
        self._addition: ZSetAddition[Fact] = ZSetAddition()
        self._materialized: EDB = self._addition.identity()

    def step(
        self,
        *,
        facts: EDB | Mapping[Fact, int] | Iterable[Fact] | None = None,
        program: Program | Mapping[Rule, int] | Iterable[Rule] | None = None,
        max_inner: int = 1 << 16,
    ) -> EDB:
        outer_tick = self.tick
        self.circuit.edb.push((outer_tick,), _as_zset(facts))
        self.circuit.program.push((outer_tick,), _as_zset(program))
        self.last_inner_ticks = saturate_stratified(
            self.circuit,
            outer_tick=outer_tick,
            max_inner=max_inner,
        )
        delta = self.delta_at(outer_tick)
        self._materialized = self._addition.add(self._materialized, delta)
        self.tick += 1
        return delta

    def delta_at(self, tick: int) -> EDB:
        return _copy_zset(self.circuit.observable_at((tick,)))

    def materialized(self) -> EDB:
        return _copy_zset(self._materialized)

    def relation(self, predicate: str) -> ZSet[tuple[Constant, ...]]:
        return ZSet(
            {
                args: weight
                for (pred, args), weight in self._materialized.inner.items()
                if pred == predicate and weight != 0
            }
        )


class Reachability:
    """Stateful wrapper around incremental transitive closure."""

    def __init__(self, *, indexed: bool = True, parallelism: int = 1) -> None:
        if indexed:
            self.circuit: ReachabilityCircuit = IncrementalReachabilityWithIndexing(parallelism=parallelism)
        else:
            self.circuit = IncrementalReachability(parallelism=parallelism)
        self.tick = 0
        self.last_inner_tick: int | None = None
        self._addition: ZSetAddition[Edge] = ZSetAddition()
        self._materialized: ZSet[Edge] = self._addition.identity()

    def step(
        self,
        edges: ZSet[Edge] | Mapping[Edge, int] | Iterable[Edge] | None = None,
        *,
        max_inner: int = 1 << 16,
    ) -> ZSet[Edge]:
        outer_tick = self.tick
        self.circuit.edges.push((outer_tick,), _as_zset(edges))
        self.last_inner_tick = saturate_reach(
            self.circuit,
            outer_tick=outer_tick,
            max_inner=max_inner,
        )
        delta = self.delta_at(outer_tick)
        self._materialized = self._addition.add(self._materialized, delta)
        self.tick += 1
        return delta

    def delta_at(self, tick: int) -> ZSet[Edge]:
        return _copy_zset(self.circuit.observable_at((tick,)))

    def materialized(self) -> ZSet[Edge]:
        return _copy_zset(self._materialized)


class RDFS:
    """Stateful wrapper around incremental RDFS materialization."""

    def __init__(self, *, indexed: bool = True, parallelism: int = 1) -> None:
        if indexed:
            self.circuit: RDFSCircuit = IncrementalRDFSMaterializationWithIndexing(parallelism=parallelism)
        else:
            self.circuit = IncrementalRDFSMaterialization(parallelism=parallelism)
        self.tick = 0
        self.last_inner_tick: int | None = None
        self._addition: ZSetAddition[RDFTuple] = ZSetAddition()
        self._materialized: RDFGraph = self._addition.identity()

    def step(
        self,
        *,
        abox: RDFGraph | Mapping[RDFTuple, int] | Iterable[RDFTuple] | None = None,
        tbox: RDFGraph | Mapping[RDFTuple, int] | Iterable[RDFTuple] | None = None,
        max_inner: int = 1 << 16,
    ) -> RDFGraph:
        outer_tick = self.tick
        self.circuit.abox.push((outer_tick,), _as_zset(abox))
        self.circuit.tbox.push((outer_tick,), _as_zset(tbox))
        self.last_inner_tick = saturate_rdfs(
            self.circuit,
            outer_tick=outer_tick,
            max_inner=max_inner,
        )
        delta = self.delta_at(outer_tick)
        self._materialized = self._addition.add(self._materialized, delta)
        self.tick += 1
        return delta

    def delta_at(self, tick: int) -> RDFGraph:
        return _copy_zset(self.circuit.observable_at((tick,)))

    def materialized(self) -> RDFGraph:
        return _copy_zset(self._materialized)


__all__ = [
    "Datalog",
    "RDFS",
    "Reachability",
    "StratifiedDatalog",
    "V",
    "atom",
    "fact",
    "facts",
    "not_",
    "rule",
    "rules",
]
