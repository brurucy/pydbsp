from abc import abstractmethod
from dataclasses import dataclass, field
import math as _math
from typing import Protocol, TypeVar, cast

T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")


class AbelianGroupOperation(Protocol[T]):
    """
    This protocol defines the basic operations and properties of an Abelian group,
    addition and negation.
    """

    @abstractmethod
    def add(self, a: T, b: T) -> T:
        raise NotImplementedError

    @abstractmethod
    def neg(self, a: T) -> T:
        """Returns the inverse of the given element"""
        raise NotImplementedError

    @abstractmethod
    def identity(self) -> T:
        """Returns the identity, zero, element of the group"""
        raise NotImplementedError

    def is_commutative(self, a: T, b: T) -> bool:
        """
        Returns whether a + b == b + a.
        """
        test = self.add(a, b) == self.add(b, a)
        if not test:
            print(f"Failed commutativity assertion: {self.add(a, b)} == {self.add(b, a)}")

        return test

    def is_associative(self, a: T, b: T, c: T) -> bool:
        """
        Returns whether (a + b) + c == a + (b + c).
        """
        test = self.add(self.add(a, b), c) == self.add(a, self.add(b, c))
        if not test:
            print(f"Failed associativity assertion: {self.add(self.add(a, b), c)} == {self.add(a, self.add(b, c))}")

        return test

    def has_identity(self, a: T) -> bool:
        """
        Checks if the identity element behaves correctly for the given element.
        """
        identity = self.identity()
        test = self.add(a, identity) == a and self.add(identity, a) == a
        if not test:
            print(f"Failed identity assertion: {self.add(a, identity)} == {self.add(identity, a)}")

        return test

    def has_inverse(self, a: T) -> bool:
        """
        Returns if the given element has a well defined inverse.
        """
        identity = self.identity()
        inv_a = self.neg(a)
        test = self.add(a, inv_a) == identity and self.add(inv_a, a) == identity
        if not test:
            print(f"Failed inverse assertion: {self.add(a, inv_a)} == {self.add(inv_a, a)}")

        return test


class BoundedBelowLattice[T](Protocol):
    """
    Partial order with binary meet, binary join, and a least element.

    ``leq`` has a default derived from ``meet``: ``a ≤ b`` iff
    ``meet(a, b) == a``. Subclasses may override for efficiency.
    """

    @abstractmethod
    def join(self, a: T, b: T) -> T:
        raise NotImplementedError

    @abstractmethod
    def meet(self, a: T, b: T) -> T:
        raise NotImplementedError

    @abstractmethod
    def bottom(self) -> T:
        raise NotImplementedError

    @abstractmethod
    def top(self) -> T:
        """Supremum — the greatest element. For per-axis chains this
        is ω, the completion of ℕ as a bounded chain. For product
        lattices it is componentwise ω.
        """
        raise NotImplementedError

    def leq(self, a: T, b: T) -> bool:
        return self.meet(a, b) == a

    def is_commutative(self, a: T, b: T) -> bool:
        test = self.join(a, b) == self.join(b, a) and self.meet(a, b) == self.meet(b, a)
        if not test:
            print(f"Failed commutativity at ({a}, {b})")
        return test

    def is_associative(self, a: T, b: T, c: T) -> bool:
        test = self.join(self.join(a, b), c) == self.join(a, self.join(b, c)) and self.meet(
            self.meet(a, b), c
        ) == self.meet(a, self.meet(b, c))
        if not test:
            print(f"Failed associativity at ({a}, {b}, {c})")
        return test

    def is_absorptive(self, a: T, b: T) -> bool:
        test = self.join(a, self.meet(a, b)) == a and self.meet(a, self.join(a, b)) == a
        if not test:
            print(f"Failed absorption at ({a}, {b})")
        return test

    def is_idempotent(self, a: T) -> bool:
        test = self.join(a, a) == a and self.meet(a, a) == a
        if not test:
            print(f"Failed idempotence at {a}")
        return test

    def is_reflexive(self, a: T) -> bool:
        test = self.leq(a, a)
        if not test:
            print(f"Failed reflexivity at {a}")
        return test

    def is_antisymmetric(self, a: T, b: T) -> bool:
        test = not (self.leq(a, b) and self.leq(b, a)) or a == b
        if not test:
            print(f"Failed antisymmetry at ({a}, {b})")
        return test

    def is_transitive(self, a: T, b: T, c: T) -> bool:
        test = not (self.leq(a, b) and self.leq(b, c)) or self.leq(a, c)
        if not test:
            print(f"Failed transitivity at ({a}, {b}, {c})")
        return test

    def is_leq_consistent_with_meet(self, a: T, b: T) -> bool:
        test = self.leq(a, b) == (self.meet(a, b) == a)
        if not test:
            print(f"Failed leq/meet consistency at ({a}, {b})")
        return test

    def is_least_at(self, a: T) -> bool:
        test = self.leq(self.bottom(), a)
        if not test:
            print(f"Failed bottom ≤ {a}")
        return test


class Chain[T](BoundedBelowLattice[T], Protocol):
    """
    Totally-ordered lattice with discrete successor and predecessor.

    A chain refines ``BoundedBelowLattice`` with **totality**: for
    every pair a, b, either ``leq(a, b)`` or ``leq(b, a)``. Under
    totality ``join`` and ``meet`` collapse to ``max`` and ``min``,
    provided as defaults via ``leq`` — which Chain re-asserts as
    abstract so subclasses supply the native order.

    A chain is **discrete**: every element has a unique ``successor``,
    and every non-bottom element has a unique ``predecessor``. Bottom
    has no predecessor (``None``).

    In DBSP each chain is an **axis**: a dimension along which operators
    like delay step. 1D timestamps live in a single chain (ℕ). Nested
    timestamps live in products of chains — one chain per nesting level.
    """

    @abstractmethod
    def leq(self, a: T, b: T) -> bool:
        raise NotImplementedError

    @abstractmethod
    def successor(self, t: T) -> T:
        raise NotImplementedError

    @abstractmethod
    def predecessor(self, t: T) -> T | None:
        """Predecessor of t, or None when t == bottom()."""
        raise NotImplementedError

    def join(self, a: T, b: T) -> T:
        return b if self.leq(a, b) else a

    def meet(self, a: T, b: T) -> T:
        return a if self.leq(a, b) else b

    def is_total(self, a: T, b: T) -> bool:
        test = self.leq(a, b) or self.leq(b, a)
        if not test:
            print(f"Failed totality at ({a}, {b})")
        return test

    def has_predecessor_inverse(self, t: T) -> bool:
        pred = self.predecessor(t)
        if pred is None:
            test = t == self.bottom()
            if not test:
                print(f"Failed predecessor: None returned for non-bottom {t}")
            return test

        test = self.successor(pred) == t
        if not test:
            print(f"Failed predecessor inverse at {t}: successor({pred}) = {self.successor(pred)}")
        return test


OMEGA: float = _math.inf
"""Supremum of the per-axis chain — ω, the top element of ℕ∪{ω}.

Represented as ``math.inf`` so Python's built-in comparisons
(``<=``, ``max``, ``min``) handle it correctly against any int. ω
is absorbing under ``successor``: ``successor(ω) = ω``.
"""


class NaturalChain(Chain[int]):
    """
    The natural numbers completed with a top: ℕ ∪ {ω}, as a bounded
    ``Chain[int]``.

    Elements are non-negative ints plus the sentinel ``OMEGA``
    (``math.inf``). The chain is a complete lattice of order type
    ω+1. ``top()`` returns ω; every downset has a finite antichain
    representation.

    Per-axis ω lets antichains express partial-universality (e.g.
    ``{(0, ω)}`` = "outer 0, every inner tick settled") — the
    completion needed to make flat product lattices equivalent to
    the nested ``Stream[Stream[…]]`` model's independent-per-cell
    inner antichains.
    """

    def bottom(self) -> int:
        return 0

    def top(self) -> int:
        return OMEGA  # type: ignore[return-value]

    def leq(self, a: int, b: int) -> bool:
        return a <= b

    def successor(self, t: int) -> int:
        # ω is absorbing: successor(ω) = ω. math.inf + 1 == math.inf.
        return t + 1 if t != OMEGA else t

    def predecessor(self, t: int) -> int | None:
        return None if t == 0 else (t if t == OMEGA else t - 1)


class DBSPTime[T: tuple[int, ...]](BoundedBelowLattice[T]):
    """
    The timestamp lattice for N-nested DBSP streams: ℕᴺ under
    componentwise product order. Parameterised by the tuple element
    type ``T`` (bounded by ``tuple[int, ...]``) so specialisations
    like ``DBSPTime[tuple[int, int]]`` thread a concrete arity through
    the whole API (antichains, operators, History).

    ``nestedness`` is the single runtime parameter; must be ``>= 1``.
    ``factors[i]`` exposes the per-axis ``NaturalChain`` — operators
    that step along a specific axis (delay, integrate) reach into
    ``factors[i]`` for the chain they need.

    * elements: ``N``-tuples of non-negative ints (``tuple[int]`` for
      ``N=1``)
    * order: componentwise ≤
    * ``join`` / ``meet`` / ``bottom``: componentwise

    Distributive lattice. Not a chain for ``nestedness >= 2`` — two
    elements can be incomparable (e.g. ``(1, 3)`` and ``(3, 1)``). For
    ``nestedness == 1`` elements are 1-tuples, retaining the uniform
    tuple shape of the API.
    """

    nestedness: int
    factors: tuple[NaturalChain, ...]

    def __init__(self, nestedness: int) -> None:
        if nestedness < 0:
            raise ValueError(f"nestedness must be >= 0; got {nestedness}")
        self.nestedness = nestedness
        self.factors = tuple(NaturalChain() for _ in range(nestedness))

    def bottom(self) -> T:
        return cast(T, (0,) * self.nestedness)

    def top(self) -> T:
        return cast(T, (OMEGA,) * self.nestedness)

    def at(self, *components: int) -> T:
        """Build a validated time tuple for this lattice's arity.

        Prefer ``lattice.at(k, j)`` over raw ``(k, j)`` at API
        boundaries (``Input.push``, ``Evaluator.at_op``, …): arity
        mismatches and negative components get caught at call time
        instead of surfacing as opaque dict-lookup misses downstream.
        """
        if len(components) != self.nestedness:
            raise ValueError(
                f"time arity mismatch: "
                f"DBSPTime(nestedness={self.nestedness}) expects "
                f"{self.nestedness}-tuple, got {len(components)} "
                f"components: {components}"
            )
        for i, c in enumerate(components):
            if c < 0:
                raise ValueError(
                    f"time components must be >= 0; component[{i}] = {c}"
                )
        return cast(T, components)

    def leq(self, a: T, b: T) -> bool:
        return all(ai <= bi for ai, bi in zip(a, b))

    def join(self, a: T, b: T) -> T:
        return cast(T, tuple(max(ai, bi) for ai, bi in zip(a, b)))

    def meet(self, a: T, b: T) -> T:
        return cast(T, tuple(min(ai, bi) for ai, bi in zip(a, b)))

    def advance_antichain(
        self,
        frontier: "Antichain[T]",
        axis: int,
    ) -> "Antichain[T]":
        """Extend ``frontier`` by one successor on ``axis``. Empty
        frontier seeds at ``bottom`` — the clock's first tick.
        Universal frontier is a no-op (already total).
        """
        if frontier.is_universal:
            return frontier
        if not frontier.elements:
            out: Antichain[T] = Antichain(self)
            out.insert(self.bottom())
            return out
        chain = self.factors[axis]
        out = Antichain(self)
        for e in frontier.elements:
            shifted = cast(T, tuple(chain.successor(e[i]) if i == axis else e[i] for i in range(self.nestedness)))
            out.insert(shifted)
        return out


# DBSPTime presets. Every timestamp is a tuple, even in 1D —
# ``Time1D`` holds 1-tuples so the API is uniform across nesting
# depths. ``NaturalChain`` is retained as the per-axis factor primitive
# inside ``DBSPTime.factors`` but is no longer used as a stream's time
# lattice directly.
Time0D: DBSPTime[tuple[()]] = DBSPTime(nestedness=0)
Time1D: DBSPTime[tuple[int]] = DBSPTime(nestedness=1)
Time2D: DBSPTime[tuple[int, int]] = DBSPTime(nestedness=2)
Time3D: DBSPTime[tuple[int, int, int]] = DBSPTime(nestedness=3)


def dbsp_time(nestedness: int) -> DBSPTime[tuple[int, ...]]:
    """Return the canonical DBSP time lattice for the given nesting depth."""
    return DBSPTime(nestedness=nestedness)


@dataclass
class Antichain[T]:
    """
    Minimal set of mutually-incomparable elements of a ``BoundedBelowLattice[T]``.

    In DBSP, an Antichain is a **progress frontier**: the set of maximal
    observed timestamps. The down-set of the antichain (every element
    ``≤`` some antichain member) is the **settled region** — timestamps
    whose values are determined.

    The set of all antichains over a ``BoundedBelowLattice[T]`` itself
    forms a distributive **bounded** lattice under down-set inclusion:

    * ``⊥`` — the empty antichain. Down-set = ∅.
    * ``⊤`` — the **universal** antichain (``is_universal = True``).
      Down-set = the whole lattice; ``covers(x) = True`` for every x.
    * ``∨`` = union of down-sets, ``∧`` = intersection, ``⊑`` = down-set
      containment.

    Note that the antichain lattice is bounded **above** even when the
    base lattice (e.g. ``NaturalChain`` = ℕ) isn't — the universal
    antichain represents "settled everywhere" without needing a top
    element in the base lattice. It is the frontier of identity
    streams: the zero stream of ``StreamAddition`` is total (``0``
    everywhere), so its settled region is the whole domain.
    """

    lattice: BoundedBelowLattice[T]
    elements: list[T] = field(default_factory=list)
    is_universal: bool = False

    def __post_init__(self) -> None:
        if self.is_universal:
            self.elements = []

    @classmethod
    def universal(cls, lattice: "BoundedBelowLattice[T]") -> "Antichain[T]":
        """The top of the antichain lattice — covers every element."""
        return cls(lattice=lattice, elements=[], is_universal=True)

    def insert(self, element: T) -> None:
        """
        Add element. Noop if element is already covered; otherwise remove
        any existing elements it dominates, then add it. Universal
        antichains absorb every insert.
        """
        if self.is_universal:
            return

        for existing in self.elements:
            if self.lattice.leq(element, existing):
                return

        self.elements = [e for e in self.elements if not self.lattice.leq(e, element)]
        self.elements.append(element)
        self.elements.sort(key=repr)

    def covers(self, element: T) -> bool:
        """
        True iff ``element`` is in the down-set — some antichain member
        is ``≥ element``, or the antichain is universal.
        """
        if self.is_universal:
            return True
        return any(self.lattice.leq(element, f) for f in self.elements)

    def leq(self, other: "Antichain[T]") -> bool:
        """
        Antichain ordering: ``self ⊑ other`` iff ``self``'s down-set is
        contained in ``other``'s. Equivalently: every element of ``self``
        is covered by ``other``. Universal ⊑ X iff X is universal.
        """
        if other.is_universal:
            return True
        if self.is_universal:
            return False
        return all(other.covers(a) for a in self.elements)

    def join(self, other: "Antichain[T]") -> "Antichain[T]":
        """
        Antichain whose down-set is the union of both down-sets.
        """
        if self.is_universal or other.is_universal:
            return Antichain.universal(self.lattice)
        out = Antichain(self.lattice)
        for e in self.elements + other.elements:
            out.insert(e)
        return out

    def meet(self, other: "Antichain[T]") -> "Antichain[T]":
        """
        Antichain whose down-set is the intersection of both down-sets.

        Computed as the maximal elements of pairwise lattice-meets of
        ``self`` and ``other`` — justified by ``x ≤ a ∧ x ≤ b`` iff
        ``x ≤ meet(a, b)``. Universal is the identity of ``meet``.
        """
        if self.is_universal:
            return other.clone()
        if other.is_universal:
            return self.clone()
        out = Antichain(self.lattice)
        for a in self.elements:
            for b in other.elements:
                out.insert(self.lattice.meet(a, b))
        return out

    def clone(self) -> "Antichain[T]":
        c = Antichain(self.lattice, list(self.elements))
        c.is_universal = self.is_universal
        return c


class ProductGroup[A, B](AbelianGroupOperation[tuple[A, B]]):
    """Direct product of two abelian groups. Identity, add, and neg
    are componentwise. Used when a fixpoint's state has multiple
    components (e.g. ``(Facts × Rewrites)`` in Datalog) — a single
    stream over the product group plays the role of Lean's
    ``stream (α × β)``.
    """

    def __init__(
        self,
        first: AbelianGroupOperation[A],
        second: AbelianGroupOperation[B],
    ) -> None:
        self._first: AbelianGroupOperation[A] = first
        self._second: AbelianGroupOperation[B] = second

    def identity(self) -> tuple[A, B]:
        return (self._first.identity(), self._second.identity())

    def add(self, a: tuple[A, B], b: tuple[A, B]) -> tuple[A, B]:
        return (self._first.add(a[0], b[0]), self._second.add(a[1], b[1]))

    def neg(self, a: tuple[A, B]) -> tuple[A, B]:
        return (self._first.neg(a[0]), self._second.neg(a[1]))
