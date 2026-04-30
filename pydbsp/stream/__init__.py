from abc import abstractmethod
from typing import Callable, Protocol
from pydbsp.core import (
    AbelianGroupOperation,
    Antichain,
    BoundedBelowLattice,
    DBSPTime,
)


class Stream[V, Time](Protocol):
    """A DBSP stream: a function ``Time → V`` presented through a
    progress frontier and on-demand reads.

    ``Time`` is an element of some lattice. In practice this is
    ``DBSPTime(N)`` — the product lattice ℕᴺ — so every ``Time`` value
    is a ``tuple[int, ...]``, including the 1D case. Each of the
    lattice's ``N`` axes is a ``NaturalChain`` along which operators
    like ``Delay`` / ``Integrate`` / ``Differentiate`` can step.

    Progress is tracked as an ``Antichain[Time]``. The down-set of the
    frontier is the settled region — the timestamps at which ``at(t)``
    is defined. Stepping = monotone extension of the frontier.

    **Frontier rule vs value rule.** A concrete operator supplies a
    frontier rule (``settled_frontier``) and a value rule
    (``_compute``). The base ``at`` stitches them together: it
    enforces the causality check against the frontier and delegates to
    ``_compute`` for the actual computation. Subclasses implement
    ``_compute`` and never override ``at`` — the causality invariant
    is maintained structurally.

    Invariants every implementation must honor:

    1. **Frontier monotonicity.** Over the lifetime of a stream
       instance, the settled frontier only grows in the covers
       relation.
    2. **Stability of settled values.** For any ``t`` covered by the
       frontier, repeated ``at(t)`` calls return equal values.
    3. **Causality well-formedness.** Handled uniformly by ``at`` —
       reads outside the frontier raise ``IndexError``.
    4. **Fixed group and time lattice.** Neither ``group`` nor
       ``time_lattice`` changes over a stream's lifetime.

    Induced: streams form a pointwise abelian group when ``V`` does.
    """

    _stream_attrs: tuple[str, ...] = ()
    """Names of attributes on this stream that hold input streams —
    used to walk the circuit graph. Operators that take
    stream inputs override at the class level (e.g. ``Lift2`` sets
    ``("_left", "_right")``). Leaf operators (sources) leave it empty.
    """

    @property
    def group(self) -> AbelianGroupOperation[V]: ...

    @property
    def time_lattice(self) -> BoundedBelowLattice[Time]: ...

    @property
    def settled_frontier(self) -> Antichain[Time]: ...

    @abstractmethod
    def _compute(self, t: Time) -> V:
        """Value rule: the stream's value at a frontier-legal ``t``.
        Invoked only from ``at`` after the causality check passes.
        """
        ...

    def at(self, t: Time) -> V:
        """Read the value at ``t``. Delegates the value rule to
        ``_compute``; subclasses never override this — override
        ``_compute`` instead.

        **Causality is the observer's responsibility.** Legality of
        ``t`` is checked at the observation boundary (e.g. ``History``
        against its cursor) — not here, on every internal read. The
        frontier rule composes: if the observer's read is frontier-
        legal and every operator's frontier rule is honest, every dep
        read this cascades into is legal by construction. Putting the
        check here once demanded caching the frontier to stay cheap,
        which broke the moment an input's frontier could grow. Kept
        out of the hot path, frontiers remain a live property of the
        stream and are consulted only when the observer needs them.
        """
        return self._compute(t)

    def __add__(self, other: "Stream[V, Time]") -> "Stream[V, Time]":
        """Pointwise sum. ``a + b`` is ``Lift2(a, b, group.add, group)``.

        Both operands must share the same abelian group; ``self.group``
        is used. Saves the ``StreamAddition(group, lattice).add(a, b)``
        incantation.
        """
        return Lift2(self, other, self.group.add, self.group)

    def __neg__(self) -> "Stream[V, Time]":
        """Pointwise negation."""
        return Lift1(self, self.group.neg, self.group)

    def __sub__(self, other: "Stream[V, Time]") -> "Stream[V, Time]":
        return self + (-other)

    def is_settled(self, t: Time) -> bool:
        return self.settled_frontier.covers(t)


class Lift1[A, B, Time](Stream[B, Time]):
    """Pointwise unary operator: ``Lift1(s, f, g).at(t) = f(s.at(t))``.

    Frontier rule: **identity** — same as ``s``.
    Value rule: apply ``f`` to ``s.at(t)``.
    """

    _stream_attrs = ("_base",)

    def __init__(
        self,
        base: Stream[A, Time],
        f: Callable[[A], B],
        out_group: AbelianGroupOperation[B],
    ) -> None:
        self._base = base
        self._f = f
        self._out_group = out_group

    @property
    def group(self) -> AbelianGroupOperation[B]:
        return self._out_group

    @property
    def time_lattice(self) -> BoundedBelowLattice[Time]:
        return self._base.time_lattice

    @property
    def settled_frontier(self) -> Antichain[Time]:
        return self._base.settled_frontier

    def _compute(self, t: Time) -> B:
        return self._f(self._base.at(t))

    def deps(self, t):
        return [(self._base, t)]

    def compute_from(self, t, slots):
        return self._f(slots[(id(self._base), t)])


class Lift2[A, B, C, Time](Stream[C, Time]):
    """Pointwise binary operator:
    ``Lift2(a, b, f, g).at(t) = f(a.at(t), b.at(t))``.

    Frontier rule: **meet** — ``a.frontier ⊓ b.frontier``.
    Value rule: apply ``f`` to both inputs at the same ``t``.
    """

    _stream_attrs = ("_left", "_right")

    def __init__(
        self,
        left: Stream[A, Time],
        right: Stream[B, Time],
        f: Callable[[A, B], C],
        out_group: AbelianGroupOperation[C],
    ) -> None:
        self._left = left
        self._right = right
        self._f = f
        self._out_group = out_group

    @property
    def group(self) -> AbelianGroupOperation[C]:
        return self._out_group

    @property
    def time_lattice(self) -> BoundedBelowLattice[Time]:
        return self._left.time_lattice

    @property
    def settled_frontier(self) -> Antichain[Time]:
        l = self._left.settled_frontier
        r = self._right.settled_frontier
        cached = getattr(self, "_fr_cache", None)
        if cached is not None and cached[0] is l and cached[1] is r:
            return cached[2]
        out = l.meet(r)
        self._fr_cache = (l, r, out)
        return out

    def _compute(self, t: Time) -> C:
        return self._f(self._left.at(t), self._right.at(t))

    def deps(self, t):
        return [(self._left, t), (self._right, t)]

    def compute_from(self, t, slots):
        return self._f(slots[(id(self._left), t)], slots[(id(self._right), t)])


class Input[V, T: tuple[int, ...]](Stream[V, T]):
    """Externally-driven stream — values pushed in via ``push(t, v)``.

    A leaf in the operator graph: no ``_stream_attrs``. Memoize treats
    it as a source; downstream ``Evaluator``s record it and
    keys its frontier cache on the identity of this stream's
    ``settled_frontier`` antichain.

    **Frontier grows monotonically.** Each ``push(t, v)`` installs a
    fresh ``Antichain`` object that extends the previous one to cover
    ``t``. The new object identity is what signals progress to
    downstream caches — they self-invalidate on next read.

    **Values are read-only at settled timestamps** (stability
    invariant). Reads at unsettled timestamps return
    ``group.identity()``; the observation boundary (``History``) is
    what gates whether a caller is *allowed* to look at a given ``t``.
    """

    _stream_attrs: tuple[str, ...] = ()

    def __init__(
        self,
        group: AbelianGroupOperation[V],
        lattice: DBSPTime[T],
    ) -> None:
        self._group = group
        self._lattice = lattice
        self._values: dict[T, V] = {}
        self._frontier: Antichain[T] = Antichain(lattice)

    @property
    def group(self) -> AbelianGroupOperation[V]:
        return self._group

    @property
    def time_lattice(self) -> BoundedBelowLattice[T]:
        return self._lattice

    @property
    def settled_frontier(self) -> Antichain[T]:
        return self._frontier

    def push(self, t: T, value: V) -> None:
        """Publishes ``value`` at timestamp ``t``. ``t`` must not already
        be covered by the frontier (frontiers grow monotonically).
        """
        if self._frontier.covers(t):
            raise ValueError(f"{t} already settled — cannot re-push")
        self._values[t] = value
        fresh: Antichain[T] = Antichain(self._lattice)
        for e in self._frontier.elements:
            fresh.insert(e)
        fresh.insert(t)
        self._frontier = fresh

    def _compute(self, t: T) -> V:
        return self._values.get(t, self._group.identity())

    def deps(self, t):
        return ()

    def compute_from(self, t, slots):
        return self._values.get(t, self._group.identity())


