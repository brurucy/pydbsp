from typing import cast

from pydbsp.core import (
    AbelianGroupOperation,
    Antichain,
    BoundedBelowLattice,
    DBSPTime,
)
from pydbsp.stream import Lift1, Lift2, Stream
from pydbsp.stream.functions.linear import StreamIntroduction


class Delay[V, T: tuple[int, ...]](Stream[V, T]):
    """Delay along a chosen axis of a ``DBSPTime`` lattice.

    ``Delay(s, lattice, axis=i).at(t)`` is ``identity`` when
    ``t[i] == 0``, otherwise ``s.at(t')`` with ``t'[i] = t[i] - 1``
    and other coordinates unchanged.

    Frontier rule:

    * Base frontier universal → output frontier universal.
    * Otherwise, **shift + seed**: each ``e`` in ``base.frontier``
      gets its ``axis``-coordinate incremented; plus
      ``lattice.bottom()`` to seed identity at the base.

    Value rule: identity at the axis-bottom, base at axis-predecessor
    elsewhere.
    """

    _stream_attrs = ("_base",)

    def __init__(
        self,
        base: Stream[V, T],
        lattice: DBSPTime[T],
        axis: int = 0,
    ) -> None:
        self._base = base
        self._lattice = lattice
        self._axis = axis

    @property
    def group(self) -> AbelianGroupOperation[V]:
        return self._base.group

    @property
    def time_lattice(self) -> BoundedBelowLattice[T]:
        return self._lattice

    @property
    def settled_frontier(self) -> Antichain[T]:
        base_fr = self._base.settled_frontier
        cached = getattr(self, "_fr_cache", None)
        if cached is not None and cached[0] is base_fr:
            return cached[1]
        if base_fr.is_universal:
            out = Antichain.universal(self._lattice)
        else:
            out = Antichain(self._lattice)
            out.insert(self._lattice.bottom())
            axis = self._axis
            for e in base_fr.elements:
                shifted = tuple(x + 1 if i == axis else x for i, x in enumerate(e))
                out.insert(cast(T, shifted))
        self._fr_cache = (base_fr, out)
        return out

    def _compute(self, t: T) -> V:
        axis = self._axis
        if t[axis] == 0:
            return self._base.group.identity()
        t_prev = tuple(x - 1 if i == axis else x for i, x in enumerate(t))
        return self._base.at(cast(T, t_prev))

    def deps(self, t):
        axis = self._axis
        if t[axis] == 0:
            return ()
        prev = tuple(x - 1 if i == axis else x for i, x in enumerate(t))
        return [(self._base, prev)]

    def compute_from(self, t, slots):
        axis = self._axis
        if t[axis] == 0:
            return self._base.group.identity()
        prev = tuple(x - 1 if i == axis else x for i, x in enumerate(t))
        return slots[(id(self._base), prev)]


class Integrate[V, T: tuple[int, ...]](Stream[V, T]):
    """Running prefix sum along a chosen axis of a ``DBSPTime``
    lattice:

        Integrate(s, lattice, axis=i).at(t)
            = Σ_{k=0..t[i]} s.at(t') where t'[i] = k, others equal t.

    Frontier rule: **identity** — same as ``s``.
    """

    _stream_attrs = ("_base",)

    def __init__(
        self,
        base: Stream[V, T],
        lattice: DBSPTime[T],
        axis: int = 0,
    ) -> None:
        self._base = base
        self._lattice = lattice
        self._axis = axis

    @property
    def group(self) -> AbelianGroupOperation[V]:
        return self._base.group

    @property
    def time_lattice(self) -> BoundedBelowLattice[T]:
        return self._lattice

    @property
    def settled_frontier(self) -> Antichain[T]:
        return self._base.settled_frontier

    def _compute(self, t: T) -> V:
        axis = self._axis
        acc = self._base.at(t)
        k = t[axis]
        while k > 0:
            k -= 1
            t_prev = tuple(x if i != axis else k for i, x in enumerate(t))
            acc = self._base.group.add(self._base.at(cast(T, t_prev)), acc)
        return acc

    def deps(self, t):
        # Self-recurrence: Integrate.at(t) = base.at(t) + Integrate.at(t - e_i).
        axis = self._axis
        if t[axis] == 0:
            return [(self._base, t)]
        prev = tuple(x - 1 if i == axis else x for i, x in enumerate(t))
        return [(self._base, t), (self, prev)]

    def compute_from(self, t, slots):
        axis = self._axis
        current = slots[(id(self._base), t)]
        if t[axis] == 0:
            return current
        prev = tuple(x - 1 if i == axis else x for i, x in enumerate(t))
        prior = slots[(id(self), prev)]
        # Identity short-circuit: if ``base(t)`` is the group identity
        # (e.g. an empty ZSet), ``Integrate(t) = Integrate(t-e_α)``. Alias
        # the prior slot's value — no allocation, no ``group.add``. Cells
        # beyond the base's pushed support cost O(1) in Python terms.
        inner = getattr(current, "inner", None)
        if inner is not None and not inner:
            return prior
        return self._base.group.add(prior, current)


class Differentiate[V, T: tuple[int, ...]](Stream[V, T]):
    """Pairwise difference along a chosen axis:

        Differentiate(s, lattice, axis=i).at(t)
            = s.at(t) - s.at(t')    where t'[i] = t[i] - 1

    At ``t[i] == 0``, returns ``s.at(t)`` directly.

    Frontier rule: **identity** — same as ``s``.
    """

    _stream_attrs = ("_base",)

    def __init__(
        self,
        base: Stream[V, T],
        lattice: DBSPTime[T],
        axis: int = 0,
    ) -> None:
        self._base = base
        self._lattice = lattice
        self._axis = axis

    @property
    def group(self) -> AbelianGroupOperation[V]:
        return self._base.group

    @property
    def time_lattice(self) -> BoundedBelowLattice[T]:
        return self._lattice

    @property
    def settled_frontier(self) -> Antichain[T]:
        return self._base.settled_frontier

    def _compute(self, t: T) -> V:
        axis = self._axis
        current = self._base.at(t)
        if t[axis] == 0:
            return current
        t_prev = tuple(x if i != axis else x - 1 for i, x in enumerate(t))
        prev = self._base.at(cast(T, t_prev))
        return self._base.group.add(current, self._base.group.neg(prev))

    def deps(self, t):
        axis = self._axis
        if t[axis] == 0:
            return [(self._base, t)]
        prev = tuple(x - 1 if i == axis else x for i, x in enumerate(t))
        return [(self._base, t), (self._base, prev)]

    def compute_from(self, t, slots):
        axis = self._axis
        current = slots[(id(self._base), t)]
        if t[axis] == 0:
            return current
        prev = tuple(x - 1 if i == axis else x for i, x in enumerate(t))
        return self._base.group.add(current, self._base.group.neg(slots[(id(self._base), prev)]))


class StreamAddition[V, T: tuple[int, ...]](AbelianGroupOperation[Stream[V, T]]):
    """Pointwise abelian group on streams over an inner group.

    ``add(s1, s2) = Lift2(s1, s2, inner.add, inner)``;
    ``neg(s)      = Lift1(s, inner.neg, inner)``;
    ``identity()  = δ₀(inner.identity(), lattice)`` — total function.
    """

    def __init__(
        self,
        inner_group: AbelianGroupOperation[V],
        lattice: DBSPTime[T],
    ) -> None:
        self._inner_group = inner_group
        self._lattice = lattice

    def add(self, a: Stream[V, T], b: Stream[V, T]) -> Stream[V, T]:
        return Lift2(a, b, self._inner_group.add, self._inner_group)

    def neg(self, a: Stream[V, T]) -> Stream[V, T]:
        return Lift1(a, self._inner_group.neg, self._inner_group)

    def identity(self) -> Stream[V, T]:
        return StreamIntroduction(
            self._inner_group.identity(),
            self._inner_group,
            self._lattice,
        )


