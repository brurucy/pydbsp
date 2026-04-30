from typing import cast

from pydbsp.core import OMEGA, Time0D
from pydbsp.core import (
    AbelianGroupOperation,
    Antichain,
    BoundedBelowLattice,
    DBSPTime,
)
from pydbsp.stream import Stream


class StreamIntroduction[V, T: tuple[int, ...]](Stream[V, T]):
    """``δ₀(v)`` — total function with ``v`` at the lattice's bottom
    and ``group.identity()`` everywhere else. Settled frontier is
    universal. Derived: ``δ_n = Delay ∘ δ_{n-1}``.
    """

    _stream_attrs = ()

    def __init__(
        self,
        value: V,
        group: AbelianGroupOperation[V],
        lattice: DBSPTime[T],
    ) -> None:
        self._value = value
        self._group = group
        self._lattice = lattice
        self._bottom = lattice.bottom()
        self._identity = group.identity()
        # Stable frontier object — identity is the progress indicator;
        # δ₀ is total, so the frontier never changes and the same
        # Antichain object is returned from every read.
        self._frontier: Antichain[T] = Antichain.universal(lattice)

    @property
    def group(self) -> AbelianGroupOperation[V]:
        return self._group

    @property
    def time_lattice(self) -> BoundedBelowLattice[T]:
        return self._lattice

    @property
    def settled_frontier(self) -> Antichain[T]:
        return self._frontier

    def _compute(self, t: T) -> V:
        return self._value if t == self._bottom else self._identity

    def deps(self, t):
        return ()

    def compute_from(self, t, slots):
        return self._value if t == self._bottom else self._identity


def StreamElimination[V, T: tuple[int, ...]](
    s: Stream[V, T],
    lattice: DBSPTime[T],
) -> V:
    """``∫ s`` — sum of ``s`` along its first axis, escaping to the
    underlying group. Thin shim over ``TimeAxisElimination`` collapsing to
    the 0-D lattice ``Time0D`` (single cell ``()``).
    """
    return TimeAxisElimination(s, axis=0, lattice_in=lattice, lattice_out=Time0D).at(())


# ---- Flat product-lattice variants -----------------------------------------


class TimeAxisIntroduction[V, T: tuple[int, ...]](Stream[V, T]):
    """Dim-raising pulse on a flat product lattice: at each ``t``,
    return ``base.at(drop_axis(t))`` when ``t[axis] == 0``,
    ``group.identity()`` elsewhere.

    ``base`` can be a ``Stream[V, T_in]`` on a smaller lattice
    (``T_in`` has one fewer axis than ``T``) or a raw value ``V``
    (implicit 0-D wrapping — the constant pulse).

    Frontier: if the base frontier is universal the output is
    universal too; otherwise the output frontier is the base
    frontier's elements lifted by inserting ``0`` at ``axis``
    (plus ``lattice.bottom()`` seed).
    """

    _stream_attrs = ("_base_stream",)

    def __init__(
        self,
        base: "Stream[V, tuple[int, ...]] | V",
        group: AbelianGroupOperation[V],
        lattice: DBSPTime[T],
        axis: int = 0,
    ) -> None:
        self._group = group
        self._lattice = lattice
        self._axis = axis
        self._identity = group.identity()
        # Duck-type: streams have a ``settled_frontier`` property.
        if hasattr(base, "settled_frontier"):
            self._base_stream: Stream[V, tuple[int, ...]] | None = cast(
                Stream[V, tuple[int, ...]], base
            )
            self._base_value: V | None = None
        else:
            self._base_stream = None
            self._base_value = cast(V, base)

    @property
    def group(self) -> AbelianGroupOperation[V]:
        return self._group

    @property
    def time_lattice(self) -> BoundedBelowLattice[T]:
        return self._lattice

    @property
    def settled_frontier(self) -> Antichain[T]:
        if self._base_stream is None:
            return Antichain.universal(self._lattice)
        base_fr = self._base_stream.settled_frontier
        cached = getattr(self, "_fr_cache", None)
        if cached is not None and cached[0] is base_fr:
            return cached[1]
        if base_fr.is_universal:
            out = Antichain.universal(self._lattice)
        else:
            out = Antichain(self._lattice)
            axis = self._axis
            for e in base_fr.elements:
                lifted = list(e)
                lifted.insert(axis, OMEGA)
                out.insert(cast(T, tuple(lifted)))
        self._fr_cache = (base_fr, out)
        return out

    def _compute(self, t: T) -> V:
        if t[self._axis] != 0:
            return self._identity
        if self._base_stream is None:
            return cast(V, self._base_value)
        dropped = tuple(x for i, x in enumerate(t) if i != self._axis)
        return self._base_stream.at(cast(tuple[int, ...], dropped))

    def deps(self, t):
        if t[self._axis] != 0 or self._base_stream is None:
            return ()
        dropped = tuple(x for i, x in enumerate(t) if i != self._axis)
        return [(self._base_stream, dropped)]

    def compute_from(self, t, slots):
        if t[self._axis] != 0:
            return self._identity
        if self._base_stream is None:
            return self._base_value
        dropped = tuple(x for i, x in enumerate(t) if i != self._axis)
        return slots[(id(self._base_stream), dropped)]


class TimeAxisElimination[V, T: tuple[int, ...]](Stream[V, T]):
    """Collapse one axis of a flat product lattice, reducing lattice
    rank by one.

    ``TimeAxisElimination(base, axis, lattice_in, lattice_out).at(t_out)``
    sums ``base`` along the dropped ``axis`` while pinning the
    remaining coordinates to ``t_out``:

        sum_{k=0..K} base.at(t_in(k))

    where ``t_in(k)`` inserts ``k`` at position ``axis`` into
    ``t_out``. ``K`` is bounded by the dropped-axis extent of
    ``base.settled_frontier`` when finite; otherwise iteration
    follows the "first-zero" heuristic.

    Generalises ``StreamElimination`` (1D → scalar) and
    ``LiftedStreamElimination`` (2D nested → 1D outer) to a single
    axis-parameterised dim-reducing operator.
    """

    _stream_attrs = ("_base",)

    def __init__(
        self,
        base: Stream[V, tuple[int, ...]],
        axis: int = -1,
        lattice_in: DBSPTime[tuple[int, ...]] | None = None,
        lattice_out: DBSPTime[T] | None = None,
    ) -> None:
        # Infer lattice_in from base's time_lattice; default axis to
        # the last (innermost) axis; default lattice_out to a DBSPTime
        # with one fewer dimension.
        if lattice_in is None:
            lattice_in = base.time_lattice  # type: ignore[assignment]
        if axis < 0:
            axis = lattice_in.nestedness + axis
        if lattice_out is None:
            lattice_out = DBSPTime(nestedness=lattice_in.nestedness - 1)  # type: ignore[assignment]
        self._base = base
        self._axis = axis
        self._lattice_out = lattice_out

    @property
    def group(self) -> AbelianGroupOperation[V]:
        return self._base.group

    @property
    def time_lattice(self) -> BoundedBelowLattice[T]:
        return self._lattice_out

    @property
    def settled_frontier(self) -> Antichain[T]:
        base_fr = self._base.settled_frontier
        cached = getattr(self, "_fr_cache", None)
        if cached is not None and cached[0] is base_fr:
            return cached[1]
        if base_fr.is_universal:
            out = Antichain.universal(self._lattice_out)
        else:
            out = Antichain(self._lattice_out)
            axis = self._axis
            for e in base_fr.elements:
                dropped = tuple(x for i, x in enumerate(e) if i != axis)
                out.insert(cast(T, dropped))
        self._fr_cache = (base_fr, out)
        return out

    def _base_t(self, t_out: T, k: int) -> tuple[int, ...]:
        axis = self._axis
        out: list[int] = list(t_out)
        out.insert(axis, k)
        return cast(tuple[int, ...], tuple(out))

    def _max_axis_for(self, t_out: T) -> float:
        """Max axis value across all frontier elements (over-approx:
        includes cells outside the per-``t_out`` down-set, but cheap and
        safe — identity cells skip via ``_is_identity``)."""
        axis = self._axis
        best: float = -1
        for e in self._base.settled_frontier.elements:
            if e[axis] > best:
                best = e[axis]
        return best

    def _compute(self, t_out: T) -> V:
        base_fr = self._base.settled_frontier
        identity = self._base.group.identity()
        if not base_fr.is_universal and base_fr.elements:
            max_axis = self._max_axis_for(t_out)
            if max_axis < 0:
                return identity
            if max_axis != OMEGA:
                acc = identity
                for k in range(int(max_axis) + 1):
                    v = self._base.at(self._base_t(t_out, k))
                    inner = getattr(v, "inner", None)
                    if inner is not None and not inner:
                        continue
                    acc = self._base.group.add(acc, v)
                return acc
            # ω on the collapsed axis — fall through to first-zero loop.
        max_steps = 1 << 20
        acc = identity
        for k in range(max_steps):
            delta = self._base.at(self._base_t(t_out, k))
            if delta == identity:
                return acc
            acc = self._base.group.add(acc, delta)
        raise RuntimeError(f"TimeAxisElimination did not converge in {max_steps} steps")

    def deps(self, t):
        base_fr = self._base.settled_frontier
        if base_fr.is_universal or not base_fr.elements:
            return ()
        max_axis_val = self._max_axis_for(t)
        if max_axis_val < 0 or max_axis_val == OMEGA:
            return ()
        out = []
        for k in range(int(max_axis_val) + 1):
            lifted = list(t)
            lifted.insert(self._axis, k)
            out.append((self._base, tuple(lifted)))
        return out

    def compute_from(self, t, slots):
        base_fr = self._base.settled_frontier
        identity = self._base.group.identity()
        if not base_fr.is_universal and base_fr.elements:
            max_axis_val = self._max_axis_for(t)
            if max_axis_val < 0:
                return identity
            if max_axis_val != OMEGA:
                acc = identity
                for k in range(int(max_axis_val) + 1):
                    lifted = list(t)
                    lifted.insert(self._axis, k)
                    v = slots.get((id(self._base), tuple(lifted)), identity)
                    inner = getattr(v, "inner", None)
                    if inner is not None and not inner:
                        continue
                    acc = self._base.group.add(acc, v)
                return acc
        return self._compute(t)
