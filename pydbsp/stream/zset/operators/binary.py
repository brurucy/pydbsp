from pydbsp.core import DBSPTime
from pydbsp.stream import Lift2, Stream
from pydbsp.stream.operators.linear import (
    Delay,
    Differentiate,
    Integrate,
)
from pydbsp.zset import ZSet, ZSetAddition
from pydbsp.zset.functions.binary import H


def DLDDistinct[V, T: tuple[int, ...]](
    diff_stream: Stream[ZSet[V], T],
    inner_group: ZSetAddition[V],
    lattice: DBSPTime[T],
    outer_axis: int = 0,
    inner_axis: int = 1,
) -> Stream[ZSet[V], T]:
    """``Dᵒ(H(z⁻¹ⁱ Iⁱ Iᵒ s, Iᵒ s))`` — doubly-incremental distinct
    on a flat product lattice.
    """
    integrated = Integrate(diff_stream, lattice, axis=outer_axis)
    int_int = Integrate(integrated, lattice, axis=inner_axis)
    del_int_int = Delay(int_int, lattice, axis=inner_axis)
    h = Lift2(del_int_int, integrated, H, inner_group)
    return Differentiate(h, lattice, axis=outer_axis)


def DLDDistinct3D[V, T: tuple[int, ...]](
    diff_stream: Stream[ZSet[V], T],
    inner_group: ZSetAddition[V],
    lattice: DBSPTime[T],
    axes: tuple[int, int, int] = (0, 1, 2),
) -> Stream[ZSet[V], T]:
    """``D_{α₀}(D_{α₁}(H(z⁻¹_{α₂} I_{α₂} I_{α₀} I_{α₁} s, I_{α₀} I_{α₁} s)))``
    — triply-incremental distinct on a 3-axis flat product lattice.

    Derivation: `Δ_{α₂} distinct(I_{α₀} I_{α₁} I_{α₂} s) =
    H(z⁻¹_{α₂} I_{α₂} I_{α₀} I_{α₁} s, I_{α₀} I_{α₁} s)`. Then the outer
    `Δ_{α₁}` and `Δ_{α₀}` are ordinary differentiates because
    `H` already encodes the `Δ_{α₂}`. Extends the 2D formula
    ``Dᵒ(H(z⁻¹ⁱIⁱIᵒs, Iᵒs))`` by pre-integrating over the
    second axis before `H` and adding a corresponding differentiate.
    """
    a0, a1, a2 = axes
    int_a0 = Integrate(diff_stream, lattice, axis=a0)
    int_a01 = Integrate(int_a0, lattice, axis=a1)
    int_a012 = Integrate(int_a01, lattice, axis=a2)
    del_int_a012 = Delay(int_a012, lattice, axis=a2)
    h = Lift2(del_int_a012, int_a01, H, inner_group)
    d1 = Differentiate(h, lattice, axis=a1)
    return Differentiate(d1, lattice, axis=a0)
