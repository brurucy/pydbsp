from typing import Callable

from pydbsp.core import DBSPTime
from pydbsp.stream import Lift2, Stream
from pydbsp.stream.operators.linear import (
    Delay,
    Integrate,
    StreamAddition,
)
from pydbsp.zset import ZSet, ZSetAddition
from pydbsp.zset.functions.bilinear import join


def DLDJoin[A, B, C, T: tuple[int, ...]](
    diff_a: Stream[ZSet[A], T],
    diff_b: Stream[ZSet[B], T],
    pred: Callable[[A, B], bool],
    proj: Callable[[A, B], C],
    out_group: ZSetAddition[C],
    lattice: DBSPTime[T],
    outer_axis: int = 0,
    inner_axis: int = 1,
) -> Stream[ZSet[C], T]:
    """4-term doubly-incremental bilinear join on a flat product
    lattice, parameterised by ``outer_axis`` and ``inner_axis``:

        J(z⁻¹ᵒ Iᵒ a,   z⁻¹ⁱ Iⁱ b)
      + J(Iᵒ Iⁱ a,      b)
      + J(Iⁱ a,          z⁻¹ᵒ Iᵒ b)
      + J(a,             z⁻¹ⁱ Iᵒ Iⁱ b)
    """
    sg: StreamAddition[ZSet[C], T] = StreamAddition(out_group, lattice)

    int_a_o = Integrate(diff_a, lattice, axis=outer_axis)
    del_int_a_o = Delay(int_a_o, lattice, axis=outer_axis)
    int_b_o = Integrate(diff_b, lattice, axis=outer_axis)
    del_int_b_o = Delay(int_b_o, lattice, axis=outer_axis)

    int_a_i = Integrate(diff_a, lattice, axis=inner_axis)
    int_b_i = Integrate(diff_b, lattice, axis=inner_axis)
    int_a_oi = Integrate(int_a_i, lattice, axis=outer_axis)
    int_b_oi = Integrate(int_b_i, lattice, axis=outer_axis)

    del_int_b_i = Delay(int_b_i, lattice, axis=inner_axis)
    del_int_b_oi = Delay(int_b_oi, lattice, axis=inner_axis)

    term: Callable[[ZSet[A], ZSet[B]], ZSet[C]] = lambda za, zb: join(za, zb, pred, proj)

    j1 = Lift2(del_int_a_o, del_int_b_i, term, out_group)
    j2 = Lift2(int_a_oi, diff_b, term, out_group)
    j3 = Lift2(int_a_i, del_int_b_o, term, out_group)
    j4 = Lift2(diff_a, del_int_b_oi, term, out_group)

    return sg.add(sg.add(j1, j2), sg.add(j3, j4))
