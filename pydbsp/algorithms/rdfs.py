"""Incremental RDFS materialization on the flat 2-D product lattice.

RDFS is a constrained Datalog program — we bake its fixed rule set
into the circuit rather than going through the general Datalog
reasoner. State is a single ``ZSet[RDFTuple]`` accumulating the
materialized ABox plus the TBox closures (SCO and SPO transitive
chains), distinguished by predicate position (SCO/SPO/TYPE codes).

Rules encoded:
    SCO TC:   (x, SCO, z) :- (x, SCO, y), (y, SCO, z)
    SPO TC:   (x, SPO, z) :- (x, SPO, y), (y, SPO, z)
    Property: (x, b, y) :- (a, SPO, b), (x, a, y)
    Class TC: (z, TYPE, y) :- (x, SCO, y), (z, TYPE, x)
    Domain:   (y, TYPE, x) :- (a, DOMAIN, x), (y, a, z)
    Range:    (z, TYPE, x) :- (a, RANGE, x), (y, a, z)

Flat 2-D lattice: axis 0 = outer (input progression), axis 1 =
inner (fixpoint iteration). ``state`` is a 2-D ``Input``;
``abox`` / ``tbox`` are 1-D outer inputs bridged to 2-D via
``TimeAxisIntroduction`` with ω-frontier on inner.
"""

from dataclasses import dataclass
from typing import Any

from pydbsp.core import DBSPTime, Time1D, Time2D
from pydbsp.evaluator import Evaluator
from pydbsp.stream import Input, Stream
from pydbsp.stream.functions.linear import (
    TimeAxisElimination,
    TimeAxisIntroduction,
)
from pydbsp.stream.operators.linear import Delay
from pydbsp.stream.zset.operators.bilinear import DLDJoin
from pydbsp.stream.zset.operators.binary import DLDDistinct
from pydbsp.stream.zset.operators.linear import Select
from pydbsp.zset import ZSet, ZSetAddition


Subject = Any
Object = Any
Property = Any
RDFTuple = tuple[Subject, Property, Object]
RDFGraph = ZSet[RDFTuple]

SCO = 0
SPO = 1
DOMAIN = 2
RANGE = 3
TYPE = 4

Time = tuple[int, int]


@dataclass
class RDFSCircuit:
    observable: Stream[RDFGraph, tuple[int]]
    body_out: Stream[RDFGraph, Time]
    state: Input[RDFGraph, Time]
    abox: Input[RDFGraph, tuple[int]]
    tbox: Input[RDFGraph, tuple[int]]
    tuple_group: ZSetAddition[RDFTuple]
    lattice: DBSPTime[Time]
    evaluator: Evaluator

    def body_at(self, t: Time) -> RDFGraph:
        """Returns the per-iteration diff at the 2-D timestamp ``t``."""
        return self.evaluator.at_op(self.body_out, t)

    def observable_at(self, t: tuple[int]) -> RDFGraph:
        """Returns the RDFS materialization delta at outer ``t``."""
        return self.evaluator.at(t)


def saturate_rdfs(
    circuit: RDFSCircuit,
    outer_tick: int = 0,
    *,
    max_inner: int = 1 << 16,
) -> int:
    """Drives the inner fixpoint at ``outer_tick`` to convergence. Returns
    the final inner tick.
    """
    state = circuit.state
    prior_max = -1
    for (o, k) in state._values.keys():
        if o < outer_tick and k > prior_max:
            prior_max = k

    k_final = -1
    for k in range(max_inner):
        diff = circuit.body_at((outer_tick, k))
        is_zero = diff.inner == {}
        state.push((outer_tick, k), diff)
        if is_zero and k > prior_max:
            k_final = k
            break
    else:
        raise RuntimeError(f"RDFS did not converge in {max_inner} inner ticks")
    return k_final


def IncrementalRDFSMaterialization(parallelism: int = 1) -> RDFSCircuit:
    """Builds an RDFS materialization circuit. Push ``abox`` (data) and
    ``tbox`` (schema) tuples at outer ticks, call ``saturate_rdfs``,
    read the materialization via ``observable_at``. For larger graphs,
    prefer ``IncrementalRDFSMaterializationWithIndexing``.
    """
    tuple_group: ZSetAddition[RDFTuple] = ZSetAddition()
    lattice = Time2D

    abox = Input(tuple_group, Time1D)
    tbox = Input(tuple_group, Time1D)

    abox_2d = TimeAxisIntroduction(abox, tuple_group, lattice, axis=1)
    tbox_2d = TimeAxisIntroduction(tbox, tuple_group, lattice, axis=1)

    tbox_domain = Select(tbox_2d, lambda t: t[1] == DOMAIN)
    tbox_range = Select(tbox_2d, lambda t: t[1] == RANGE)

    state = Input(tuple_group, lattice)
    delay_state = Delay(state, lattice, axis=1)

    state_sco = Select(delay_state, lambda t: t[1] == SCO)
    state_spo = Select(delay_state, lambda t: t[1] == SPO)
    state_type = Select(delay_state, lambda t: t[1] == TYPE)
    state_non_type = Select(
        delay_state,
        lambda t: t[1] not in (SCO, SPO, TYPE, DOMAIN, RANGE),
    )

    j = lambda a, b, pred, proj: DLDJoin(a, b, pred, proj, tuple_group, lattice)
    sco_tc = j(state_sco, state_sco,
               lambda l, r: l[2] == r[0],
               lambda l, r: (l[0], SCO, r[2]))
    spo_tc = j(state_spo, state_spo,
               lambda l, r: l[2] == r[0],
               lambda l, r: (l[0], SPO, r[2]))
    prop = j(state_spo, state_non_type,
             lambda s, a: s[0] == a[1],
             lambda s, a: (a[0], s[2], a[2]))
    class_sco = j(state_sco, state_type,
                  lambda s, ty: s[0] == ty[2],
                  lambda s, ty: (ty[0], TYPE, s[2]))
    domain = j(tbox_domain, state_non_type,
               lambda d, a: d[0] == a[1],
               lambda d, a: (a[0], TYPE, d[2]))
    range_ = j(tbox_range, state_non_type,
               lambda r, a: r[0] == a[1],
               lambda r, a: (a[2], TYPE, r[2]))

    fresh = sco_tc + spo_tc + prop + class_sco + domain + range_
    seeded = fresh + tbox_2d + abox_2d
    body_out = DLDDistinct(seeded, tuple_group, lattice)
    observable = TimeAxisElimination(body_out)

    evaluator = Evaluator(observable, parallelism=parallelism)
    setattr(observable, "_evaluator", evaluator)
    return RDFSCircuit(
        observable=observable,
        body_out=body_out,
        evaluator=evaluator,
        state=state,
        abox=abox,
        tbox=tbox,
        tuple_group=tuple_group,
        lattice=lattice,
    )
