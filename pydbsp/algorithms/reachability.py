"""Incremental transitive-closure / reachability circuit on a flat
product lattice.

``edges`` is a 1-D ``Input`` (it varies across outer ticks, constant
within the inner fixpoint) lifted into the 2-D body lattice via
``TimeAxisIntroduction``; ``state`` is the 2-D feedback input fed
by the inner fixpoint. Axis 0 is outer; axis 1 is inner.
``saturate_reach`` drives the inner axis at a chosen outer tick.
The public ``observable`` collapses the inner axis so
``observable.at((t0,))`` yields the outer-delta of the cumulative
reachability fixpoint at outer tick ``t0``.
"""

from dataclasses import dataclass

from pydbsp.core import DBSPTime, Time1D, Time2D
from pydbsp.evaluator import Evaluator
from pydbsp.stream import Input, Stream
from pydbsp.stream.functions.linear import TimeAxisElimination, TimeAxisIntroduction
from pydbsp.stream.operators.linear import Delay
from pydbsp.stream.zset.operators.bilinear import DLDJoin
from pydbsp.stream.zset.operators.binary import DLDDistinct
from pydbsp.zset import ZSet, ZSetAddition


Edge = tuple[int, int]
Time = tuple[int, int]


@dataclass
class ReachabilityCircuit:
    observable: Stream[ZSet[Edge], tuple[int]]
    body_out: Stream[ZSet[Edge], Time]
    state: Input[ZSet[Edge], Time]
    edges: Input[ZSet[Edge], tuple[int]]
    edge_group: ZSetAddition[Edge]
    lattice: DBSPTime[Time]
    evaluator: Evaluator

    def body_at(self, t: Time) -> ZSet[Edge]:
        """Returns the per-iteration diff at the 2-D timestamp ``t``."""
        return self.evaluator.at_op(self.body_out, t)

    def observable_at(self, t: tuple[int]) -> ZSet[Edge]:
        """Returns the reachability output delta at outer timestamp ``t``."""
        return self.evaluator.at(t)


def saturate_reach(
    circuit: ReachabilityCircuit,
    outer_tick: int = 0,
    *,
    max_inner: int = 1 << 16,
) -> int:
    """Drives the inner fixpoint at ``outer_tick`` to convergence. Returns
    the final inner tick. Prior outer ticks' inner cells are padded with
    identity so inner frontiers stay aligned across outers.
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
        raise RuntimeError(f"iteration did not converge in {max_inner} inner ticks")

    for o_prior in range(outer_tick):
        for k_pad in range(prior_max + 1, k_final + 1):
            if (o_prior, k_pad) not in state._values:
                state.push((o_prior, k_pad), state.group.identity())
    return k_final


def IncrementalReachability(parallelism: int = 1) -> ReachabilityCircuit:
    """Builds an incremental graph-reachability circuit (plain nested-loop
    join). Push edges at outer ticks, call ``saturate_reach``, read the
    closure via ``observable_at``. For larger graphs, prefer
    ``IncrementalReachabilityWithIndexing``.
    """
    edge_group: ZSetAddition[Edge] = ZSetAddition()
    lattice = Time2D

    edges = Input(edge_group, Time1D)
    edges_2d = TimeAxisIntroduction(edges, edge_group, lattice, axis=1)
    state = Input(edge_group, lattice)

    hop = DLDJoin(
        Delay(state, lattice, axis=1),
        edges_2d,
        lambda l, r: l[1] == r[0],
        lambda l, r: (l[0], r[1]),
        edge_group, lattice,
    )
    fresh = hop + edges_2d
    body_out = DLDDistinct(fresh, edge_group, lattice)
    observable = TimeAxisElimination(body_out)

    evaluator = Evaluator(observable, parallelism=parallelism)
    setattr(observable, "_evaluator", evaluator)
    return ReachabilityCircuit(
        observable=observable,
        body_out=body_out,
        evaluator=evaluator,
        state=state,
        edges=edges,
        edge_group=edge_group,
        lattice=lattice,
    )
