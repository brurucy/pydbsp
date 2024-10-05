from typing import Tuple

from pydbsp.stream import LiftedGroupAdd, Stream, StreamHandle, UnaryOperator
from pydbsp.stream.operators.linear import LiftedDelay, LiftedStreamElimination, LiftedStreamIntroduction
from pydbsp.zset import ZSet
from pydbsp.zset.operators.bilinear import DeltaLiftedDeltaLiftedJoin
from pydbsp.zset.operators.unary import DeltaLiftedDeltaLiftedDistinct

Edge = Tuple[int, int]
GraphZSet = ZSet[Edge]


class IncrementalGraphReachability(UnaryOperator[GraphZSet, GraphZSet]):
    delta_input: LiftedStreamIntroduction[GraphZSet]
    join: DeltaLiftedDeltaLiftedJoin[Edge, Edge, Edge]
    delta_input_join_sum: LiftedGroupAdd[Stream[GraphZSet]]
    distinct: DeltaLiftedDeltaLiftedDistinct[Edge]
    lift_delayed_distinct: LiftedDelay[GraphZSet]
    flattened_output: LiftedStreamElimination[GraphZSet]

    def __init__(self, stream: StreamHandle[GraphZSet]):
        self.input_stream_handle = stream

        self.delta_input = LiftedStreamIntroduction(self.input_stream_handle)

        self.join = DeltaLiftedDeltaLiftedJoin(
            None,
            None,
            lambda left, right: left[1] == right[0],
            lambda left, right: (left[0], right[1]),
        )
        self.delta_input_join_sum = LiftedGroupAdd(self.delta_input.output_handle(), self.join.output_handle())

        self.distinct = DeltaLiftedDeltaLiftedDistinct(self.delta_input_join_sum.output_handle())
        self.lift_delayed_distinct = LiftedDelay(self.distinct.output_handle())

        self.join.set_input_a(self.lift_delayed_distinct.output_handle())
        self.join.set_input_b(self.delta_input.output_handle())

        self.flattened_output = LiftedStreamElimination(self.distinct.output_handle())
        self.output_stream_handle = self.flattened_output.output_handle()

    def step(self) -> bool:
        self.delta_input.step()
        self.delta_input_join_sum.step()
        self.distinct.step()
        self.lift_delayed_distinct.step()
        self.join.step()
        self.flattened_output.step()

        latest = self.flattened_output.output().latest()
        id = self.output().group().identity()
        if latest == id:
            return True

        return False
