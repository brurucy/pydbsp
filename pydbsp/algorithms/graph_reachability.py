from typing import Tuple

from pydbsp.indexed_zset.operators.bilinear import DeltaLiftedDeltaLiftedSortMergeJoin
from pydbsp.indexed_zset.operators.linear import LiftedLiftedIndex
from pydbsp.lazy_zset import LazyZSet
from pydbsp.lazy_zset.operators.bilinear import DeltaLiftedDeltaLiftedJoin as LazyZSetDeltaLiftedDeltaLiftedJoin
from pydbsp.lazy_zset.operators.unary import DeltaLiftedDeltaLiftedDistinct as LazyZSetDeltaLiftedDeltaLiftedDistinct
from pydbsp.stream import LiftedGroupAdd, Stream, StreamHandle, UnaryOperator
from pydbsp.stream.functions.linear import stream_elimination
from pydbsp.stream.operators.linear import LiftedDelay, LiftedStreamElimination, LiftedStreamIntroduction
from pydbsp.zset import ZSet
from pydbsp.zset.operators.bilinear import DeltaLiftedDeltaLiftedJoin as ZSetDeltaLiftedDeltaLiftedJoin
from pydbsp.zset.operators.unary import DeltaLiftedDeltaLiftedDistinct as ZSetDeltaLiftedDeltaLiftedDistinct

Edge = Tuple[int, int]
GraphZSet = ZSet[Edge]


class IncrementalGraphReachability(UnaryOperator[GraphZSet, GraphZSet]):
    delta_input: LiftedStreamIntroduction[GraphZSet]
    join: ZSetDeltaLiftedDeltaLiftedJoin[Edge, Edge, Edge]
    delta_input_join_sum: LiftedGroupAdd[Stream[GraphZSet]]
    distinct: ZSetDeltaLiftedDeltaLiftedDistinct[Edge]
    lift_delayed_distinct: LiftedDelay[GraphZSet]
    flattened_output: LiftedStreamElimination[GraphZSet]

    def __init__(self, stream: StreamHandle[GraphZSet]):
        self.input_stream_handle = stream

        self.delta_input = LiftedStreamIntroduction(self.input_stream_handle)

        self.join = ZSetDeltaLiftedDeltaLiftedJoin(
            None,
            None,
            lambda left, right: left[1] == right[0],
            lambda left, right: (left[0], right[1]),
        )
        self.delta_input_join_sum = LiftedGroupAdd(self.delta_input.output_handle(), self.join.output_handle())

        self.distinct = ZSetDeltaLiftedDeltaLiftedDistinct(self.delta_input_join_sum.output_handle())
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
        print(f"Without indexing ldd: {stream_elimination(self.lift_delayed_distinct.output().latest())}")
        self.join.step()
        print(f"Without indexing Join: {stream_elimination(self.join.output().latest())}")

        self.flattened_output.step()

        latest = self.flattened_output.output().latest()
        id = self.output().group().identity()
        if latest == id:
            return True

        return False


class IndexedIncrementalGraphReachability(UnaryOperator[GraphZSet, GraphZSet]):
    delta_input: LiftedStreamIntroduction[GraphZSet]
    join: DeltaLiftedDeltaLiftedSortMergeJoin[int, Edge, Edge, Edge]
    delta_input_join_sum: LiftedGroupAdd[Stream[GraphZSet]]
    distinct: ZSetDeltaLiftedDeltaLiftedDistinct[Edge]
    lift_delayed_distinct: LiftedDelay[GraphZSet]
    flattened_output: LiftedStreamElimination[GraphZSet]

    def __init__(self, stream: StreamHandle[GraphZSet]):
        self.input_stream_handle = stream

        self.delta_input = LiftedStreamIntroduction(self.input_stream_handle)
        self.delta_input_by_fst = LiftedLiftedIndex(self.delta_input.output_handle(), lambda edge: edge[0])

        self.join = DeltaLiftedDeltaLiftedSortMergeJoin(
            None,
            None,
            lambda key, left, right: (left[0], right[1]),
        )
        self.delta_input_join_sum = LiftedGroupAdd(self.delta_input.output_handle(), self.join.output_handle())

        self.distinct = ZSetDeltaLiftedDeltaLiftedDistinct(self.delta_input_join_sum.output_handle())
        self.lift_delayed_distinct = LiftedDelay(self.distinct.output_handle())
        self.lift_delayed_distinct_by_snd = LiftedLiftedIndex(
            self.lift_delayed_distinct.output_handle(), lambda edge: edge[1]
        )

        self.join.set_input_a(self.lift_delayed_distinct_by_snd.output_handle())
        self.join.set_input_b(self.delta_input_by_fst.output_handle())

        self.flattened_output = LiftedStreamElimination(self.distinct.output_handle())
        self.output_stream_handle = self.flattened_output.output_handle()

    def step(self) -> bool:
        self.delta_input.step()
        self.delta_input_by_fst.step()
        self.delta_input_join_sum.step()
        self.distinct.step()
        self.lift_delayed_distinct.step()
        self.lift_delayed_distinct_by_snd.step()
        print(f"With indexing ldd: {stream_elimination(self.lift_delayed_distinct.output().latest())}")
        print(
            f"With indexing ldd snd: {stream_elimination(self.lift_delayed_distinct_by_snd.output().latest()).index._lists}"
        )
        self.join.step()
        print(f"With indexing Join: {stream_elimination(self.join.output().latest())}")
        self.flattened_output.step()

        latest = self.flattened_output.output().latest()
        output_id = self.output().group().identity()
        if latest == output_id:
            return True

        return False


LazyGraphZSet = LazyZSet[Edge]


class LazyIncrementalGraphReachability(UnaryOperator[LazyGraphZSet, LazyGraphZSet]):
    delta_input: LiftedStreamIntroduction[LazyGraphZSet]
    join: LazyZSetDeltaLiftedDeltaLiftedJoin[Edge, Edge, Edge]
    delta_input_join_sum: LiftedGroupAdd[Stream[LazyGraphZSet]]
    distinct: LazyZSetDeltaLiftedDeltaLiftedDistinct[Edge]
    lift_delayed_distinct: LiftedDelay[LazyGraphZSet]
    flattened_output: LiftedStreamElimination[LazyGraphZSet]

    def __init__(self, stream: StreamHandle[LazyGraphZSet]):
        self.input_stream_handle = stream

        self.delta_input = LiftedStreamIntroduction(self.input_stream_handle)

        self.join = LazyZSetDeltaLiftedDeltaLiftedJoin(
            None,
            None,
            lambda left, right: left[1] == right[0],
            lambda left, right: (left[0], right[1]),
        )
        self.delta_input_join_sum = LiftedGroupAdd(self.delta_input.output_handle(), self.join.output_handle())

        self.distinct = LazyZSetDeltaLiftedDeltaLiftedDistinct(self.delta_input_join_sum.output_handle())
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
