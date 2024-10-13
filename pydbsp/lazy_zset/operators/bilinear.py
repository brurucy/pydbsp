from typing import Optional

from pydbsp.lazy_zset import LazyZSet, LazyZSetAddition
from pydbsp.lazy_zset.functions.bilinear import join
from pydbsp.stream import (
    BinaryOperator,
    Lift2,
    Stream,
    StreamAddition,
    StreamHandle,
    step_until_fixpoint_and_return,
)
from pydbsp.stream.operators.linear import Delay, Integrate, LiftedDelay, LiftedIntegrate
from pydbsp.zset.functions.bilinear import JoinCmp, PostJoinProjection


class LiftedJoin[T, R, S](Lift2[LazyZSet[T], LazyZSet[R], LazyZSet[S]]):
    def __init__(
        self,
        stream_a: Optional[StreamHandle[LazyZSet[T]]],
        stream_b: Optional[StreamHandle[LazyZSet[R]]],
        p: JoinCmp[T, R],
        f: PostJoinProjection[T, R, S],
    ):
        super().__init__(stream_a, stream_b, lambda x, y: join(x, y, p, f), None)


class LiftedLiftedJoin[T, R, S](
    Lift2[
        Stream[LazyZSet[T]],
        Stream[LazyZSet[R]],
        Stream[LazyZSet[S]],
    ]
):
    """
    Computes the Lazy Z-Set join between two streams element-wise.
    """

    def __init__(
        self,
        stream_a: Optional[StreamHandle[Stream[LazyZSet[T]]]],
        stream_b: Optional[StreamHandle[Stream[LazyZSet[R]]]],
        p: JoinCmp[T, R],
        f: PostJoinProjection[T, R, S],
    ):
        super().__init__(
            stream_a,
            stream_b,
            lambda x, y: step_until_fixpoint_and_return(
                LiftedJoin(StreamHandle(lambda: x), StreamHandle(lambda: y), p, f)
            ),
            None,
        )


class DeltaLiftedDeltaLiftedJoin[T, R, S](
    BinaryOperator[Stream[LazyZSet[T]], Stream[LazyZSet[R]], Stream[LazyZSet[S]]]
):
    """
    Incrementally computes the Lazy Z-Set join between two streams element-wise. Equivalent to - but keeps less state - incrementalizing a doubly-lifted join. See :func:`~pydbsp.stream.operators.Incrementalize2` to grasp what it means to incrementalize a singly-lifted join.
    """

    p: JoinCmp[T, R]
    f: PostJoinProjection[T, R, S]
    frontier_a: int
    frontier_b: int

    integrated_stream_a: Integrate[Stream[LazyZSet[T]]]
    delayed_integrated_stream_a: Delay[Stream[LazyZSet[T]]]
    lift_integrated_stream_a: LiftedIntegrate[LazyZSet[T]]
    integrated_lift_integrated_stream_a: Integrate[Stream[LazyZSet[T]]]

    integrated_stream_b: Integrate[Stream[LazyZSet[R]]]
    delayed_integrated_stream_b: Delay[Stream[LazyZSet[R]]]
    lift_integrated_stream_b: LiftedIntegrate[LazyZSet[R]]
    integrated_lift_integrated_stream_b: Integrate[Stream[LazyZSet[R]]]
    lift_delayed_integrated_lift_integrated_stream_b: LiftedDelay[LazyZSet[R]]
    lift_delayed_lift_integrated_stream_b: LiftedDelay[LazyZSet[R]]

    join_1: LiftedLiftedJoin[T, R, S]
    join_2: LiftedLiftedJoin[T, R, S]
    join_3: LiftedLiftedJoin[T, R, S]
    join_4: LiftedLiftedJoin[T, R, S]

    output_stream: Stream[Stream[LazyZSet[S]]]

    def set_input_a(self, stream_handle_a: StreamHandle[Stream[LazyZSet[T]]]) -> None:
        self.input_stream_handle_a = stream_handle_a
        self.integrated_stream_a = Integrate(self.input_stream_handle_a)
        self.delayed_integrated_stream_a = Delay(self.integrated_stream_a.output_handle())

        self.lift_integrated_stream_a = LiftedIntegrate(self.input_stream_handle_a)
        self.integrated_lift_integrated_stream_a = Integrate(self.lift_integrated_stream_a.output_handle())

    def set_input_b(self, stream_handle_b: StreamHandle[Stream[LazyZSet[R]]]) -> None:
        self.input_stream_handle_b = stream_handle_b
        self.integrated_stream_b = Integrate(self.input_stream_handle_b)
        self.delayed_integrated_stream_b = Delay(self.integrated_stream_b.output_handle())

        self.lift_integrated_stream_b = LiftedIntegrate(self.input_stream_handle_b)
        self.integrated_lift_integrated_stream_b = Integrate(self.lift_integrated_stream_b.output_handle())
        self.lift_delayed_integrated_lift_integrated_stream_b = LiftedDelay(
            self.integrated_lift_integrated_stream_b.output_handle()
        )
        self.lift_delayed_lift_integrated_stream_b = LiftedDelay(self.lift_integrated_stream_b.output_handle())

        self.join_1 = LiftedLiftedJoin(
            self.delayed_integrated_stream_a.output_handle(),
            self.lift_delayed_lift_integrated_stream_b.output_handle(),
            self.p,
            self.f,
        )
        self.join_2 = LiftedLiftedJoin(
            self.integrated_lift_integrated_stream_a.output_handle(),
            self.input_stream_handle_b,
            self.p,
            self.f,
        )
        self.join_3 = LiftedLiftedJoin(
            self.lift_integrated_stream_a.output_handle(),
            self.delayed_integrated_stream_b.output_handle(),
            self.p,
            self.f,
        )
        self.join_4 = LiftedLiftedJoin(
            self.input_stream_handle_a,
            self.lift_delayed_integrated_lift_integrated_stream_b.output_handle(),
            self.p,
            self.f,
        )

    def __init__(
        self,
        diff_stream_a: Optional[StreamHandle[Stream[LazyZSet[T]]]],
        diff_stream_b: Optional[StreamHandle[Stream[LazyZSet[R]]]],
        p: JoinCmp[T, R],
        f: PostJoinProjection[T, R, S],
    ):
        self.p = p
        self.f = f
        self.frontier_a = 0
        self.frontier_b = 0
        inner_group: LazyZSetAddition[S] = LazyZSetAddition()
        group: StreamAddition[LazyZSet[S]] = StreamAddition(inner_group)  # type: ignore

        self.output_stream = Stream(group)
        self.output_stream_handle = StreamHandle(lambda: self.output_stream)

        if diff_stream_a is not None:
            self.set_input_a(diff_stream_a)

        if diff_stream_b is not None:
            self.set_input_b(diff_stream_b)

    def output(self) -> Stream[Stream[LazyZSet[S]]]:
        return self.output_stream

    def step(self) -> bool:
        current_a_timestamp = self.input_a().current_time()
        current_b_timestamp = self.input_b().current_time()
        # Not sure about this.
        if current_a_timestamp == self.frontier_a and current_b_timestamp == self.frontier_b:
            return True

        self.integrated_stream_a.step()
        self.delayed_integrated_stream_a.step()
        self.lift_integrated_stream_a.step()
        self.integrated_lift_integrated_stream_a.step()
        self.integrated_stream_b.step()
        self.delayed_integrated_stream_b.step()
        self.lift_integrated_stream_b.step()
        self.integrated_lift_integrated_stream_b.step()
        self.lift_delayed_integrated_lift_integrated_stream_b.step()
        self.lift_delayed_lift_integrated_stream_b.step()
        self.join_1.step()
        self.join_2.step()
        self.join_3.step()
        self.join_4.step()

        group = self.output().group()
        sum_1 = group.add(self.join_1.output().latest(), self.join_2.output().latest())
        sum_2 = group.add(self.join_3.output().latest(), self.join_4.output().latest())
        self.output_stream.send(group.add(sum_1, sum_2))

        self.frontier_a += 1
        self.frontier_b += 1

        return False
