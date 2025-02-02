from typing import Optional, TypeVar

from pydbsp.stream import (
    BinaryOperator,
    Lift2,
    Stream,
    StreamAddition,
    StreamHandle,
    step_until_fixpoint_and_return,
)
from pydbsp.stream.operators.linear import Delay, Integrate, LiftedDelay, LiftedIntegrate
from pydbsp.stream import LiftedGroupAdd, LiftedGroupNegate
from pydbsp.zset import ZSet, ZSetAddition
from pydbsp.zset.functions.bilinear import JoinCmp, PostJoinProjection, join

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")


class LiftedJoin(Lift2[ZSet[T], ZSet[R], ZSet[S]]):
    def __init__(
        self,
        stream_a: Optional[StreamHandle[ZSet[T]]],
        stream_b: Optional[StreamHandle[ZSet[R]]],
        p: JoinCmp[T, R],
        f: PostJoinProjection[T, R, S],
    ):
        super().__init__(stream_a, stream_b, lambda x, y: join(x, y, p, f), None)


class LiftedLiftedJoin(
    Lift2[
        Stream[ZSet[T]],
        Stream[ZSet[R]],
        Stream[ZSet[S]],
    ]
):
    """
    Computes the Z-Set join between two streams element-wise.
    """

    def __init__(
        self,
        stream_a: Optional[StreamHandle[Stream[ZSet[T]]]],
        stream_b: Optional[StreamHandle[Stream[ZSet[R]]]],
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


class DeltaLiftedDeltaLiftedJoin(BinaryOperator[Stream[ZSet[T]], Stream[ZSet[R]], Stream[ZSet[S]]]):
    """
    Incrementally computes the Z-Set join between two streams element-wise. Equivalent to - but keeps less state - incrementalizing a doubly-lifted join. See :func:`~pydbsp.stream.operators.Incrementalize2` to grasp what it means to incrementalize a singly-lifted join.
    """

    p: JoinCmp[T, R]
    f: PostJoinProjection[T, R, S]
    frontier_a: int
    frontier_b: int

    integrated_stream_a: Integrate[Stream[ZSet[T]]]
    delayed_integrated_stream_a: Delay[Stream[ZSet[T]]]
    lift_integrated_stream_a: LiftedIntegrate[ZSet[T]]
    integrated_lift_integrated_stream_a: Integrate[Stream[ZSet[T]]]

    integrated_stream_b: Integrate[Stream[ZSet[R]]]
    delayed_integrated_stream_b: Delay[Stream[ZSet[R]]]
    lift_integrated_stream_b: LiftedIntegrate[ZSet[R]]
    integrated_lift_integrated_stream_b: Integrate[Stream[ZSet[R]]]
    lift_delayed_integrated_lift_integrated_stream_b: LiftedDelay[ZSet[R]]
    lift_delayed_lift_integrated_stream_b: LiftedDelay[ZSet[R]]

    join_1: LiftedLiftedJoin[T, R, S]
    join_2: LiftedLiftedJoin[T, R, S]
    join_3: LiftedLiftedJoin[T, R, S]
    join_4: LiftedLiftedJoin[T, R, S]

    output_stream: Stream[Stream[ZSet[S]]]

    def set_input_a(self, stream_handle_a: StreamHandle[Stream[ZSet[T]]]) -> None:
        self.input_stream_handle_a = stream_handle_a
        self.integrated_stream_a = Integrate(self.input_stream_handle_a)
        self.delayed_integrated_stream_a = Delay(self.integrated_stream_a.output_handle())

        self.lift_integrated_stream_a = LiftedIntegrate(self.input_stream_handle_a)
        self.integrated_lift_integrated_stream_a = Integrate(self.lift_integrated_stream_a.output_handle())

    def set_input_b(self, stream_handle_b: StreamHandle[Stream[ZSet[R]]]) -> None:
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
        diff_stream_a: Optional[StreamHandle[Stream[ZSet[T]]]],
        diff_stream_b: Optional[StreamHandle[Stream[ZSet[R]]]],
        p: JoinCmp[T, R],
        f: PostJoinProjection[T, R, S],
    ):
        self.p = p
        self.f = f
        self.frontier_a = 0
        self.frontier_b = 0
        inner_group = ZSetAddition[S]()
        group: StreamAddition[ZSet[S]] = StreamAddition(inner_group)  # type: ignore

        self.output_stream = Stream(group)
        self.output_stream_handle = StreamHandle(lambda: self.output_stream)

        if diff_stream_a is not None:
            self.set_input_a(diff_stream_a)

        if diff_stream_b is not None:
            self.set_input_b(diff_stream_b)

    def output(self) -> Stream[Stream[ZSet[S]]]:
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
        sum_3 = group.add(sum_1, sum_2)
        self.output_stream.send(sum_3)

        self.frontier_a += 1
        self.frontier_b += 1

        return False