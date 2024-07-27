from algebra import AbelianGroupOperation
from zset import Cmp, Projection, ZSet, join, JoinCmp, PostJoinProjection, project, select
from typing import TypeVar, Optional 
from stream_operators import Lifted1, Lifted2, step_n_times_and_return, Integrate, LiftedIntegrate, Delay, LiftedDelay, LiftedGroupAdd
from stream import StreamHandle, Stream
from stream_operator import BinaryOperator 

T = TypeVar("T")


class ZSetAddition(AbelianGroupOperation[ZSet[T]]):
    def add(self, a: ZSet[T], b: ZSet[T]) -> ZSet[T]:
        c = {k: v for k, v in a.inner.items() if v != 0}
        for k, v in b.inner.items():
            if k in c:
                new_weight = c[k] + v
                if new_weight != 0:
                    c[k] = new_weight
                else:
                    del c[k]
            else:
                c[k] = v

        return ZSet(c)

    def neg(self, a: ZSet[T]) -> ZSet[T]:
        return ZSet({k: v * -1 for k, v in a.inner.items()})

    def identity(self) -> ZSet[T]:
        return ZSet({})

class LiftedSelect(Lifted1[ZSet[T], ZSet[T]]):
     def __init__(
        self,
        stream: Optional[StreamHandle[ZSet[T]]],
        p: Cmp[T]
    ):
        super().__init__(stream, lambda z: select(z, p), None)

class LiftedLiftedSelect(Lifted1[Stream[ZSet[T]], Stream[ZSet[T]]]):
    def __init__(
        self,
        stream: Optional[StreamHandle[Stream[ZSet[T]]]],
        p: Cmp[T]
    ):
        super().__init__(
            stream,
            lambda x: step_n_times_and_return(
                LiftedSelect(StreamHandle(lambda: x), p),
                x.current_time() + 1
            ),
            None
        )

R = TypeVar("R")

class LiftedProject(Lifted1[ZSet[T], ZSet[R]]):
      def __init__(
        self,
        stream: Optional[StreamHandle[ZSet[T]]],
        f: Projection[T, R]
    ):
        super().__init__(stream, lambda z: project(z, f), None)

class LiftedLiftedProject(Lifted1[Stream[ZSet[T]], Stream[ZSet[R]]]):
    def __init__(
        self,
        stream: Optional[StreamHandle[Stream[ZSet[T]]]],
        f: Projection[T, R]
    ):
        super().__init__(
            stream,
            lambda x: step_n_times_and_return(
                LiftedProject(StreamHandle(lambda: x), f),
                x.current_time() + 1
            ),
            None
        )

S = TypeVar("S")

class LiftedJoin(Lifted2[ZSet[T], ZSet[R], ZSet[S]]):
    def __init__(
        self,
        stream_a: Optional[StreamHandle[ZSet[T]]],
        stream_b: Optional[StreamHandle[ZSet[R]]],
        p: JoinCmp[T, R],
        f: PostJoinProjection[T, R, S],
    ):
        super().__init__(stream_a, stream_b, lambda x, y: join(x, y, p, f), None)

class LiftedLiftedJoin(
    Lifted2[
        Stream[ZSet[T]],
        Stream[ZSet[R]],
        Stream[ZSet[S]], 
    ]
):
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
            lambda x, y: step_n_times_and_return(
                LiftedJoin(StreamHandle(lambda: x), StreamHandle(lambda: y), p, f),
                max(x.current_time(), y.current_time()) + 1,
            ),
            None,
        )

class LiftedLiftedDeltaJoin(BinaryOperator[Stream[ZSet[T]], Stream[ZSet[R]], Stream[ZSet[S]]]):
    p: JoinCmp[T, R]
    f: PostJoinProjection[T, R, S]

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

    sum_one: LiftedGroupAdd[Stream[ZSet[S]]]
    sum_two: LiftedGroupAdd[Stream[ZSet[S]]]
    sum_three: LiftedGroupAdd[Stream[ZSet[S]]]

    output_stream: Stream[ZSet[S]]

    def set_input_a(self, stream_handle_a: StreamHandle[Stream[ZSet[T]]]) -> None:
        self.input_stream_handle_a = stream_handle_a
        self.integrated_stream_a = Integrate(self.input_stream_handle_a)
        self.delayed_integrated_stream_a = Delay(
            self.integrated_stream_a.output_handle()
        )

        self.lift_integrated_stream_a = LiftedIntegrate(self.input_stream_handle_a)
        self.integrated_lift_integrated_stream_a = Integrate(
            self.lift_integrated_stream_a.output_handle()
        )

    def set_input_b(self, stream_handle_b: StreamHandle[Stream[ZSet[R]]]) -> None:
        self.input_stream_handle_b = stream_handle_b
        self.integrated_stream_b = Integrate(self.input_stream_handle_b)
        self.delayed_integrated_stream_b = Delay(
            self.integrated_stream_b.output_handle()
        )

        self.lift_integrated_stream_b = LiftedIntegrate(self.input_stream_handle_b)
        self.integrated_lift_integrated_stream_b = Integrate(
            self.lift_integrated_stream_b.output_handle()
        )
        self.lift_delayed_integrated_lift_integrated_stream_b = LiftedDelay(
            self.integrated_lift_integrated_stream_b.output_handle()
        )
        self.lift_delayed_lift_integrated_stream_b = LiftedDelay(
            self.lift_integrated_stream_b.output_handle()
        )

        self.join_1 = LiftedLiftedJoin[T, R, S](
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

        self.sum_one = LiftedGroupAdd(
            self.join_1.output_handle(), self.join_2.output_handle()
        )
        self.sum_two = LiftedGroupAdd(
            self.sum_one.output_handle(), self.join_3.output_handle()
        )
        self.sum_three = LiftedGroupAdd(
            self.sum_two.output_handle(), self.join_4.output_handle()
        )

        self.set_output_stream(self.sum_three.output_handle())

    def __init__(
        self,
        diff_stream_a: Optional[StreamHandle[Stream[ZSet[T]]]],
        diff_stream_b: Optional[StreamHandle[Stream[ZSet[R]]]],
        p: JoinCmp[T, R],
        f: PostJoinProjection[T, R, S],
    ):
        self.p = p
        self.f = f

        if diff_stream_a is not None:
            self.set_input_a(diff_stream_a)

        if diff_stream_b is not None:
            self.set_input_b(diff_stream_b)

    def step(self) -> bool:
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

        self.sum_one.step()
        self.sum_two.step()
        self.sum_three.step()

        return True
