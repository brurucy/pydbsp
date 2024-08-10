from typing import Dict, Generic, Iterable, Tuple, TypeVar, Callable

T = TypeVar("T")


class ZSet(Generic[T]):
    inner: Dict[T, int]

    def __init__(self, values: Dict[T, int]) -> None:
        self.inner = values

    def items(self) -> Iterable[Tuple[T, int]]:
        return self.inner.items()

    def __repr__(self) -> str:
        return self.inner.__repr__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZSet):
            return False

        return self.inner == other.inner  # type: ignore

    def __contains__(self, item: T) -> bool:
        return self.inner.__contains__(item)

    def __getitem__(self, item: T) -> int:
        if item not in self:
            return 0

        return self.inner[item]


Cmp = Callable[[T], bool]


def select[T](zset: ZSet[T], p: Cmp[T]) -> ZSet[T]:
    return ZSet({k: v for k, v in zset.items() if p(k)})


R = TypeVar("R")
Projection = Callable[[T], R]


def project[T, R](zset: ZSet[T], f: Projection[T, R]) -> ZSet[R]:
    output: Dict[R, int] = {}
    for value, weight in zset.items():
        fvalue = f(value)
        if fvalue not in output:
            output[fvalue] = weight
        else:
            output[fvalue] += weight

    return ZSet(output)


S = TypeVar("S")
JoinCmp = Callable[[T, R], bool]
PostJoinProjection = Callable[[T, R], S]


def join[T, R, S](
    left_zset: ZSet[T],
    right_zset: ZSet[R],
    p: JoinCmp[T, R],
    f: PostJoinProjection[T, R, S],
) -> ZSet[S]:
    output: Dict[S, int] = {}
    for left_value, left_weight in left_zset.items():
        for right_value, right_weight in right_zset.items():
            if p(left_value, right_value):
                projected_value = f(left_value, right_value)
                new_weight = left_weight * right_weight

                if projected_value in output:
                    output[projected_value] += new_weight
                else:
                    output[projected_value] = new_weight

    return ZSet(output)


def H[T](diff: ZSet[T], integrated_state: ZSet[T]) -> ZSet[T]:
    distincted_diff: Dict[T, int] = {}
    for k, v in diff.items():
        current_k_latest_diff_weight = v

        if k in integrated_state:
            current_k_latest_delayed_state_weight = integrated_state[k]
            coalesced_weight = current_k_latest_diff_weight + current_k_latest_delayed_state_weight

            if current_k_latest_delayed_state_weight > 0 and coalesced_weight <= 0:
                distincted_diff[k] = -1

                continue

            if current_k_latest_delayed_state_weight <= 0 and coalesced_weight > 0:
                distincted_diff[k] = 1
        else:
            if v > 0:
                distincted_diff[k] = 1

    return ZSet(distincted_diff)
