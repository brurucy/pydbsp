from typing import Callable, Dict, TypeVar

from pydbsp.zset import ZSet

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")
JoinCmp = Callable[[T, R], bool]
PostJoinProjection = Callable[[T, R], S]


def join[T, R, S](
    left_zset: ZSet[T],
    right_zset: ZSet[R],
    p: JoinCmp[T, R],
    f: PostJoinProjection[T, R, S],
) -> ZSet[S]:
    """
    Joins two ZSets. Implemented as a nested loop join.

    Args:
       left_zset
       right_zset
       p: Join key function
       f: projection to be applied to the join
    """
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
