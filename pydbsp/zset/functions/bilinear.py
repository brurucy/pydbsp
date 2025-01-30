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

def anti_join[T, R](
    left_zset: ZSet[T],
    right_zset: ZSet[R],
    p: JoinCmp[T, R],
) -> ZSet[T]:
    """
    Performs an anti-join between two ZSets. Returns elements from left_zset
    that do not match any elements in right_zset according to predicate p.

    Args:
        left_zset: The left ZSet to anti-join
        right_zset: The right ZSet to anti-join against
        p: Join predicate function that returns True if elements match
    """
    output: Dict[T, int] = {}
    for left_value, left_weight in left_zset.items():
        final_left_weight = left_weight
        for right_value, right_weight in right_zset.items():
            if p(left_value, right_value):
                final_left_weight -= right_weight 
        
        output[left_value] = final_left_weight

    return ZSet(output)