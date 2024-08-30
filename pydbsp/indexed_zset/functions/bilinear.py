from itertools import product

from pydbsp.indexed_zset import IndexedZSet, sort_merge_join
from pydbsp.zset import ZSet
from pydbsp.zset.functions.bilinear import PostJoinProjection


def join_with_index[T, I1, R, S](
    left_indexed_zset: IndexedZSet[T, I1],
    right_indexed_zset: IndexedZSet[R, I1],
    f: PostJoinProjection[T, R, S],
) -> ZSet[S]:
    output: ZSet[S] = ZSet({})

    for match in sort_merge_join(left_indexed_zset.index, right_indexed_zset.index):
        left_values = [(value, left_indexed_zset.inner[value]) for value in left_indexed_zset.index_to_value[match]]
        right_values = [(value, right_indexed_zset.inner[value]) for value in right_indexed_zset.index_to_value[match]]
        left_x_right = product(left_values, right_values)

        projected_values = [
            (f(left_value, right_value), left_weight * right_weight)
            for ((left_value, left_weight), (right_value, right_weight)) in left_x_right
        ]

        for projected_value, new_weight in projected_values:
            if projected_value in output:
                output[projected_value] += new_weight
            else:
                output[projected_value] = new_weight

    return output
