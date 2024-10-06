from pydbsp.lazy_zset import LazyZSet
from pydbsp.zset.functions.bilinear import JoinCmp, PostJoinProjection
from pydbsp.zset.functions.bilinear import join as zset_join


def join[T, R, S](
    left_zset: LazyZSet[T],
    right_zset: LazyZSet[R],
    p: JoinCmp[T, R],
    f: PostJoinProjection[T, R, S],
) -> LazyZSet[S]:
    zset_product = [(left_df, right_df) for left_df in left_zset.inner for right_df in right_zset.inner]

    result = [zset_join(left_zset, right_zset, p, f) for left_zset, right_zset in zset_product]

    return LazyZSet(result)
