from typing import Dict

from pydbsp.lazy_zset import LazyZSet
from pydbsp.zset import ZSet


def H[T](diff: LazyZSet[T], integrated_state: LazyZSet[T]) -> LazyZSet[T]:
    distincted_diff: Dict[T, int] = {}

    jiff = diff.coalesce()
    # for diff_zset in jiff:
    for k, v in jiff.items():
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

    return LazyZSet([ZSet(distincted_diff)])
