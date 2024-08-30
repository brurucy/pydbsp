from typing import Dict

from pydbsp.zset import ZSet


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
