from typing import Dict

from pydbsp.zset import ZSet


def H[T](diff: ZSet[T], integrated_state: ZSet[T]) -> ZSet[T]:
    """
    Computes a "distincted" difference between a diff and an integrated state.

    Args:
        diff (ZSet[T]): The difference ZSet, representing changes.
        integrated_state (ZSet[T]): The current integrated state ZSet.

    Returns:
        ZSet[T]: A new ZSet representing the "distincted" difference.

    Behavior:
    1. For each element k in the diff:
       a. If k is not in the integrated_state and has a positive weight in diff,
          it's added to the result with weight 1.
       b. If k is in the integrated_state:
          - If its weight becomes non-positive after applying the diff and was
            previously positive, it's added to the result with weight -1.
          - If its weight becomes positive after applying the diff and was
            previously non-positive, it's added to the result with weight 1.
    2. All other cases result in the element not being included in the output.

    This function effectively tracks which elements are being added to or removed from
    a distinct set representation, based on the integrated state and the incoming changes.
    """
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
