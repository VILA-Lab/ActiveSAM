"""Sequential bucket scheduler.
"""

from __future__ import annotations

from typing import List


def schedule_sequential(num_classes: int, bucket_size: int) -> List[List[int]]:
    """Split ``range(num_classes)`` into contiguous buckets of ``bucket_size``.
    """
    assert num_classes > 0 and bucket_size > 0
    return [list(range(i, min(i + bucket_size, num_classes)))
            for i in range(0, num_classes, bucket_size)]
