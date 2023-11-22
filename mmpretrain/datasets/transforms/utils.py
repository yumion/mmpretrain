# Copyright (c) OpenMMLab. All rights reserved.
import copy
import ctypes
import random
import string
from typing import List, Union

from mmcv.transforms import BaseTransform

PIPELINE_TYPE = List[Union[dict, BaseTransform]]


def get_transform_idx(pipeline: PIPELINE_TYPE, target: str) -> int:
    """Returns the index of the transform in a pipeline.

    Args:
        pipeline (List[dict] | List[BaseTransform]): The transforms list.
        target (str): The target transform class name.

    Returns:
        int: The transform index. Returns -1 if not found.
    """
    for i, transform in enumerate(pipeline):
        if isinstance(transform, dict):
            if isinstance(transform["type"], type):
                if transform["type"].__name__ == target:
                    return i
            else:
                if transform["type"] == target:
                    return i
        else:
            if transform.__class__.__name__ == target:
                return i

    return -1


def remove_transform(pipeline: PIPELINE_TYPE, target: str, inplace=False):
    """Remove the target transform type from the pipeline.

    Args:
        pipeline (List[dict] | List[BaseTransform]): The transforms list.
        target (str): The target transform class name.
        inplace (bool): Whether to modify the pipeline inplace.

    Returns:
        The modified transform.
    """
    idx = get_transform_idx(pipeline, target)
    if not inplace:
        pipeline = copy.deepcopy(pipeline)
    while idx >= 0:
        pipeline.pop(idx)
        idx = get_transform_idx(pipeline, target)

    return pipeline


def get_random_string(length: int = 15) -> str:
    """Get random string with letters and digits.

    Args:
        length (int): Length of random string. Defaults to 15.
    """
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def get_thread_id() -> int:
    """Get current thread id."""
    # use ctype to find thread id
    thread_id = ctypes.CDLL("libc.so.6").syscall(186)
    return thread_id


def get_shm_dir() -> str:
    """Get shm dir for temporary usage."""
    return "/dev/shm"
