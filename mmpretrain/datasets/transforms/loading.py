import os
import os.path as osp
import numpy as np
import mmcv
from mmcv.transforms import BaseTransform
from mmengine.fileio import FileClient
from mmpretrain.registry import TRANSFORMS

from .utils import get_random_string, get_shm_dir, get_thread_id


@TRANSFORMS.register_module()
class LoadImageFromVideo(BaseTransform):
    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = "color",
        channel_order: str = "rgb",
        io_backend: str = "disk",
        ignore_empty: bool = False,
        **kwargs,
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.io_backend = io_backend
        self.ignore_empty = ignore_empty
        self.kwargs = kwargs

        self.file_client = None
        self.tmp_folder = None
        if self.io_backend != "disk":
            random_string = get_random_string()
            thread_id = get_thread_id()
            self.tmp_folder = osp.join(get_shm_dir(), f"{random_string}_{thread_id}")
            os.mkdir(self.tmp_folder)

    def transform(self, results: dict):
        """Load image from video.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: ``results`` will be updated with new key "img" and
                ``img_shape`` after loading.
        """

        if self.io_backend == "disk":
            new_path = results["video_path"]
        else:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend, **self.kwargs)

            thread_id = get_thread_id()
            # save the file of same thread at the same place
            new_path = osp.join(self.tmp_folder, f"tmp_{thread_id}.mp4")
            with open(new_path, "wb") as f:
                f.write(self.file_client.get(results["video_path"]))

        frame_id = results["frame_id"]
        try:
            img = mmcv.VideoReader(new_path)[frame_id]
        except Exception as e:
            if self.ignore_empty:
                return
            else:
                raise e

        if self.color_type == "grayscale":
            img = mmcv.bgr2gray(img)
        elif self.color_type == "color" and self.channel_order == "rgb":
            img = img[..., ::-1]

        if self.to_float32:
            img = img.astype(np.float32)

        results["img"] = img
        results["ori_shape"] = img.shape[:2]
        results["img_shape"] = img.shape[:2]
        return results
