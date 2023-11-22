# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Optional, Sequence, Union

from mmengine.fileio import get_file_backend, list_from_file
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class VideoDataset(CustomDataset):
    """A generic dataset for multiple tasks.

    The dataset supports two kinds of style.

    1. Use an annotation file to specify all samples, and each line indicates a
       sample:

       The annotation file (for ``with_label=True``, supervised tasks.): ::

            0 1
            10 1
            40 1
            128 1
            148 0
            246 0
           ...

       The annotation file (for ``with_label=False``, unsupervised tasks.): ::

            0
            10
            40
            128
            148
            246
           ...

       Sample files: ::

           data_root/
           ├── folder_1
           │   ├── folder_1.mp4
           │   └── folder_1.txt
           ├── folder_2
           │   ├── video_name.mp4
           │   └── ann_file.txt
           └── ...

       Please use the argument ``metainfo`` to specify extra information for
       the task, like ``{'classes': ('bird', 'cat', 'deer', 'dog', 'frog')}``.

    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for the data. Defaults to ''.
        ann_file (str): Annotation file path. Defaults to ''.
        with_label (bool): Whether the annotation file includes ground truth
            labels, or use sub-folders to specify categories.
            Defaults to True.
        extensions (Sequence[str]): A sequence of allowed extensions. Defaults
            to ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif').
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        lazy_init (bool): Whether to load annotation during instantiation.
            In some cases, such as visualization, only the meta information of
            the dataset is needed, which is not necessary to load annotation
            file. ``Basedataset`` can skip load annotations to save time by set
            ``lazy_init=False``. Defaults to False.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 ann_file: str,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 video_file: Optional[str] = None,
                 with_label=True,
                 extensions: Sequence[str] = ('.mp4', '.avi'),
                 metainfo: Optional[dict] = None,
                 lazy_init: bool = False,
                 **kwargs):
        assert (ann_file or data_prefix or data_root), \
            'One of `ann_file`, `data_root` and `data_prefix` must '\
            'be specified.'
        self.video_file = video_file

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            with_label=with_label,
            metainfo=metainfo,
            lazy_init=lazy_init,
            **kwargs)

    def load_data_list(self):
        """Load image paths and gt_labels."""
        if self.with_label:
            lines = list_from_file(self.ann_file)
            samples = [x.strip().rsplit(' ', 1) for x in lines]
        else:
            samples = list_from_file(self.ann_file)

        # Pre-build file backend to prevent verbose file backend inference.
        backend = get_file_backend(self.img_prefix, enable_singleton=True)
        video_file = self.video_file or Path(self.ann_file).stem + '.mp4'
        video_path = backend.join_path(self.img_prefix, video_file)
        data_list = []
        for sample in samples:
            if self.with_label:
                frame_id, gt_label = sample
                info = {
                    'video_path': video_path,
                    "frame_id": int(frame_id),
                    'gt_label': int(gt_label),
                }
            else:
                info = {'video_path': video_path, "frame_id": int(sample)}
            data_list.append(info)
        logger = MMLogger.get_current_instance()
        logger.info(f'Loaded {len(data_list)} frames from {self.ann_file}')
        return data_list
