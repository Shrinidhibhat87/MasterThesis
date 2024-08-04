"""
File that contains the Cityscapes dataloader class
Override the default torch functions
Need to override __len__() and __getitem__() methods
"""
import copy
import os
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors


class CityScapesDataset(Dataset):
    """
    CityScapes related dataloader class
    """

    METAINFO = {
        'classes': (
            'road',
            'sidewalk',
            'building',
            'wall',
            'fence',
            'pole',
            'traffic light',
            'traffic sign',
            'vegetation',
            'terrain',
            'sky',
            'person',
            'rider',
            'car',
            'truck',
            'bus',
            'train',
            'motorcycle',
            'bicycle',
        ),
        'palette': [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ],
    }
    # Because the ids would have the full list of class, we create a mapping method to
    # reduce the class size to the valid 20 classes
    mapping_20 = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 1,
        8: 2,
        9: 0,
        10: 0,
        11: 3,
        12: 4,
        13: 5,
        14: 0,
        15: 0,
        16: 0,
        17: 6,
        18: 0,
        19: 7,
        20: 8,
        21: 9,
        22: 10,
        23: 11,
        24: 12,
        25: 13,
        26: 14,
        27: 15,
        28: 16,
        29: 0,
        30: 0,
        31: 17,
        32: 18,
        33: 19,
        -1: 0,
    }

    mean_rgb = (123.675, 116.28, 103.53)
    std_rgb = (58.395, 57.12, 57.375)
    ignore_index = 255

    def __init__(
        self, root: str, transform=None, split='train', debug=False, serialize_data=True
    ) -> None:
        self.root = root
        self.transform = transform
        self.debug = debug
        self.serialize_data = serialize_data

        # Get all the images dir, which is stored in the folder called leftImg8bit
        # self.img_path = glob(os.path.join(root, "leftImg8bit", split, '**\*.png'))
        self.img_dir = os.path.join(self.root, 'leftImg8bit', split)
        # Get the segmentation mask paths
        # self.segmap_map = glob(os.path.join(root, "gtFine", split, '**\*.png'))
        self.segmap_dir = os.path.join(self.root, 'gtFine', split)

        # Get a list of all the images and mask file paths
        self.img_subdirs = sorted(os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir))
        self.mask_subdirs = sorted(
            os.path.join(self.segmap_dir, f) for f in os.listdir(self.segmap_dir)
        )

        # Because the above method will give us the sub-directory, we need to append path
        self.img_files = []
        self.mask_files = []
        for img_subdir, mask_subdir in zip(self.img_subdirs, self.mask_subdirs):
            self.img_files.extend([os.path.join(img_subdir, f) for f in os.listdir(img_subdir)])
            # Since for segmentation, the names of files are different, we replace the same
            self.mask_files.extend(
                [
                    os.path.join(mask_subdir, f.replace('leftImg8bit', 'gtFine_labelTrainIds'))
                    for f in os.listdir(img_subdir)
                ]
            )

        # Just to be sure, assert if the number of images and masks are not equal
        assert len(self.img_files) == len(self.mask_files)

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: Total length of the dataset
        """
        return len(self.img_files)

    def encode_labels(self, mask: torch.tensor) -> torch.tensor:
        """Encode the incoming labels from 33 to 20.

        Args:
            mask (torch.tensor): Mask read from the file

        Returns:
            torch.tensor: Output mask.
        """
        label_mask = torch.zeros_like(mask, dtype=torch.uint8)
        for key, value in self.mapping_20.items():
            label_mask[mask == key] = value

        return label_mask

    def __getitem__(self, index) -> Dict:
        """Method to return a sample from the dataset

        Args:
            index (int): The index of the sample to retrieve

        Returns:
            Dict: A dictionary containing the image and the segmentation mask.
        """
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        # Convert the image to torch tensor
        image = F.pil_to_tensor(Image.open(img_path))
        # image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = F.pil_to_tensor(Image.open(mask_path)).squeeze(0)
        # Convert the mask to valid format (Needed if the seg map gtFine_labelIds)
        # mask = self.encode_labels(mask)
        # mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)

        sample = {
            'image': tv_tensors.Image(image),
            'mask': tv_tensors.Mask(mask),
            'img_path': img_path,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def len(self):
        return self.img_files

    ### The below functions are from the mmlab_datasets file
    # https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/base_dataset.py
    ###########################################################################
    """
    Bug report:
        Right now we are not able to get the subset of the dataset, use the above link as reference
        to make this happen
        This is important when one wants to actually get a portion of data to create
            the LUT.
    """

    ###########################################################################
    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
              ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def _get_serialized_subset(
        self, indices: Union[Sequence[int], int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get subset of serialized data information list.

        Args:
            indices (int or Sequence[int]): If type of indices is int,
                indices represents the first or last few data of serialized
                data information list. If type of indices is Sequence, indices
                represents the target data information index which consist of
                subset data information.

        Returns:
            Tuple[np.ndarray, np.ndarray]: subset of serialized data
            information.
        """
        sub_data_bytes: Union[List, np.ndarray]
        sub_data_address: Union[List, np.ndarray]
        if isinstance(indices, int):
            if indices >= 0:
                assert indices < len(
                    self.data_address
                ), f'{indices} is out of dataset length({len(self)}'
                # Return the first few data information.
                end_addr = self.data_address[indices - 1].item() if indices > 0 else 0
                # Slicing operation of `np.ndarray` does not trigger a memory
                # copy.
                sub_data_bytes = self.data_bytes[:end_addr]
                # Since the buffer size of first few data information is not
                # changed,
                sub_data_address = self.data_address[:indices]
            else:
                assert -indices <= len(
                    self.data_address
                ), f'{indices} is out of dataset length({len(self)}'
                # Return the last few data information.
                ignored_bytes_size = self.data_address[indices - 1]
                start_addr = self.data_address[indices - 1].item()
                sub_data_bytes = self.data_bytes[start_addr:]
                sub_data_address = self.data_address[indices:]
                sub_data_address = sub_data_address - ignored_bytes_size
        elif isinstance(indices, Sequence):
            sub_data_bytes = []
            sub_data_address = []
            for idx in indices:
                assert len(self) > idx >= -len(self)
                start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
                end_addr = self.data_address[idx].item()
                # Get data information by address.
                sub_data_bytes.append(self.data_bytes[start_addr:end_addr])
                # Get data information size.
                sub_data_address.append(end_addr - start_addr)
            # Handle indices is an empty list.
            if sub_data_bytes:
                sub_data_bytes = np.concatenate(sub_data_bytes)
                sub_data_address = np.cumsum(sub_data_address)
            else:
                sub_data_bytes = np.array([])
                sub_data_address = np.array([])
        else:
            raise TypeError(
                'indices should be a int or sequence of int, ' f'but got {type(indices)}'
            )
        return sub_data_bytes, sub_data_address  # type: ignore

    def get_subset(self, indices: Union[Sequence[int], int]):
        """Return a subset of dataset.

        This method will return a subset of original dataset. If type of
        indices is int, ``get_subset_`` will return a subdataset which
        contains the first or last few data information according to
        indices is positive or negative. If type of indices is a sequence of
        int, the subdataset will extract the information according to the index
        given in indices.

        Examples:
              >>> dataset = BaseDataset('path/to/ann_file')
              >>> len(dataset)
              100
              >>> subdataset = dataset.get_subset(90)
              >>> len(sub_dataset)
              90
              >>> # if type of indices is list, extract the corresponding
              >>> # index data information
              >>> subdataset = dataset.get_subset([0, 1, 2, 3, 4, 5, 6, 7,
              >>>                                  8, 9])
              >>> len(sub_dataset)
              10
              >>> subdataset = dataset.get_subset(-3)
              >>> len(subdataset) # Get the latest few data information.
              3

        Args:
            indices (int or Sequence[int]): If type of indices is int, indices
                represents the first or last few data of dataset according to
                indices is positive or negative. If type of indices is
                Sequence, indices represents the target data information
                index of dataset.

        Returns:
            BaseDataset: A subset of dataset.
        """
        # Get subset of data from serialized data or data information list
        # according to `self.serialize_data`. Since `_get_serialized_subset`
        # will recalculate the subset data information,
        # `_copy_without_annotation` will copy all attributes except data
        # information.
        sub_dataset = self._copy_without_annotation()
        # Get subset of dataset with serialize and unserialized data.
        if self.serialize_data:
            data_bytes, data_address = self._get_serialized_subset(indices)
            sub_dataset.data_bytes = data_bytes.copy()
            sub_dataset.data_address = data_address.copy()
        else:
            data_list = self._get_unserialized_subset(indices)
            sub_dataset.data_list = copy.deepcopy(data_list)
        return sub_dataset

    def _copy_without_annotation(self, memo=Dict[str, Any]):
        """Deepcopy for all attributes other than ``data_list``,
        ``data_address`` and ``data_bytes``.

        Args:
            memo: Memory dict which used to reconstruct complex object
                correctly.
        """
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, _ in self.__dict__.items():
            if key in ['data_list', 'data_address', 'data_bytes']:
                continue
            # super(BaseDataset, other).__setattr__(key, copy.deepcopy(value, memo))

        return other

    def get_subset_(self, indices: Union[Sequence[int], int]) -> None:
        """The in-place version of ``get_subset`` to convert dataset to a
        subset of original dataset.

        This method will convert the original dataset to a subset of dataset.
        If type of indices is int, ``get_subset_`` will return a subdataset
        which contains the first or last few data information according to
        indices is positive or negative. If type of indices is a sequence of
        int, the subdataset will extract the data information according to
        the index given in indices.

        Examples:
              >>> dataset = BaseDataset('path/to/ann_file')
              >>> len(dataset)
              100
              >>> dataset.get_subset_(90)
              >>> len(dataset)
              90
              >>> # if type of indices is sequence, extract the corresponding
              >>> # index data information
              >>> dataset.get_subset_([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
              >>> len(dataset)
              10
              >>> dataset.get_subset_(-3)
              >>> len(dataset) # Get the latest few data information.
              3

        Args:
            indices (int or Sequence[int]): If type of indices is int, indices
                represents the first or last few data of dataset according to
                indices is positive or negative. If type of indices is
                Sequence, indices represents the target data information
                index of dataset.
        """
        # Get subset of data from serialized data or data information sequence
        # according to `self.serialize_data`.
        if self.serialize_data:
            self.data_bytes, self.data_address = self._get_serialized_subset(indices)
        else:
            self.data_list = self._get_unserialized_subset(indices)

    def _get_unserialized_subset(self, indices: Union[Sequence[int], int]) -> list:
        """Get subset of data information list.

        Args:
            indices (int or Sequence[int]): If type of indices is int,
                indices represents the first or last few data of data
                information. If type of indices is Sequence, indices represents
                the target data information index which consist of subset data
                information.

        Returns:
            Tuple[np.ndarray, np.ndarray]: subset of data information.
        """
        if isinstance(indices, int):
            if indices >= 0:
                # Return the first few data information.
                sub_data_list = self.data_list[:indices]
            else:
                # Return the last few data information.
                sub_data_list = self.data_list[indices:]
        elif isinstance(indices, Sequence):
            # Return the data information according to given indices.
            sub_data_list = []
            for idx in indices:
                sub_data_list.append(self.data_list[idx])
        else:
            raise TypeError(
                'indices should be a int or sequence of int, ' f'but got {type(indices)}'
            )
        return sub_data_list


def dataloader():
    root_dir = '/data/datasets/PytorchDatasets/CityScapes-pytorch'
    train_dataset = CityScapesDataset(root=root_dir, split='train', transform=None)
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    dataloader.dataset[5]


if __name__ == '__main__':
    dataloader()
