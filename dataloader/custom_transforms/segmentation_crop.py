import typing
from typing import Any, Dict, List, Literal

import numpy as np
import torchvision.transforms.v2 as transforms
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2._utils import _get_fill


class RandomSegmentationCrop:
    """
    Class meant to crop images based on conditions of labels.
    """

    def __init__(
        self,
        size: List[int],
        cat_max_ratio: float = 0.75,
        max_attempts: int = 100,
        pad_if_needed: bool = False,
        padding: List = None,
        fill_val: str = '127',
        padding_mode: str = 'constant',
    ):
        """
        Random Segmentation crop that ensures that, if no valid segmentation is cropped,
        then this will repeat until done.

        Args:
            size (List[int, int]): [H, W] crop size.
            cat_max_ratio (float, optional): Max ratio of one class overloading in the cropped segmented map. Defaults to 0.75.
            max_attempts (int, optional): Maximum attempts to find crop satisfying the cat_max_ratio. Defaults to 100.
            pad_if_needed (bool, optional): Pad if cropsize is bigger than image.. Defaults to False.
            padding (List, optional): Padding on each img border. " "len(padding) == 0, 1, 2 and 4 are valid.. Defaults to [].
            fill_val (int, optional): Fill value when padding is used and padding_mode is 'constant'. Valid values are '0', '127' and 'mean'. Defaults to 127.
            padding_mode (str, optional): Padding mode if padding is enabled. Valid values are 'constant', 'edge', 'reflect', 'symmetric. Defaults to "constant".
        """

        self.size = size
        self.cat_max_ratio = cat_max_ratio
        self.max_attempts = max_attempts
        self.pad_if_needed = pad_if_needed
        self.padding = padding
        self.fill_val = fill_val
        self.padding_mode = padding_mode

    def transform(self):
        """Transform function that calls the required transformation.

        Raises:
            ValueError: If invalid ratio given.
            ValueError: If invalid fill value given.

        Returns:
            _type_: Transformed image.
        """
        # Get the ignore index from the data if present
        # Here, the index value is hardcoded for semantic segmentation (CityScape)
        self.ignore_index = 255
        # If the ignore index is still invalid, assign accordingly
        if self.ignore_index is None or self.ignore_index < 0:
            self.ignore_index = -100

        # Raise ValueError if the cat_max_ratio is not in the valid range
        if self.cat_max_ratio < 0.0 and self.cat_max_ratio > 1.0:
            raise ValueError(
                f'Cat_max ratio should be between 0 and 1. Given value is' f' {self.cat_max_ratio}'
            )

        padding = self.padding if self.padding else None
        if self.fill_val == '0':
            fill_img = [
                0,
            ]
        elif self.fill_val == '127':
            fill_img = [
                127,
            ]
        elif self.fill_val == 'mean':
            fill_img = list(map(int, mean_rgb=(123.675, 116.28, 103.53)))
        else:
            print(f'self.fill_val: {self.fill_val}')
            raise ValueError

        fill_val = {
            tv_tensors.Image: fill_img,
            tv_tensors.Mask: 255,
            'others': None,
        }

        return _RandomSegmentationCrop(
            size=self.size,
            padding=padding,
            fill=fill_val,
            pad_if_needed=self.pad_if_needed,
            padding_mode=self.padding_mode,
            ignore_index=self.ignore_index,
            cat_max_ratio=self.cat_max_ratio,
            max_attempts=self.max_attempts,
        )


class _RandomSegmentationCrop(transforms.RandomCrop):
    """Random segmentation crop class.

    Args:
        transforms (_type_): Random crop transform inherited and overwritten.
    """

    def __init__(
        self,
        size: int | typing.Sequence[int],
        padding: int | typing.Sequence[int] | None = None,
        pad_if_needed: bool = False,
        fill: int
        | float
        | typing.Sequence[int]
        | typing.Sequence[float]
        | None
        | Dict[
            typing.Type | str,
            int | float | typing.Sequence[int] | typing.Sequence[float] | None,
        ] = 0,
        padding_mode: Literal['constant']
        | Literal['edge']
        | Literal['reflect']
        | Literal['symmetric'] = 'constant',
        ignore_index: int = 255,
        cat_max_ratio: float = 0.75,
        max_attempts: int = 10,
    ) -> None:
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
        self.ignore_index = ignore_index
        self.cat_max_ratio = cat_max_ratio
        self.max_attempts = max_attempts

    # Overwrite transform to expect interpolation to be in params.
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Overriding the transform method.

        Args:
            inpt (Any): Input on which the transform is applied.
            params (Dict[str, Any]): The parameters of the transform.
        """
        if params['needs_pad']:
            fill = _get_fill(self._fill, type(inpt))
            inpt = self._call_kernel(
                F.pad,
                inpt,
                padding=params['padding'],
                fill=fill,
                padding_mode=self.padding_mode,
            )

        if params['needs_crop']:
            inpt = self._call_kernel(
                F.crop,
                inpt,
                top=params['top'],
                left=params['left'],
                height=params['height'],
                width=params['width'],
            )

        return inpt

    def forward(self, *inputs: Any) -> Any:
        """
        Overriding the forward method for the transformation.
        """
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        self._check_inputs(flat_inputs)
        # Input is flattened
        # Needs transform list assigns True for Image and Mask
        needs_transform_list = self._needs_transform_list(flat_inputs)
        for _ in range(self.max_attempts):
            # Get the parameters indicating if the crop is needed
            # if the pad is needed and what padding value
            params = self._get_params(
                [
                    inpt
                    for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
                    if needs_transform
                ]
            )
            # Flat outputs indicate the cropped part of the image and Mask
            flat_outputs = [
                self._transform(inpt, params) if needs_transform else inpt
                for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
            ]

            for inpt, needs_transform in zip(flat_outputs, needs_transform_list):
                if needs_transform and type(inpt) == tv_tensors.Mask and self.cat_max_ratio != 1.0:
                    # Convert the tensor to a numpy array
                    np_seg_mp = np.array(inpt)
                    # Get the different labels and its each count
                    labels, count = np.unique(np_seg_mp, return_counts=True)
                    # Prune out the ignore index
                    count = count[labels != self.ignore_index]
                    # Make sure that there is more than one class in the cropped seg map
                    # or Mask and it does not dominate more than ratio specified
                    if len(count) > 1 and (np.max(count) / np.sum(count) < self.cat_max_ratio):
                        return tree_unflatten(flat_outputs, spec)
        return tree_unflatten(flat_outputs, spec)
