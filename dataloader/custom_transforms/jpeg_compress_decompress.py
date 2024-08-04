import io
import os
import pickle
from typing import Any, Dict, List, Sequence, Union

import PIL
import torch
import torchvision.transforms.v2 as transforms
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import tv_tensors
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms.v2 import functional as F
from torchvision.utils import _log_api_usage_once


class JPEGCompressDecompress:
    """
    Class to Parametrize JPEG compression - decompression transformation.
    The idea is to simulate cases where training images are stored in a certain
    compression ratio (even though they aren't really) and decompressed for training.
    """

    def __init__(self, quality: int, lut: str) -> None:
        self.quality = quality
        self.lut = lut

    def jpeg_transform(self) -> transforms.Transform:
        if self.lut == 'None':
            self.lut = None
        return JPEG(quality=self.quality, lut_path=self.lut)


def _check_sequence_input(x, name, req_sizes):
    # The below functions was simply stolen from 1.19.0 as this is used
    # the latest torchvision version
    # https://pytorch.org/vision/master/_modules/torchvision/transforms/v2/_augment.html#JPEG

    msg = req_sizes[0] if len(req_sizes) < 2 else ' or '.join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError(f'{name} should be a sequence of length {msg}.')
    if len(x) not in req_sizes:
        raise ValueError(f'{name} should be a sequence of length {msg}.')


def jpeg(image: torch.Tensor, quality: int) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.JPEG` for details."""
    if torch.jit.is_scripting():
        return jpeg_image(image, quality=quality)

    _log_api_usage_once(jpeg)

    kernel = F._utils._get_kernel(jpeg, type(image))
    return kernel(image, quality=quality)


@F._utils._register_kernel_internal(jpeg, torch.Tensor)
@F._utils._register_kernel_internal(jpeg, tv_tensors.Image)
def jpeg_image(image: torch.Tensor, quality: int) -> torch.Tensor:
    original_shape = image.shape
    image = image.view((-1,) + image.shape[-3:])

    if image.shape[0] == 0:  # degenerate
        return image.reshape(original_shape).clone()

    # Note that the below api does the compress and decompress in one go
    # In order to get only the encoded part, try
    # encoded_img = encode_jpeg(image[i], quality=quality)
    image = [decode_jpeg(encode_jpeg(image[i], quality=quality)) for i in range(image.shape[0])]
    image = torch.stack(image, dim=0).view(original_shape)
    return image


@F._utils._register_kernel_internal(jpeg, PIL.Image.Image)
def _jpeg_image_pil(image: PIL.Image.Image, quality: int) -> PIL.Image.Image:
    raw_jpeg = io.BytesIO()
    image.save(raw_jpeg, format='JPEG', quality=quality)

    # we need to copy since PIL.Image.open() will return PIL.JpegImagePlugin.
    # JpegImageFile which is a sub-class of PIL.Image.Image.
    # this will fail check_transform() test.
    return PIL.Image.open(raw_jpeg).copy()


class JPEG(transforms.Transform):
    """Apply JPEG compression and decompression to the given images.

    If the input is a :class:`torch.Tensor`, it is expected
    to be of dtype uint8, on CPU, and have [..., 3 or 1, H, W] shape,
    where ... means an arbitrary number of leading dimensions.

    Args:
        quality (sequence or number): JPEG quality, from 1 to 100.
            Lower means more compression. If quality is a sequence like (min, max),
            it specifies the range of JPEG quality to randomly select from
            (inclusive of both ends).

    Returns:
        image with JPEG compression.
    """

    def __init__(self, quality: Union[int, Sequence[int]], lut_path: str):
        super().__init__()
        if isinstance(quality, int):
            quality = [quality, quality]
        else:
            _check_sequence_input(quality, 'quality', req_sizes=(2,))

        if not (
            1 <= quality[0] <= quality[1] <= 100
            and isinstance(quality[0], int)
            and isinstance(quality[1], int)
        ):
            raise ValueError(f'quality must be an integer from 1 to 100,' f' got {quality =}')
        # Check if the path given is valid
        if lut_path is not None:
            if not os.path.exists(lut_path):
                raise ValueError('Invalid path for look-up-table')
            # Check if the file extension is .pkl file
            if not lut_path.endswith('.pkl'):
                raise ValueError('Received file should be a valid .pkl file')

        self.quality = quality
        self.lut_path = lut_path
        self.lut = {}
        self.set_lut()

    def set_lut(self) -> None:
        """Function to set LUT based on the path."""
        # Load the look up table to know the most important patches.
        if self.lut_path is not None:
            with open(self.lut_path, 'rb') as file:
                self.lut = pickle.load(file)

    def _needs_transform_list(self, flat_inputs: List[Any], params: Dict[str, Any]) -> List[bool]:
        """Find out the elements in the input list that needs transform.

        Args:
            flat_inputs (List[Any]): Flat inputs.

        Returns:
            List[bool]: A list of boolean values for each input.
        """
        needs_transform_list = []
        # The transformation here right now is applicable only to the img
        # If there is a need to do this to the seg_map, simply use the:
        # the base needs_transform_list OR
        # super()._needs_transform_list(flat_inputs)
        # Get file name from params
        file_name = params['img_name']
        # Check if this is present in the lut
        # If the data is presnt, that means this is an unseen data that has
        # attn_scores, so patchify, encode decode etc.
        present = False
        # If the data is not present, dont perform any transformation on it
        if self.lut is not None and file_name in self.lut:
            present = True
        # Iterate over each item
        for item in flat_inputs:
            if isinstance(item, (tv_tensors.Image)) and present:
                needs_transform_list.append(True)
            else:
                needs_transform_list.append(False)

        return needs_transform_list

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        quality = torch.randint(self.quality[0], self.quality[1] + 1, ()).item()
        # Get the name of the file from the inputs, can be extended to idx too
        file_name = os.path.splitext(os.path.basename(flat_inputs[2]))[0]
        return {
            'quality': quality,
            'img_name': file_name,
        }  # dict(quality=quality, img_name=file_name)

    def forward(self, *inputs: Any) -> Any:
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])

        self._check_inputs(flat_inputs)

        params = self._get_params(flat_inputs)

        needs_transform_list = self._needs_transform_list(flat_inputs, params)

        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(jpeg, inpt, quality=params['quality'])
