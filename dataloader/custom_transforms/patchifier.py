import os
import pickle
from typing import Any, Dict, List

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import tv_tensors
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms.v2 import Transform

from patchify.patchify import Patchify


class Patchifier:
    """
    Class representing the transformation to patchify images. Each patch can
    then be worked upon based on requirements. A look-up table is necessary as
    this is going to be used as reference to determine the important patches.
    """

    def __init__(
        self,
        lut: str = 'None',
        patch: int = 256,
        imp_patches_compress_quality: int = 100,
        other_compress_quality: int = 10,
    ) -> None:
        self.lut = lut
        self.patch_size = patch
        self.imp_patches_compress_quality = imp_patches_compress_quality
        self.other_compress_quality = other_compress_quality

    def transform(self) -> Transform:
        if self.lut == 'None':
            self.lut = None

        return PatchifyAndDePatchify(
            lut_path=self.lut,
            patch_size=self.patch_size,
            other_compress_quality=self.other_compress_quality,
            imp_patches_compress_quality=self.imp_patches_compress_quality,
        )


class PatchifyAndDePatchify(Transform):
    """
    Transform meant to Patchify and Depatchify the image.
    The most important patches are encoded to the high quality specified,
    while the other patches are encoded to the low quality.
    The patches that are kept are based on a LUT.
    """

    def __init__(
        self,
        lut_path: str,
        patch_size: int,
        other_compress_quality: int,
        imp_patches_compress_quality: int,
    ) -> None:
        super().__init__()
        # Check before construction if the patch size is valid
        if (patch_size % 32) != 0:
            raise ValueError(
                f'Patch size should be valid and divisible by 32. Received:' f'{patch_size}'
            )
        # Check if the path given is valid
        if lut_path is not None:
            if not os.path.exists(lut_path):
                raise ValueError('Invalid path for look-up-table')
            # Check if the file extension is .pkl file
            if not lut_path.endswith('.pkl'):
                raise ValueError('Received file should be a valid .pkl file')
        # Check for compress and decompress quality
        if other_compress_quality > 100 or other_compress_quality < 1:
            raise ValueError(
                f'Others Compress quality should be between 1-100'
                f'Received: {other_compress_quality}'
            )
        # Check for compress and decompress quality
        if imp_patches_compress_quality > 100 or imp_patches_compress_quality < 1:
            raise ValueError(
                f'Imp Patches Compress quality should be between 1-100'
                f'Received: {imp_patches_compress_quality}'
            )
        self.patch_size = patch_size
        self.lut_path = lut_path
        self.other_compress_quality = other_compress_quality
        self.imp_patches_compress_quality = imp_patches_compress_quality
        self.lut = {}
        self.set_lut()
        self.patchifier = Patchify(patchsize=self.patch_size)

    def set_lut(self) -> None:
        """Function to set LUT based on the path."""
        # Load the look up table to know the most important patches.
        if self.lut_path is not None:
            with open(self.lut_path, 'rb') as file:
                self.lut = pickle.load(file)

    def _transform(self, inputs: Any, params: Dict[str, Any]) -> Any:
        """
        Override the transform method of the torchvision transform.
        """

        img_tensor = inputs.unsqueeze(0)

        # Get the name of the file to check with the LUT
        # @NOTE: This needs to be changed to: data_info["sample_idx"] = idx
        # file_name = os.path.splitext(os.path.basename(inputs[0]['img_path']))[0]
        file_name = params['img_name']

        # Create patches. Need to convert to float32 to avoid an issue
        # The output of this would be (N_p, C, H_p, W_p)
        patched_parts = self.patchifier(img_tensor.to(torch.float32)).squeeze(0)
        # Convert to uint8
        patched_parts = patched_parts.to(torch.uint8)

        # Check if the file_name, indicating the key is present in the LUT
        # If the data is present in the LUT, do patchify and what not
        if self.lut is not None and file_name in self.lut:
            high_scoring_indexes = self.lut[file_name]
            """
            # Simply replace the indices from the compressed_parts with orig
            patched_jpeg_parts[high_scoring_indexes]
                = patched_parts[high_scoring_indexes]
            """

            # Create an empty list to have the decoded part of jpeg
            image_patched = []
            # Iterate over each patch and determine the encode, decode quality
            for index in range(patched_parts.shape[0]):
                if index in high_scoring_indexes:
                    image_patched.append(
                        decode_jpeg(
                            encode_jpeg(
                                patched_parts[index],
                                quality=self.imp_patches_compress_quality,
                            )
                        )
                    )
                else:
                    image_patched.append(
                        decode_jpeg(
                            encode_jpeg(
                                patched_parts[index],
                                quality=self.other_compress_quality,
                            )
                        )
                    )

            patched_parts = torch.stack(image_patched, dim=0)

        # Construct the patched parts to a full image
        reshaped_color_patches = patched_parts.view(4, 8, 3, 256, 256)
        reshaped_color_patches = reshaped_color_patches.permute(0, 3, 1, 4, 2)
        original_img = reshaped_color_patches.contiguous().view(1024, 2048, 3).permute(2, 0, 1)

        tv_tensor_img = tv_tensors.Image(original_img)

        return tv_tensor_img

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
        """Get the parameters from the inputs.

        Args:
            flat_inputs (List[Any]): Flat inputs.

        Returns:
            Dict[str, Any]: A dictionary based on the inputs containing params.
        """
        # super()._get_params(flat_inputs)
        # Right now simply add a dictionary containing the image path as key
        # And value as file_name which is used as reference
        # Also return the sample index which will be later used for LUT
        # The above one is simply hardcoded
        file_name = os.path.splitext(os.path.basename(flat_inputs[2]))[0]
        return {'img_name': file_name}  # dict(img_name=file_name)  # , sample_idx=flat_inputs[3])

    def forward(self, *inputs: Any) -> Any:
        """Forward method.

        Returns:
            Any: Inputs that needs to be transformed.
        """
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])

        # This will simply be a pass because we are not overriding
        self._check_inputs(flat_inputs)

        # Not sure if we need to have parameters here
        # Ignore _get_params
        params = self._get_params(flat_inputs)

        # Get the needs transform list
        needs_transform_list = self._needs_transform_list(flat_inputs, params)

        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)
