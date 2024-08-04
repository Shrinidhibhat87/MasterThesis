import torch
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors


class ToDtype:
    def __init__(self, dtype_image: str = 'float32', dtype_mask: str = 'long'):
        """Simple wrapper class around the original todtype transform

        Args:
            dtype_image (str, optional): Dtype of image. Defaults to 'float32'.
            dtype_mask (str, optional): Dtype of mask. Defaults to 'long'.

        Returns:
            v2.transforms: Original transform class.
        """
        self.dtype_image = self.get_dtype(dtype_image)
        self.dtype_mask = self.get_dtype(dtype_mask)

    def __call__(self, sample):
        """Method called when invoking the object.

        Args:
            sample (Dict): image sample to convert.

        Returns:
            Dict: Converted sample.
        """
        image, mask = sample['image'], sample['mask']
        image = image.to(self.dtype_image)
        mask = mask.to(self.dtype_mask)
        sample['image'] = image
        sample['mask'] = mask
        return sample

    def transform(self):
        dtype = {
            tv_tensors.Image: self.dtype_image,
            tv_tensors.Mask: self.dtype_mask,
            'others': None,
        }

        return transforms.ToDtype(dtype=dtype, scale=False)

    def get_dtype(self, str_dtype: str):
        """Function to return torch dtype based on input

        Args:
            str_dtype (str): Dtype specified.

        Raises:
            NotImplementedError: When asked to convert to an unimplemented dtype

        Returns:
            torch.dtype: torch dtype
        """
        if str_dtype == 'float32':
            dtype = torch.float32
        elif str_dtype == 'float16':
            dtype = torch.float16
        elif str_dtype == 'uint8':
            dtype = torch.uint8
        elif str_dtype == 'long':
            dtype = torch.long
        else:
            raise NotImplementedError('TRANSFORMS | ToDtype | dtype not recognized.')
        return dtype
