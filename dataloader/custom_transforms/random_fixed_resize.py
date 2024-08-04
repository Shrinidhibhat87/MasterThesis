from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2._utils import query_size


class RandomFixedResize(transforms.Transform):
    """
    Transform meant to resize images based on fixed scale.

    Args:
        transforms (_type_): Inherit torch transform.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = [0.2, 1.0],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
    ):
        """Constructor method.

        Args:
            scale_range (Tuple[float, float], optional): Range of fixed scale.. Defaults to [0.2, 1.0].
            interpolation (Union[InterpolationMode, int], optional): inerpolation method. Defaults to InterpolationMode.BILINEAR.
            antialias (Optional[bool], optional): Antialias bool. Defaults to True.
        """
        super().__init__()
        self.scale_range = scale_range
        self.interpolation = interpolation
        self.antialias = antialias

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        """Overriden get params function to return parameters.

        Args:
            flat_inputs (List[Any]): Flattened inputs.

        Returns:
            Dict[str, Any]: Mapping of the parameters required.
        """
        orig_height, orig_width = query_size(flat_inputs)
        # Randomly choose a number from the list
        r = np.random.choice(self.scale_range)
        # The height and width will be a valid integer eventhough r is float because
        # of type conversion.
        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        return {'size': (new_height, new_width)}

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """
        Calling the resize kernel.

        Args:
            inpt (Any): Input to transform.
            params (Dict[str, Any]): Parameters that help in the size.
        """
        return self._call_kernel(
            F.resize,
            inpt,
            size=params['size'],
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
