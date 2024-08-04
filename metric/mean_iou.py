from typing import Dict

import torch


class MeanIoU:
    def __init__(self, num_classes: int, ignore_index: int = -100, from_logits: bool = False):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.from_logits = from_logits

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.from_logits:
            output = torch.argmax(output, dim=1)

        # Verify if the image and the target are of same size
        output = (output + 1) * (target != self.ignore_index)
        target = (target + 1) * (target != self.ignore_index)

        # Create intersections of the prediction and true labels
        intersections = output * (output == target)

        # Create individual histograms that will be used for mIoU
        output = torch.histc(
            output,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        target = torch.histc(
            target,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        intersections = torch.histc(
            intersections,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )

        # Calculate the union between the predictions and target
        unions = output + target - intersections

        # Once we have the overall union, we will also calculate individual IoUs
        # Adding a small number here prevents division by 0
        individual_iou = intersections / (unions + 1e-6)

        # Once we create individual ious, log each
        ind_iou = {}
        for i in range(self.num_classes):
            ind_iou[f'IoU_class_{i}'] = individual_iou[i]

        # logs[self.name] = torch.stack([intersections, unions])

        # Calculate the mean IoU
        # mean_iou = torch.mean(intersections/(unions + 1e-6))

        return torch.stack([intersections, unions])
