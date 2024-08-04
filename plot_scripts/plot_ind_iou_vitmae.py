import os
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class Individual_IoUPlotter:
    def __init__(self, classes: List[str], file_name: str = 'None') -> None:
        self.classes = classes
        self.file_name = file_name

    def plot_and_save(
        self,
        baseline_mask2former_iou: List[float],
        baseline_mask2former_miou: float,
        vitmae_baseline_iou: List[float],
        vitmae_baseline_miou: float,
        vitmae_induced_attn_based_masking: List[float],
        vitmae_induced_attn_based_masking_miou: float,
        vitmae_pure_attn_based_masking: List[float],
        vitmae_pure_attn_based_masking_miou: float,
    ):
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot the histograms
        x = np.arange(len(self.classes))
        width = 0.2

        ax.bar(
            x - 1.5 * width,
            baseline_mask2former_iou,
            width,
            label=f'Baseline Mask2Former (mIoU: {baseline_mask2former_miou}%)',
        )
        ax.bar(
            x - 0.5 * width,
            vitmae_baseline_iou,
            width,
            label=f'Random masking (mIoU: {vitmae_baseline_miou}%)',
        )
        ax.bar(
            x + 0.5 * width,
            vitmae_induced_attn_based_masking,
            width,
            label=f'Random masking induced with attention masking (10%+15%) (mIoU: {vitmae_induced_attn_based_masking_miou}%)',
        )
        ax.bar(
            x + 1.5 * width,
            vitmae_pure_attn_based_masking,
            width,
            label=f'Pure Attention based masking: (mIoU: {vitmae_pure_attn_based_masking_miou}%)',
        )

        # Set x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(self.classes, rotation=90, ha='right')

        # Set axis labels and title
        ax.set_xlabel('Classes')
        ax.set_ylabel('Individual IoU (%)')
        ax.set_title('Histogram of IoU for Different Models')

        # Create a new axes for the legend
        legend_ax = fig.add_axes([0.7, 0.75, 0.2, 0.2])
        legend_ax.axis('off')
        legend_ax.legend(
            ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1], loc='center'
        )

        plt.subplots_adjust(bottom=0.2, top=0.9, right=0.75)

        # Show the plot
        plt.show()

        # Generate a unique file name to avoid overwriting
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        if not self.file_name == 'None':
            file_name = f'Ind_iou_metrics_{current_time}.png'
        else:
            file_name = f'{self.file_name}_{current_time}.png'

        # Save the plot to the file
        plt.savefig(file_name, dpi=300)
        print(f'Plot saved to: {os.path.join(os.getcwd(), file_name)}')


def plot_em_all():
    # Data
    classes = [
        'Road',
        'Sidewalk',
        'Building',
        'Wall',
        'Fence',
        'Pole',
        'Traffic Light',
        'Traffic Sign',
        'Vegetation',
        'Terrain',
        'Sky',
        'Person',
        'Rider',
        'Car',
        'Truck',
        'Bus',
        'Train',
        'Motorcycle',
        'Bicycle',
    ]

    # Input individual IoU
    baseline_mask2former_iou = [
        98.37,
        86.27,
        92.51,
        45.74,
        60.80,
        67.74,
        72.69,
        81.59,
        92.93,
        64.70,
        95.06,
        83.51,
        64.88,
        95.44,
        82.27,
        90.86,
        82.53,
        66.31,
        78.91,
    ]
    vitmae_baseline_iou = [
        97.97,
        83.58,
        91.50,
        57.86,
        55.81,
        60.92,
        64.46,
        76.95,
        91.61,
        60.31,
        93.52,
        80.48,
        59.66,
        94.12,
        80.02,
        80.35,
        82.13,
        61.45,
        72.55,
    ]
    vitmae_induced_attn_based_masking = [
        97.06,
        77.30,
        90.88,
        37.22,
        55.20,
        57.95,
        61.65,
        73.86,
        90.95,
        53.50,
        92.51,
        79.81,
        55.84,
        93.40,
        70.57,
        77.13,
        69.85,
        59.81,
        73.68,
    ]
    vitmae_pure_attn_based_masking = [
        95.61,
        73.90,
        86.82,
        37.17,
        37.07,
        54.76,
        58.52,
        63.12,
        89.56,
        52.68,
        92.67,
        70.43,
        46.69,
        91.20,
        30.21,
        53.95,
        52.27,
        40.30,
        70.15,
    ]

    # mIoU values
    baseline_mask2former_miou = 79.11
    vitmae_baseline_miou = 76.06
    vitmae_induced_attn_based_masking_miou = 72.01
    vitmae_pure_attn_based_masking_miou = 62.64

    # Instantiate the object
    iou_plotter = Individual_IoUPlotter(classes=classes, file_name='ind_iou_metrics')

    # Call the required function
    iou_plotter.plot_and_save(
        baseline_mask2former_iou=baseline_mask2former_iou,
        baseline_mask2former_miou=baseline_mask2former_miou,
        vitmae_baseline_iou=vitmae_baseline_iou,
        vitmae_baseline_miou=vitmae_baseline_miou,
        vitmae_induced_attn_based_masking=vitmae_induced_attn_based_masking,
        vitmae_induced_attn_based_masking_miou=vitmae_induced_attn_based_masking_miou,
        vitmae_pure_attn_based_masking=vitmae_pure_attn_based_masking,
        vitmae_pure_attn_based_masking_miou=vitmae_pure_attn_based_masking_miou,
    )


def main():
    plot_em_all()


if __name__ == '__main__':
    main()
