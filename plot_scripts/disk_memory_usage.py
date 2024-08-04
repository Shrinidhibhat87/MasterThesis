import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


class UseCasePlotter:
    def __init__(self, patchify_dataframe: pd.DataFrame, jpeg_dataframe: pd.DataFrame) -> None:
        self.patchify_dataframe = patchify_dataframe
        self.jpeg_dataframe = jpeg_dataframe

    def plot_and_save(self, title: str, ideal_x: float, ideal_y: float, filename=None) -> None:
        plt.figure(figsize=(10, 6))
        # Plot the patchify_dataframe
        plt.scatter(
            self.patchify_dataframe['Disk Storage'],
            self.patchify_dataframe['mIoU'],
            s=50,
            marker='o',
            label='Patchify-Compress-Decompress',
        )
        # Plot the jpeg-compress dataframe
        plt.scatter(
            self.jpeg_dataframe['Disk Storage'],
            self.jpeg_dataframe['mIoU'],
            s=50,
            marker='x',
            label='JPEG-Compress-Decompress',
        )

        # Plot the info as annotations
        for i, (info_patch, info_jpeg) in enumerate(
            zip(self.patchify_dataframe['info'], self.jpeg_dataframe['info'])
        ):
            plt.annotate(
                info_patch,
                (
                    self.patchify_dataframe['Disk Storage'][i],
                    self.patchify_dataframe['mIoU'][i],
                ),
                xytext=(5, 0),
                textcoords='offset points',
                ha='left',
                va='top',
                fontsize=8,
            )
            plt.annotate(
                info_jpeg,
                (
                    self.jpeg_dataframe['Disk Storage'][i],
                    self.jpeg_dataframe['mIoU'][i],
                ),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=8,
            )

        plt.scatter(ideal_x, ideal_y, marker='*', s=200, color='gold', label='Ideal Point')

        plt.xlabel('Disk Storage (MB)')
        plt.ylabel('mIoU (%)')
        plt.title(title)
        plt.legend()
        plt.grid()

        # Generate a unique file name to avoid overwriting
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not filename:
            file_name = f'Usecase_70unseen_Patchify_pipeline_{current_time}.png'
        else:
            file_name = f'{filename}_{current_time}.png'

        # Save the plot to the file
        plt.savefig(file_name, dpi=300)
        print(f'Plot saved to: {os.path.join(os.getcwd(), file_name)}')


def plot_usecase_70():
    # First for the usecase of 70unseen keep percent of 10
    df_patchify_unseen_70 = pd.DataFrame(
        {
            # 'Disk Storage': [1212.05, 1180.24, 1322.93, 1146.41, 1121.84, 1221.04, 928.52, 916.94], #1180.24 is something that is being calculated
            # 'mIoU': [78.88, 77.92, 77.21, 78.12, 78.05, 78.44, 77.94, 77.86], #77.92
            # 'info':['T90:O10', 'T80:O10', 'T35:O15','T70:O10', 'T60:O10', 'T15:O15', 'T30:O10', 'T15:O10']
            'Disk Storage': [1212.05, 1180.24, 1322.93, 1146.41, 1121.84],
            'mIoU': [78.88, 77.92, 77.21, 78.12, 78.05],
            'info': ['T90:O10', 'T80:O10', 'T35:O15', 'T70:O10', 'T60:O10'],
        }
    )
    # Define parameters for other jpeg
    df_jpeg_unseen_70 = pd.DataFrame(
        {
            'Disk Storage': [2343.18, 2125.74, 1850.69, 1437.60, 1227.02],
            'mIoU': [78.58, 78.64, 78.52, 77.99, 77.71],
            'info': ['50Q', '40Q', '30Q', '20Q', '15Q'],
        }
    )

    usecase_70_plotter = UseCasePlotter(
        patchify_dataframe=df_patchify_unseen_70,
        jpeg_dataframe=df_jpeg_unseen_70,
    )

    # Plot and see usecase 70 scenario
    usecase_70_plotter.plot_and_save(
        title='Usecase 70 unseen (Disk Storage vs mIoU)',
        ideal_x=1100,
        ideal_y=79.9,
        filename='Usecase_70unseen_Patchify_pipeline_disk_memusage',
    )


def plot_usecase_50():
    # Create plots for usecase 50unseen keep percent 10
    df_patchify_unseen_50 = pd.DataFrame(
        {
            'Disk Storage': [688.47, 817.19, 784.9, 743.36, 402.58, 215.54, 157.87],
            'mIoU': [78.98, 79.02, 78.30, 78.68, 78.54, 78.06, 76.56],
            'info': [
                'T80:O07',
                'T70:O10',
                'T50:O10',
                'T30:O10',
                'T15:O05',
                'T10:O02',
                'T05:O01',
            ],
        }
    )
    # Define parameters for other jpeg
    df_jpeg_unseen_50 = pd.DataFrame(
        {
            'Disk Storage': [1514.69, 1318.57, 1024.28, 873.99, 652.32, 311.20, 119.20],
            'mIoU': [79.29, 79.05, 78.62, 78.82, 77.85, 77.95, 77.75],
            'info': ['40Q', '30Q', '20Q', '15Q', '10Q', '5Q', '1Q'],
        }
    )

    usecase_50_plotter = UseCasePlotter(
        patchify_dataframe=df_patchify_unseen_50, jpeg_dataframe=df_jpeg_unseen_50
    )

    # Plot and see usecase 70 scenario
    usecase_50_plotter.plot_and_save(
        title='Usecase 50 unseen (Disk Storage vs mIoU)',
        ideal_x=100,
        ideal_y=79.9,
        filename='Usecase_50unseen_Patchify_pipeline_disk_memusage',
    )


def plot_usecase_30():
    # Create plots for usecase 30unseen keep percent 10
    df_patchify_unseen_30 = pd.DataFrame(
        {
            'Disk Storage': [240.95, 132.21, 93.94],
            'mIoU': [79.21, 78.75, 78.62],
            'info': ['T15:O05', 'T07:O03', 'T05:O01'],
        }
    )
    # Define parameters for other jpeg
    df_jpeg_unseen_30 = pd.DataFrame(
        {
            'Disk Storage': [
                70.94,
                99.20,
                120.45,
            ],  # 120.45 is bs. The actual result is being calc
            'mIoU': [79.17, 78.85, 78.05],
            'info': ['5Q', '3Q', '1Q'],
        }
    )

    usecase_30_plotter = UseCasePlotter(
        patchify_dataframe=df_patchify_unseen_30, jpeg_dataframe=df_jpeg_unseen_30
    )

    # Plot and see usecase 70 scenario
    usecase_30_plotter.plot_and_save(
        title='Usecase 30 unseen (Disk Storage vs mIoU)',
        filename='Usecase_30unseen_Patchify_pipeline',
    )


def plot_em_all():
    plot_usecase_70()
    plot_usecase_50()
    # plot_usecase_30()


def main():
    plot_em_all()


if __name__ == '__main__':
    main()
