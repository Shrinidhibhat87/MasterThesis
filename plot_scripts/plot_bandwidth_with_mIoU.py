import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


class UseCasePlotter:
    def __init__(self, patchify_dataframe: pd.DataFrame, jpeg_dataframe: pd.DataFrame) -> None:
        self.patchify_dataframe = patchify_dataframe
        self.jpeg_dataframe = jpeg_dataframe

    def plot_and_save(self, title: str, filename=None) -> None:
        plt.figure(figsize=(10, 6))
        # Plot the patchify_dataframe
        plt.plot(
            self.patchify_dataframe['Bandwidth'],
            self.patchify_dataframe['mIoU'],
            marker='o',
            label='Patchify-Compress-Decompress',
        )
        # Plot the jpeg-compress dataframe
        plt.plot(
            self.jpeg_dataframe['Bandwidth'],
            self.jpeg_dataframe['mIoU'],
            marker='x',
            label='JPEG-Compress-Decompress',
        )
        # Plot the info as annotations
        for i, (info_patch, info_jpeg) in enumerate(
            zip(self.patchify_dataframe['info'], self.jpeg_dataframe['info'])
        ):
            plt.annotate(
                info_patch,
                (self.patchify_dataframe['Bandwidth'][i], self.patchify_dataframe['mIoU'][i]),
                xytext=(5, 8),
                textcoords='offset points',
                ha='left',
                va='top',
                fontsize=8,
            )
            plt.annotate(
                info_jpeg,
                (self.jpeg_dataframe['Bandwidth'][i], self.jpeg_dataframe['mIoU'][i]),
                xytext=(0, 5),
                textcoords='offset points',
                fontsize=8,
            )

        plt.xlabel('Bandwidth (MB)')
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
            'Bandwidth': [202.10, 177.53, 167.60, 167.93, 161.49, 159.10],
            'mIoU': [78.88, 78.37, 78.12, 77.21, 77.20, 78.44],  # 77.92
            'info': ['T90:O10', 'T80:O10', 'T70:O10', 'T35:O15', 'T60:O10', 'T15:O15'],
        }
    )
    # Define parameters for other jpeg
    df_jpeg_unseen_70 = pd.DataFrame(
        {
            'Bandwidth': [225.38, 197.08, 169.12, 137.26, 120.40],
            'mIoU': [78.58, 78.54, 78.52, 77.99, 77.71],
            'info': ['50Q', '40Q', '30Q', '20Q', '15Q'],
        }
    )

    usecase_70_plotter = UseCasePlotter(
        patchify_dataframe=df_patchify_unseen_70, jpeg_dataframe=df_jpeg_unseen_70
    )

    # Plot and see usecase 70 scenario
    usecase_70_plotter.plot_and_save(
        title='Usecase 70 unseen (Bandwidth vs mIoU)', filename='Usecase_70unseen_Patchify_pipeline'
    )


def plot_usecase_50():
    # Create plots for usecase 50unseen keep percent 10
    df_patchify_unseen_50 = pd.DataFrame(
        {
            'Bandwidth': [119.60, 119.59, 107.27, 90.48, 82.84],
            'mIoU': [78.98, 78.45, 78.68, 78.54, 78.06],
            'info': ['T80:O07', 'T70:O10', 'T30:O10', 'T15:O05', 'T10:O02'],
        }
    )
    # Define parameters for other jpeg
    df_jpeg_unseen_50 = pd.DataFrame(
        {
            'Bandwidth': [140.47, 120.55, 97.88, 85.86, 72.92],
            'mIoU': [79.29, 79.34, 78.62, 78.47, 78.65],
            'info': ['40Q', '30Q', '20Q', '15Q', '15Q'],
        }
    )

    usecase_50_plotter = UseCasePlotter(
        patchify_dataframe=df_patchify_unseen_50, jpeg_dataframe=df_jpeg_unseen_50
    )

    # Plot and see usecase 70 scenario
    usecase_50_plotter.plot_and_save(
        title='Usecase 50 unseen (Bandwidth vs mIoU)', filename='Usecase_50unseen_Patchify_pipeline'
    )


def plot_usecase_30():
    # Create plots for usecase 30unseen keep percent 10
    df_patchify_unseen_30 = pd.DataFrame(
        {
            'Bandwidth': [119.60, 119.59, 107.27, 90.48, 82.84],
            'mIoU': [78.98, 78.45, 78.68, 78.54, 78.06],
            'info': ['T80:O07', 'T70:O10', 'T30:O10', 'T15:O05', 'T10:O02'],
        }
    )
    # Define parameters for other jpeg
    df_jpeg_unseen_30 = pd.DataFrame(
        {
            'Bandwidth': [35.24, 31.95, 31.14],
            'mIoU': [79.17, 78.85, 78.05],
            'info': ['5Q', '3Q', '1Q'],
        }
    )

    usecase_30_plotter = UseCasePlotter(
        patchify_dataframe=df_patchify_unseen_30, jpeg_dataframe=df_jpeg_unseen_30
    )

    # Plot and see usecase 70 scenario
    usecase_30_plotter.plot_and_save(
        title='Usecase 50 unseen (Bandwidth vs mIoU)', filename='Usecase_50unseen_Patchify_pipeline'
    )


def plot_em_all():
    plot_usecase_70()
    plot_usecase_50()

    # Create plots for usecase 50unseen keep percent 10
    df_patchify_unseen_50 = pd.DataFrame(
        {
            'Bandwidth': [119.60, 119.59, 107.27, 90.48, 82.84],
            'mIoU': [78.98, 78.45, 78.68, 78.54, 78.06],
            'info': ['T80:O07', 'T70:O10', 'T30:O10', 'T15:O05', 'T10:O02'],
        }
    )
    # Define parameters for other jpeg
    df_jpeg_unseen_50 = pd.DataFrame(
        {
            'Bandwidth': [140.47, 120.55, 97.88, 85.86, 72.92],
            'mIoU': [79.29, 79.34, 78.62, 78.47, 78.65],
            'info': ['40Q', '30Q', '20Q', '15Q', '15Q'],
        }
    )

    usecase_70_plotter = UseCasePlotter(
        patchify_dataframe=df_patchify_unseen_50, jpeg_dataframe=df_jpeg_unseen_50
    )

    # Plot and see usecase 70 scenario
    usecase_70_plotter.plot_and_save(
        title='Usecase 50 unseen (Bandwidth vs mIoU)', filename='Usecase_50unseen_Patchify_pipeline'
    )


def main():
    plot_em_all()


if __name__ == '__main__':
    main()
