import os
from datetime import datetime

import matplotlib.pyplot as plt


def plot_em_all():
    # Data
    percentage = [100, 70, 50, 30]
    bandwidth = [6692.25, 4684.71, 3347.93, 2007.51]
    mIoU = [79.11, 78.12, 76.52, 75.5]

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the mIoU
    ax1.plot(percentage, mIoU, marker='o', color='b', label='mIoU')
    ax1.set_xlabel('Percentage of training data')
    ax1.set_ylabel('mIoU (%)', color='b')
    ax1.tick_params('y', colors='b')

    # Plot the bandwidth
    ax2 = ax1.twinx()
    ax2.plot(percentage, bandwidth, marker='s', color='r', label='Bandwidth')
    ax2.set_ylabel('Bandwidth (MB)', color='r')
    ax2.tick_params('y', colors='r')

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Set title and grid
    plt.title('mIoU and Bandwidth vs Percentage of Training Data')
    plt.grid(True)
    plt.show()

    # Generate a unique file name to avoid overwriting
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = 'None'
    if file_name == 'None':
        file_name = f'create_model_usecase_{current_time}.png'
    else:
        file_name = f'{file_name}_{current_time}.png'

    # Save the plot to the file
    plt.savefig(file_name, dpi=300)
    print(f'Plot saved to: {os.path.join(os.getcwd(), file_name)}')


def main():
    plot_em_all()


if __name__ == '__main__':
    main()
