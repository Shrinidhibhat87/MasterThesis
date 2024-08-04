"""
This utility file contains a class that helps in tracking experiments
"""
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardTracker:
    def __init__(
        self,
        log_path: str,
        track_learning_rate: bool = True,
        track_weight_dist: bool = True,
        track_gradient_dist: bool = True,
    ) -> None:
        """Class meant to track experiments on tensorboard.

        Args:
            log_path (str): Path to the log files.
            track_learning_rate (bool, optional): If lr must be tracked. Defaults to True.
            track_weight_dist (bool, optional): If wt dist must be tracked. Defaults to True.
            track_gradient_dist (bool, optional): If grad distmust be tracked. Defaults to True.
        """
        self.tracking_lr = track_learning_rate
        self.tracking_wt_dist = track_weight_dist
        self.track_grad_dist = track_gradient_dist
        self.log_path = log_path

        # Instantiate the SummaryWriter object
        self.writer = SummaryWriter(log_dir=log_path)

        # Show message that there is a tracker setup and this location
        print(
            f'Tensorboard experiment is being tracked at: http://localhost:6006/'
            f'or tensorboard --logdir={self.log_path}'
        )

    def track(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_miou: float,
        model: torch.nn.Module = None,
        learning_rate: float = 0.0,
    ) -> None:
        """Function that adds the scalar and histogram to the tensorboard writer

        Args:
            epoch (int): Current epoch.
            train_loss (float): Training loss.
            val_loss (float): Validation loss.
            val_miou (float): Validation mIoU.
            model (torch.nn.Module, optional): Torch model. Defaults to None.
            learning_rate (float, optional): Learning rate to track. Defaults to 0.0.
        """
        # Add training loss, val_loss and val_miou
        self.writer.add_scalar('train/loss', train_loss, epoch + 1)
        self.writer.add_scalar('val/loss', val_loss, epoch + 1)
        self.writer.add_scalar('val/miou', val_miou, epoch + 1)

        if self.tracking_lr:
            self.writer.add_scalar('train/learning_rate', learning_rate, epoch + 1)

        if self.tracking_wt_dist:
            for name, param in model.named_parameters():
                self.writer.add_histogram(f'weights/{name}', param, epoch + 1)

        if self.track_grad_dist:
            for name, param in model.named_parameters():
                self.writer.add_histogram(f'gradients/{name}', param.grad, epoch + 1)
