"""
This utility file contains classes and functions related to saving model checkpoints.
"""
import os

import torch


class SaveBestModel:
    """
    Class to save the best model during training.
    """

    def __init__(self, best_valid_loss=float('inf')) -> None:
        self.least_valid_loss = best_valid_loss

    def __call__(
        self,
        current_valid_loss: float,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        path_to_save: str,
    ):
        """Called when the object of the class is called.

        Args:
            current_valid_loss (float): Current validation loss.
            epoch (int): The epoch attributing to the training loop.
            model (torch.nn.Module): The model who's state needs to be saved.
            optimizer (torch.optim.Optimizer): The optimizer who's state dict to save.
            path_to_save (str): The path where the checkpoint is saved.
        """
        # Exit if the current valid loss is not greater than the best valid loss
        if not (self.least_valid_loss > current_valid_loss):
            return
        # If the current valid loss is the least, then save the state
        self.least_valid_loss = current_valid_loss

        # Create a directory recursively if the path does not exist
        os.makedirs(path_to_save, exist_ok=True)

        print(f'Saving the best model of epoch: {epoch+1} with loss: {current_valid_loss:.4f}')

        torch.save(
            {
                'epoch': (epoch + 1),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'least_valid_loss': self.least_valid_loss,
            },
            f'{path_to_save}/best_model.pth',
        )

        print(f"The best model saved at: {os.path.join(path_to_save, 'best_model.pth')}")


# Function used to save the last/latest model
def save_model(
    last_valid_loss: float,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path_to_save: str,
):
    """Function to save model checkpoints

    Args:
        last_valid_loss (float): Current validation loss.
        epoch (int): The epoch attributing to the training loop.
        model (torch.nn.Module): The model who's state needs to be saved.
        optimizer (torch.optim.Optimizer): The optimizer who's state dict to save.
        path_to_save (str): The path where the checkpoint is saved.
    """
    print(f'Saving the model checkpoint. Epoch: {epoch+1} and val_loss: {last_valid_loss}')
    torch.save(
        {
            'epoch': (epoch + 1),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'least_valid_loss': last_valid_loss,
        },
        f'{path_to_save}/last_model.pth',
    )
