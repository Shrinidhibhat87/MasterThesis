import os
import pickle
from typing import Dict, List, Union

import numpy as np
import torch
from torch import nn


class Patchify(nn.Module):
    # Make the patchsize configurable
    def __init__(self, patchsize=256) -> None:
        super().__init__()
        self.patch_size = patchsize
        self.unfold = torch.nn.Unfold(kernel_size=patchsize, stride=patchsize)

    def forward(self, image_tensor):
        # The shape of tensor x is: (B, C, H, W)
        B, C, _, _ = image_tensor.shape

        # Unfold this using the torch unfold method
        patch_unfold = self.unfold(image_tensor)

        # Reshape the tensor back to (B, Num_patches, C, P, P)
        patch_result = patch_unfold.view(B, C, self.patch_size, self.patch_size, -1).permute(
            0, 4, 1, 2, 3
        )

        return patch_result


def patchifyMAE(tensor: torch.tensor, patch_size: int):
    """
    Patchify a tensor, which is a color image based on ViTMAE
    This does not use torch.unfold as shown previously
    """
    height = tensor.shape[2]
    width = tensor.shape[3]
    # Sanity check, ensure that the H, W of the tensor is divisible by patch_size
    if (height % patch_size != 0) or (width % patch_size != 0):
        raise ValueError(f'H {height}, W {width} is not divisible by {patch_size}')

    batch_size = tensor.shape[0]
    num_patches_height = height // patch_size  # N_H
    num_patches_width = width // patch_size  # N_W

    # Reshape (B, C, H, W) -> (B, C, N_H, P, N_W, P)
    patchified_tensor = tensor.reshape(
        batch_size, tensor.shape[1], num_patches_height, patch_size, num_patches_width, patch_size
    )
    # einsum from BCHPWQ -> BHWPQC
    patchified_tensor = torch.einsum('nchpwq->nhwpqc', patchified_tensor)
    # Reshape to (B, H*W, P*Q*C)

    # patchified_tensor_channelmutli = patchified_tensor.reshape(
    #     batch_size,
    #     num_patches_height * num_patches_width,
    #     patch_size * patch_size * tensor.shape[1],
    # )

    # Reshape and return (B, H*W, C, P, Q) --> This is the expected shape for our application
    patchified_tensor = patchified_tensor.reshape(
        batch_size, num_patches_height * num_patches_width, tensor.shape[1], patch_size, patch_size
    )

    return patchified_tensor


def get_patched_attention_weight(
    patcher: torch.nn.Module,
    decoder_attention_weight: torch.tensor,
    height: int,
    width: int,
) -> torch.tensor:
    """Get the attention weights of the patched attn weight.

    Args:
        patcher (torch.nn.Module): The object that can patchify.
        decoder_attention_weight (torch.tensor): Decoder attention weight.
        height (int): Height of the image.
        width (int): Width of the image.

    Returns:
        torch.tensor: Patchified tensor.
    """

    # The decoder_attention_weight is of the shape of (1, 32768)
    # First project it to right shape (128, 256) and then interpolate
    decoder_attn_tensor = decoder_attention_weight.view(128, 256).unsqueeze(0).unsqueeze(0)
    masked_attn_weight = torch.nn.functional.interpolate(
        decoder_attn_tensor, size=(height, width), mode='bilinear', align_corners=False
    )

    # Patchify the attention weights
    patched_weights = patcher(masked_attn_weight).squeeze()

    # Output tensor is of the shape (N_p, H_p, W_p)
    return patched_weights


# Function to patchify and score patches
def score_attention_patches(
    patcher: torch.nn.Module,
    decoder_attention_weight: torch.tensor,
    height: int,
    width: int,
    keep_percentage: float,
) -> Union[List[int], torch.tensor]:
    """Function to score attention patches.

    Args:
        patcher (torch.nn.Module): The object that can patchify.
        decoder_attention_weight (torch.tensor): Decoder attention weight.
        height (int): Height of the image.
        width (int): Width of the image.
        keep_percentage (float): Percentage to keep in the total num. patches.

    Raises:
        ValueError: When keep percentage is invalid.

    Returns:
        Union[List[int], torch.tensor]: Keep indices, patched attention weight.
    """

    if keep_percentage < 0.0 and keep_percentage > 1.0:
        raise ValueError(
            f'Keep percentage should be between 0 and 1. Received:' f'{keep_percentage}'
        )

    # Get the patched attention weights from (1, 32768)
    patched_attention_weights = get_patched_attention_weight(
        patcher, decoder_attention_weight, height, width
    )

    # Shape of the patched_attenion_weights: (N_p, H_p, W_p)
    flattened_patched_weights = patched_attention_weights.view(
        patched_attention_weights.size(0), -1
    )

    # Compute the score for each patch
    # (Right now, it is direct sum, might need a diff logic)
    patch_scores = torch.sum(flattened_patched_weights, dim=1)

    # Sort the patches in descending order. High score first
    sorted_patch_indices = torch.argsort(patch_scores, descending=True)

    # Calculate the number of patches to keep
    num_patches_to_keep = int(patched_attention_weights.size(0) * keep_percentage)

    # Select the top indexes of patches
    keep_indices = sorted_patch_indices[:num_patches_to_keep]

    return keep_indices, patched_attention_weights


# Function to patchify and assign scores to the individual patches
def get_all_scores_of_patched_decoder(
    patcher: torch.nn.Module,
    decoder_attention_weight: torch.tensor,
    height: int,
    width: int,
) -> Union[List[int], torch.tensor]:
    """
    Function to score attention patches for final Global scoring.
    This function does not sort, rather simply returns the entire
    score of all the patches.

    Args:
        patcher (torch.nn.Module): The object that can patchify.
        decoder_attention_weight (torch.tensor): Decoder attention weight.
        height (int): Height of the image.
        width (int): Width of the image.

    Raises:
        ValueError: When keep percentage is invalid.

    Returns:
        Union[List[int], torch.tensor]: Keep indices, patched attention weight.
    """

    # Get the patched attention weights from (1, 32768)
    patched_attention_weights = get_patched_attention_weight(
        patcher, decoder_attention_weight, height, width
    )

    # Shape of the patched_attenion_weights: (N_p, H_p, W_p)
    flattened_patched_weights = patched_attention_weights.view(
        patched_attention_weights.size(0), -1
    )

    # Compute the score for each patch
    # (Right now, it is direct sum, might need a diff logic)
    patch_scores = torch.sum(flattened_patched_weights, dim=1)

    return patch_scores, patched_attention_weights


def global_score_index_assignment(lut: Dict[str, List], patch_keep_ratio: float) -> Dict[str, List]:
    """
    Method to keep top indices per image basis based on global scoring

    Args:
        lut (Dict[str, List]): Lut consisting of each image and the score for
            each patch index
        patch_keep_ratio (float): The amount of patches to keep

    Returns:
        Dict[str, List]: The lut consisting of each image and the indices that
            are in the top keep percentage indices.
    """
    # Flatten the patch scores into a single list
    all_scores = []
    for scores in lut.values():
        all_scores.extend(scores)

    # Find the threshold for the top patch_keep_ratio of the scores
    all_scores.sort(reverse=True)
    top_keep_threshold = all_scores[int(len(all_scores) * patch_keep_ratio)]

    # Find the top patches per image comapared to the global score
    global_patch_lut = {}
    for image_name, scores in lut.items():
        top_indices = [i for i, score in enumerate(scores) if score >= top_keep_threshold]
        global_patch_lut[image_name] = top_indices

    return global_patch_lut


def store_LUT_as_pkl(
    lut: Dict[str, List],
    data_loader_len: int,
    patch_keep_ratio: float,
    patch_size: int,
    folder_name: None,
):
    """Method to store the look-up table as a pickle file.

    Args:
        lut (Dict[str, List]): LUT that needs to be stored as a pkl file.
        data_loader_len (int): Length of the data loader.
        patch_keep_ratio (float): Patch keep ratio.
        folder_name (None): Name of the folder to store the pkl file.
    """

    if len(lut) == data_loader_len:
        print('All the images in dataloader have an entry')
    else:
        print(f'Expected {len(data_loader_len)} received {len(lut)}' f'Please check the length')
        return

    folder_name_ = (folder_name) + '_patch_lut' if folder_name else 'patches_lut'
    # Create a folder to store the LUT if not there
    if not os.path.exists(folder_name_):
        os.makedirs(folder_name_)

    # Define a file_path_name
    file_path = os.path.join(
        folder_name_,
        f'from_local_pretrained_on_30data_lb1_test'
        f'_local_keepindx_{patch_keep_ratio}_patchsize_'
        f'{patch_size}_.pkl',
    )

    with open(file_path, 'wb') as file:
        pickle.dump(lut, file)

    # Print the message just to be sure
    print(f'LUT Dictionary stored in: {file_path}')


def store_image_patchscoreforvitmae(lut: Dict[str, np.ndarray], folder_name: None):
    """Method to store the 64x128 sized scores for each image.

    Args:
        lut (Dict[str, List]): LUT that needs to be stored as a pkl file.
        folder_name (None): Name of the folder to store the pkl file.
    """

    folder_name_ = (folder_name) + '_patchscoreforvitmae' if folder_name else '_patchscoreforvitmae'
    # Create a folder to store the LUT if not there
    if not os.path.exists(folder_name_):
        os.makedirs(folder_name_)

    # Define a file_path_name
    file_path = os.path.join(
        folder_name_,
        'patchscore_model_fullytrained_hf.pkl',
    )

    with open(file_path, 'wb') as file:
        pickle.dump(lut, file)

    # Print the message just to be sure
    print(f'Dictionary stored in: {file_path}')
