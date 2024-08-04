import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from mpl_toolkits.axes_grid1 import ImageGrid


def view_patchify_attention_weight(
    patcher: torch.nn.Module,
    decoder_attention_weight: torch.tensor,
    patch_size: int,
    height: int,
    width: int,
    folder_name=None,
):
    """
    Function to view the patchified attention weights.
    """

    folder_name_ = (folder_name) + '_Patched_Weights' if folder_name else 'patchify'
    if not os.path.exists(folder_name_):
        os.makedirs(folder_name_)

    # Below commented out code is simply to view the (1, 32768)
    """
    fig, ax = plt.subplots()
    ax.imshow(decoder_attention_weight.view(128, 256).cpu().numpy())
    ax.axis('off')
    ax.set_title(f"Decoder_attention_map_128_256")
    file_path = os.path.join(
        folder_name_,
        f"Decoder_attention_map_128_256.png"
    )
    fig.savefig(file_path)
    """

    # The decoder_attention_weight is of the shape of (1, 32768)
    # First project it to right shape (128, 256) and then interpolate
    decoder_attn_tensor = decoder_attention_weight.view(128, 256).unsqueeze(0).unsqueeze(0)
    masked_attn_weight = torch.nn.functional.interpolate(
        decoder_attn_tensor, size=(height, width), mode='bilinear', align_corners=False
    )

    # Below commented out code is simply to view the interpolated attention_wt
    """
    # The shape would now be (1, 1, 1024, 2048).
    So remove the first batch for the unfold operation
    fig, ax = plt.subplots()
    ax.imshow(masked_attn_weight.squeeze(0).squeeze(0).cpu().numpy())
    ax.axis('off')
    ax.set_title(f"Decoder_attention_map_1024_2048")
    file_path = os.path.join(
        folder_name_,
        f"Decoder_attention_map_1024_2048.png"
    )
    fig.savefig(file_path)
    """
    # Pass the decoder_attn_weight through patchifier
    patched_weights = patcher(masked_attn_weight).squeeze()

    # Plot the patches in the form of grid
    # @NOTE: Each patch is normalized internally in the matplotlib
    # Therefore the patches viewed are normalized

    fig, ax = plt.subplots()
    grid = ImageGrid(
        fig, 111, nrows_ncols=((height // patch_size), (width // patch_size)), axes_pad=0.1
    )
    for i, ax in enumerate(grid):
        patch = patched_weights[i].cpu().numpy()
        # Below code is commented out, but can be uncommented to save each patch
        """
        torchvision.utils.save_image(
            torchvision.utils.make_grid(
                patched_weights[i], normalize=True, scale_each=True),
                os.path.join(folder_name_, f"patch_{i}.png"
            )
        )
        """
        ax.imshow(patch)
        ax.axis('off')

    plt.show()

    # Save the normalized each patched image to view
    file_path = os.path.join(folder_name_, 'attn_weights_patched_results.png')
    fig.savefig(file_path)


# In order to unpatchify and view to verify it does not change the weights
def construct_patched_weights_back(patched_weights: torch.tensor, folder_name=None):
    """
    Function to construct the patchified attention weights back to normal image.
    """
    folder_name_ = (
        (folder_name) + '_Restored_Patched_Weights' if folder_name else 'Restored_Patched_Weights'
    )
    if not os.path.exists(folder_name_):
        os.makedirs(folder_name_)
    # Patched_weights is of the shape (32, 256, 256)
    reshaped_patches = patched_weights.view(4, 8, 256, 256)
    # The below tensor is of the shape (1, 1024, 2048)
    original_attn_weight = reshaped_patches.transpose(1, 2).contiguous().view(1, 1024, 2048)
    # For visualization of the tensors
    torchvision.utils.save_image(
        torchvision.utils.make_grid(original_attn_weight, normalize=True, scale_each=True),
        os.path.join(folder_name_, '_full_patch.png'),
    )
    """
    fig, ax = plt.subplots()
    ax.imshow(original_attn_weight.squeeze(0).cpu().numpy())
    ax.axis('off')
    ax.set_title(f"Reconstructed_decoder_weights")
    file_path = os.path.join(folder_name_,f"Reconstructed_decoder_weights.png")
    fig.savefig(file_path)
    """


# Below code is to reconstruct the patched parts back to the image
def construct_patched_images(patched_color_img: torch.tensor, folder_name=None):
    """Construct the patched image back together.

    Args:
        patched_color_img (torch.tensor): Patched color image.
        folder_name (_type_, optional): name of the folder to save.
            Defaults to None.
    """
    folder_name_ = (
        (folder_name) + '_Restored_Patched_Images' if folder_name else 'Restored_Patched_Images'
    )
    if not os.path.exists(folder_name_):
        os.makedirs(folder_name_)

    # The patched_parts is of the shape (32, 3, 256, 256)
    reshaped_color_patches = patched_color_img.view(4, 8, 3, 256, 256)
    reshaped_color_patches = reshaped_color_patches.permute(0, 3, 1, 4, 2)
    original_img = reshaped_color_patches.contiguous().view(1024, 2048, 3).permute(2, 0, 1)
    torchvision.utils.save_image(
        torchvision.utils.make_grid(original_img, normalize=True, scale_each=True),
        os.path.join(folder_name_, 'recons_original_image.png'),
    )


def view_patchify_original_image(
    patcher: torch.nn.Module,
    img_tensor: torch.tensor,
    height: int,
    width: int,
    patch_size: int,
    folder_name: None,
):
    """
    View patchified image back to original image.
    """
    folder_name_ = (folder_name) + '_Patched_Images' if folder_name else '_Patchify'
    if not os.path.exists(folder_name_):
        os.makedirs(folder_name_)

    feed_dict_img = img_tensor.unsqueeze(0)

    # Pass the image tensor through the patchifier
    patched_parts = patcher(feed_dict_img).squeeze()

    # Plot the patched images in grid
    fig, ax = plt.subplots()
    grid = ImageGrid(
        fig, 111, nrows_ncols=((height // patch_size), (width // patch_size)), axes_pad=0.1
    )
    for i, ax in enumerate(grid):
        patch = patched_parts[i].permute(1, 2, 0).cpu().numpy()
        # Below is to save the image using torchvision
        """
        torchvision.utils.save_image(
            torchvision.utils.make_grid(
                patched_parts[i], normalize=True, scale_each=True),
                os.path.join(folder_name_, f"torch_save_{i}.png"
            )
        )
        """
        rescaled_image = (patch - np.min(patch)) / (np.max(patch) - np.min(patch)) * 255.0
        normalize_patch = rescaled_image / 255.0
        ax.imshow(normalize_patch)
        ax.axis('off')
    plt.show()

    file_path = os.path.join(folder_name_, 'patched_results.png')
    fig.savefig(file_path)


def view_high_score_indexes(
    patched_attention_weight: torch.tensor,
    keep_indices: List[int],
    patchifier: torch.nn.Module,
    original_img: torch.tensor,
    patch_size: int,
    folder_name: None,
):
    """
    Function to view high scoring indices.
    """
    folder_name_ = (folder_name) + '_kept_indices' if folder_name else '_Kept_Indices'
    if not os.path.exists(folder_name_):
        os.makedirs(folder_name_)
    _, height, width = original_img.shape
    # Patched attention weight is of the shape (N_p, H_p, W_p)
    fig, ax = plt.subplots()
    grid = ImageGrid(
        fig, 111, nrows_ncols=((height // patch_size), (width // patch_size)), axes_pad=0.1
    )

    for indx, ax in enumerate(grid):
        if indx in keep_indices:
            # The patches in the keep indices needn't be changed
            patch = patched_attention_weight[indx].cpu().numpy()
        else:
            # Simply black out the patches not in the list
            patch = np.zeros_like(patched_attention_weight[indx].cpu().numpy())

        ax.imshow(patch)  # cmap='gray'
        ax.axis('off')

    plt.show()

    file_path = os.path.join(folder_name_, '_kept_indices.png')
    fig.savefig(file_path)

    # The image must be having the batch dimension
    feed_dict_img = original_img.unsqueeze(0)

    # Pass the image tensor through the patchifier
    patched_parts = patchifier(feed_dict_img).squeeze()

    # Plot the patched images in grid
    fig, ax = plt.subplots()
    grid = ImageGrid(
        fig, 111, nrows_ncols=((height // patch_size), (width // patch_size)), axes_pad=0.1
    )
    for indx, ax in enumerate(grid):
        if indx in keep_indices:
            patch = patched_parts[indx].permute(1, 2, 0).cpu().numpy()
            # Below is to save the image using torchvision
            """
            torchvision.utils.save_image(
                torchvision.utils.make_grid(
                    patched_parts[indx], normalize=True, scale_each=True),
                    os.path.join(folder_name_, f"torch_save_{i}.png"
                )
            )
            """
            rescaled_image = (patch - np.min(patch)) / (np.max(patch) - np.min(patch)) * 255.0
            normalize_patch = rescaled_image / 255.0
        else:
            normalize_patch = np.zeros_like(patched_attention_weight[indx].cpu().numpy())
            # Below is to save the image using torchvision
            """
            torchvision.utils.save_image(
                torchvision.utils.make_grid(
                    normalize_patch, normalize=True, scale_each=True),
                    os.path.join(folder_name_, f"torch_save_{i}.png"
                )
            )
            """
        ax.imshow(normalize_patch)
        ax.axis('off')
    plt.show()

    file_path = os.path.join(folder_name_, 'kept_color_patches.png')
    fig.savefig(file_path)
