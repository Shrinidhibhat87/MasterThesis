import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# Keep here refers to the indexes out of the 100 query indices, where confidence > thres
# Function is meant to get the decoder attention maps based on the weights
def get_cum_decoder_weights(
    attn_weight: torch.tensor,
    keep_indexes=None,
) -> torch.tensor:
    # Get the indexes of the weights that would be added together
    index_range = (
        keep_indexes.nonzero() if keep_indexes is not None else list(range(attn_weight.size(1)))
    )

    # Initialize the tensor to which weights will be added
    total_selected_query_tensor = torch.zeros(1, 32768)

    # Iterate over the attention weight tensor
    for idx in index_range:
        """
        # Possible to normalize it before adding or simply adding it
        total_selected_query_tensor += F.normalize(
            attn_weight[0, idx].cpu(), p=2, dim=1
        )
        """
        total_selected_query_tensor += attn_weight[0, idx].cpu()

    # Normalize the total selected_query tensors before returning it
    normalized_selected_queries_tensor = F.normalize(total_selected_query_tensor, p=2, dim=1)

    return normalized_selected_queries_tensor


# Function is meant to get the decoder attention maps based on the weights
def get_cum_unnormalized_decoder_weights(
    attn_weight: torch.tensor,
    keep_indexes=None,
) -> torch.tensor:
    # Get the indexes of the weights that would be added together
    index_range = (
        keep_indexes.nonzero() if keep_indexes is not None else list(range(attn_weight.size(1)))
    )

    # Initialize the tensor to which weights will be added
    total_selected_query_tensor = torch.zeros(1, 32768)

    # Iterate over the attention weight tensor
    for idx in index_range:
        """
        # Possible to normalize it before adding or simply adding it
        total_selected_query_tensor += F.normalize(
            attn_weight[0, idx].cpu(), p=2, dim=1
        )
        """
        total_selected_query_tensor += attn_weight[0, idx].cpu()

    return total_selected_query_tensor


# The decoder_attn_weight passed here must be the (100, 32768)
def visualize_decoder_attn(
    attn_weight: torch.tensor,
    keep_indexes=None,
    folder_name=None,
    concat_attention_map=False,
    feed_dict_img=None,
):
    folder_name_ = folder_name if folder_name else 'attention_map'
    if not os.path.exists(folder_name_):
        os.makedirs(folder_name_)

    index_range = (
        keep_indexes.nonzero() if keep_indexes is not None else list(range(attn_weight.size(1)))
    )
    # Create subplots for visualization
    fig, ax = plt.subplots()

    # Values of height and width by default
    H, W = 1024, 2048
    if feed_dict_img is not None:
        # Get the height and width of the actual image
        C, H, W = feed_dict_img.shape
        """
        file_path = os.path.join(folder_name_, "Image.png")
        torchvision.utils.save_image(
            torchvision.utils.make_grid(
                feed_dict_img, normalize=True, scale_each=True
            ),
                file_path
        )
        """

    total_selected_query_tensor = torch.zeros(1, 32768)

    for idx in index_range:
        # Below logic is to normalize each and every patch
        """
        total_selected_query_tensor += F.normalize(
            attn_weight[0, idx].cpu(), p=2, dim=1
        )
        """
        total_selected_query_tensor += attn_weight[0, idx].cpu()
        # Visualize the attention map
        ax.imshow(attn_weight[0, idx].view(128, 256).cpu().numpy())

        # Turn off axis
        ax.axis('off')

        # Set the title with the index value
        ax.set_title(f'Attention Map for index {idx.item()}')

        # Save the figure with the unique name based on the index value
        """
        file_path = os.path.join(
            folder_name_,
            f"Attention_map_{idx.item()}.png"
        )
        fig.savefig(file_path)
        """

    if concat_attention_map:
        # Collect the tensors in a list
        all_selected_query_tensors = [attn_weight[0, idx] for idx in index_range]
        # Stack the tensors along a new dimension
        stacked_tensors = torch.stack(all_selected_query_tensors)

        # Mean them all along the batch dimension to one tensor
        mean_tensor = torch.mean(stacked_tensors, dim=0)
        # Visalize and save the figure
        """
        # Sum them all along the batch dimension to one tensor
        combined_tensor = torch.sum(stacked_tensors, dim=0)
        ax.imshow(combined_tensor.view(128, 256).cpu().numpy())
        """
        ax.imshow(mean_tensor.view(128, 256).cpu().numpy())
        ax.axis('off')
        ax.set_title('Cumulative attention map: Mean')
        """
        file_path = os.path.join(
            folder_name_,
            "Cumulative_attention_map_mean.png"
        )
        fig.savefig(file_path)
        """

    # Normalize the total sum of the selected queries and simply visualize it
    normalized_selected_queries_tensor = F.normalize(total_selected_query_tensor, p=2, dim=1)
    h_w_normalized_tensor = (
        normalized_selected_queries_tensor.view(128, 256).unsqueeze(0).unsqueeze(0)
    )
    H_W_decoder_attn_weight_tensor = torch.nn.functional.interpolate(
        h_w_normalized_tensor, size=(H, W), mode='bilinear', align_corners=False
    )
    ax.imshow(H_W_decoder_attn_weight_tensor.squeeze(0).squeeze(0).numpy())
    # ax.imshow(normalized_selected_queries_tensor.view(128, 256).numpy())
    ax.axis('off')
    ax.set_title('Normalized Attention Map (L2)')
    """
    file_path = os.path.join(folder_name_, f"Normalized_attention_map.png")
    fig.savefig(file_path, dpi=300)
    torchvision.utils.save_image(
        torchvision.utils.make_grid(
            H_W_decoder_attn_weight_tensor.squeeze(0).squeeze(0),
            normalize=True, scale_each=True
        ),
            file_path
    )
    """


def visualize_self_attn(
    attn_weight: torch.tensor,
    keep_indexes=None,
    folder_name=None,
):
    folder_name_ = folder_name if folder_name else 'self_attention_maps'

    if not os.path.exists(folder_name_):
        os.makedirs(folder_name_)

    for head in range(attn_weight.size(1)):
        self_attn_wt = attn_weight[0, head]
        index_range = (
            keep_indexes.nonzero()
            if keep_indexes is not None
            else list(range(self_attn_wt.size(1)))
        )
        fig, ax = plt.subplots()

        for idx in index_range:
            ax.imshow(self_attn_wt[idx].view(10, 10).cpu().numpy())
            ax.axis('off')
            """
            ax.set_title(f"Self attention map for head"
                         f"{head}, index {idx.item()}")
            file_path = os.path.join(
                folder_name_,
                f"Self_Attention_map_head_{head}_and_{idx.item()}.png"
            )
            fig.savefig(file_path)
            """


def visualize_all_masked_attention_weight(
    decoder_masked_attn: List[torch.tensor], folder_name=None
):
    folder_name_ = folder_name if folder_name else 'all_decoder_attention_maps'
    if not os.path.exists(folder_name_):
        os.makedirs(folder_name_)
    index_range = [
        list(range(decoder_masked_attn[0].size(1)))
    ]  # [i for i in range(decoder_masked_attn[0].size(1))]
    # Create a dictionary that for each shape
    # has a mapping of what the view/resize should be
    shape_mapping = {2048: (32, 64), 8192: (64, 128), 32768: (128, 256)}
    # Below is a very hacky way of adding tensors
    # definitely needs to be changed
    add_decoder_layers_2048 = torch.zeros((1, 100, 2048))
    add_decoder_layers_8192 = torch.zeros((1, 100, 8192))
    add_decoder_layers_32768 = torch.zeros((1, 100, 32768))
    # Create subplots for visualization
    fig, ax = plt.subplots()
    # Iterate over each layer output of the cross attention decoder layer
    for layer_id in range(len(decoder_masked_attn)):
        # current_decoder_layer = decoder_masked_attn[layer_id]
        all_query_tensors = [decoder_masked_attn[layer_id][0, idx] for idx in index_range]
        stacked_tensors = torch.stack(all_query_tensors)
        combined_tensor = torch.sum(stacked_tensors, dim=0)
        (h, w) = shape_mapping.get(decoder_masked_attn[layer_id].size(-1))
        if decoder_masked_attn[layer_id].size(-1) == 2048:
            add_decoder_layers_2048 += decoder_masked_attn[layer_id].cpu()
        elif decoder_masked_attn[layer_id].size(-1) == 8192:
            add_decoder_layers_8192 += decoder_masked_attn[layer_id].cpu()
        elif decoder_masked_attn[layer_id].size(-1) == 32768:
            add_decoder_layers_32768 += decoder_masked_attn[layer_id].cpu()

        ax.imshow(combined_tensor.view(h, w).cpu().numpy())
        ax.axis('off')
        ax.set_title(f'Masked Attention Attention Map: {layer_id}')
        file_path = os.path.join(folder_name_, f'Masked_attention_map_{layer_id}.png')
        fig.savefig(file_path)

    # Lets visualize and save the plots of total sum of the scaled attention maps
    add_decoder_layers_2048 = add_decoder_layers_2048.squeeze(0)
    summed_tensor_2048 = torch.sum(add_decoder_layers_2048, axis=0)
    ax.imshow(summed_tensor_2048.view(32, 64).cpu().numpy())
    ax.axis('off')
    ax.set_title(f'Cumulative Masked attention: {2048}')
    file_path = os.path.join(folder_name_, f'Cumulative_masked_attention_map_{2048}.png')
    fig.savefig(file_path)

    add_decoder_layers_8192 = add_decoder_layers_8192.squeeze(0)
    summed_tensor_8192 = torch.sum(add_decoder_layers_8192, axis=0)
    ax.imshow(summed_tensor_8192.view(64, 128).cpu().numpy())
    ax.axis('off')
    ax.set_title(f'Cumulative Masked attention: {8192}')
    file_path = os.path.join(folder_name_, f'Cumulative_masked_attention_map_{8192}.png')
    fig.savefig(file_path)

    add_decoder_layers_32768 = add_decoder_layers_32768.squeeze(0)
    summed_tensor_32768 = torch.sum(add_decoder_layers_32768, axis=0)
    ax.imshow(summed_tensor_32768.view(128, 256).cpu().numpy())
    ax.axis('off')
    ax.set_title(f'Cumulative Masked attention: {32768}')
    file_path = os.path.join(folder_name_, f'Cumulative_masked_attention_map_{32768}.png')
    fig.savefig(file_path)


def plot_show_results(
    color_palette: List, image: torch.tensor, segmentation_map: torch.tensor, name: str
):
    # Comments copied from base trainer
    """
    # To store the color image of one image:
    torchvision.utils.save_image(
        torchvision.utils.make_grid(
            image, normalize=True, scale_each=True
        ),
        'Image_2.png'
    )
    # To visualize the output of one image
    # Move the image to cpu
    image = (feed_dict['img'][0]).detach().cpu().numpy()[::-1]
    # The palette relies on the hugging face configuration file
    # to provide classes
    color_palette = [
        list(np.random.choice(range(256), size=3))
        for _ in range(len(model.config.id2label))
    ]
    plot_show_results(
        color_palette,
        np.transpose(image, (1, 2, 0)),
        predicted_seg_map.cpu(),
        model,
        name="Predicted_seg",
    )
    plot_show_results(
        color_palette,
        np.transpose(image, (1, 2, 0)),
        feed_dict["gt_seg_map"][0].detach().cpu().numpy(),
        model,
        name="True_seg",
    )
    """

    seg = segmentation_map
    # height, width, 3
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    palette = np.array(color_palette)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    # Save the plot_locally
    plt.savefig(name)
    plt.show()
