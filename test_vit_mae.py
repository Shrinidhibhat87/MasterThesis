import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchpack.distributed as dist
import torchvision
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch import nn
from torchpack.utils.logging import logger
from transformers import ViTImageProcessor, ViTMAEForPreTraining, ViTMAEModel
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEDecoder,
    ViTMAEEncoder,
    ViTMAEForPreTrainingOutput,
    ViTMAEModelOutput,
    ViTMAEPatchEmbeddings,
    get_2d_sincos_pos_embed,
)

from dataloader.CityScapeDataloader import LoadAndGetDataloader
from patchify.patchify import Patchify

os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def dist_init() -> None:
    try:
        torch.distributed.init_process_group(backend='nccl')
        assert torch.distributed.is_initialized()
    except Exception:
        # use torchpack
        from torchpack import distributed as dist

        dist.init()
        os.environ['RANK'] = f'{dist.rank()}'
        os.environ['WORLD_SIZE'] = f'{dist.size()}'
        os.environ['LOCAL_RANK'] = f'{dist.local_rank()}'


def setup_cuda_env() -> None:
    if not torch.distributed.is_initialized():
        dist_init()
    # torch cudnn benchmark is supposed to improve performance.
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())
    # torch.cuda.set_device(7)


class ViTMAECustomEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViTMAEPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1],
            int(self.patch_embeddings.num_patches**0.5),
            add_cls_token=True,
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape  # 1, 196, 768: 196 because of flattened 14*14
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(
                batch_size, seq_length, device=sequence.device
            )  # noise in [0, 1] # how to move sequence of device from cpu to gpu
        # The noise tensor is sorted such that the indices of the tensor values with low values are first and so on
        # sort noise for each sample
        # This was first with descending=False and now descending=True
        ids_shuffle = torch.argsort(noise, dim=1, descending=True)
        # The indices are then sorted again to get the indice value in a sorted manner
        # Hence ids_restore will restore the original order of the noise tensor
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(
            sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim)
        )

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask. This will have 147 1s and 49 0s (size 196)
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def attn_based_masking_backup(self, sequence, attn_noise, replace_percent=0.15):
        """
        Sampling of patches/points based on randomness along with attention heuristic

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            attn_noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
            replace_percent (Float value): The percentage to indicate how many top attention based
                noises needs to replace random noise.
        """
        batch_size, seq_length, dim = sequence.shape  # 1, 196, 768: 196 because of flattened 14*14
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        """
        The idea here is that we have random noise created and also have attention noise.
        Replace the top x% of random noise with attention indices and scores.
        Take the top len_keep values
        """
        noise_rand = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]
        # Normalize the attention based noise between the value [0-1]
        attn_noise_norm = (attn_noise - torch.min(attn_noise)) / (
            torch.max(attn_noise) - torch.max(attn_noise)
        )
        # Get the number of indices to replace in random noise.
        num_to_replace = int(replace_percent * attn_noise_norm.numel())
        # Get the top scoring indices
        _, top_indices = torch.topk(attn_noise_norm.squeeze(), num_to_replace, dim=0, largest=True)
        # Do an inplace replace
        noise_rand[0, top_indices] = attn_noise_norm[0, top_indices]
        # The noise tensor is sorted such that the indices of the tensor values with low values are first and so on
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise_rand, dim=1, descending=True)
        # The indices are then sorted again to get the indice value in a sorted manner
        # Hence ids_restore will restore the original order of the noise tensor
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(
            sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim)
        )

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask. This will have 147 1s and 49 0s (size 196)
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def attn_based_masking(self, sequence, attn_noise):
        """
        Sampling of patches/points based on attention based masking.
        Selects the patches which have high attention scores.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            attn_noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape  # 1, 196, 768: 196 because of flattened 14*14
        len_keep = int(seq_length * (1 - self.config.mask_ratio))
        # Normalize the attention based noise between the value [0-1]
        attn_noise_norm = (attn_noise - torch.min(attn_noise)) / (
            torch.max(attn_noise) - torch.max(attn_noise)
        )
        ids_shuffle = torch.argsort(attn_noise_norm, dim=1, descending=True)
        # The indices are then sorted again to get the indice value in a sorted manner
        # Hence ids_restore will restore the original order of the noise tensor
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(
            sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim)
        )

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask. This will have 147 1s and 49 0s (size 196)
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(self, pixel_values, noise=None):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        # embeddings, mask, ids_restore = self.random_masking(embeddings, noise)
        # masking bsaed on attention based heuristic
        embeddings, mask, ids_restore = self.attn_based_masking(embeddings, attn_noise=noise)

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore


class ViTCustomMAEModel(ViTMAEModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTMAECustomEmbeddings(config)
        self.encoder = ViTMAEEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ViTMAEModelOutput]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, mask, ids_restore = self.embeddings(pixel_values, noise=noise)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        return ViTMAEModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ViTModelCustomMasking(ViTMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vit = ViTCustomMAEModel(config)
        self.decoder = ViTMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ViTMAEForPreTrainingOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore)
        logits = (
            decoder_outputs.logits
        )  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        loss = self.forward_loss(pixel_values, logits, mask)

        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ViTMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VITMAE:
    """
    Class to test the ViT-MAE model and see how the output result is.
    Default version for now.
    """

    def __init__(
        self,
        full_train_path: str,
        dataset_folder_path: str,
        vit_mae_feature_extractor_path: str,
        vit_mae_pretrained_path: str,
        dataset_name: str,
        output_folder_path: str,
        masked_attn_scores_path: str,
        attn_lut_path: str,
        device: str,
    ) -> None:
        self.mae_pretrained_path = vit_mae_pretrained_path
        self.mae_feature_extractor_path = vit_mae_feature_extractor_path
        self.dataset_folder_path = dataset_folder_path
        self.dataset_name = dataset_name
        self.full_train_path = full_train_path
        self.output_folder_path = output_folder_path
        self.device = device
        self.images_as_tensors = []

        # Load the attn scores from a pkl file
        with open(masked_attn_scores_path, 'rb') as file:
            self.masked_attn_score_LUT = pickle.load(file)

        # Load the local attn scored patches
        with open(attn_lut_path, 'rb') as file:
            self.local_attn_scores = pickle.load(file)

        self.transform = transforms.Compose([transforms.PILToTensor()])

        dataloader = LoadAndGetDataloader(
            root_dir=self.dataset_folder_path, batch_size=1, num_workers=8, shuffle=False
        )

        self.train_dataloader, _ = dataloader.create_dataset_and_load(
            train_transform=self.transform, valid_transform=self.transform
        )

        # Define a patchifier of size 256 because we cant do 224 patches
        self.patchifier_images = Patchify(patchsize=256)
        self.patchifier_attnscores = Patchify(patchsize=16)

        self.vit_image_processor = ViTImageProcessor.from_pretrained(
            self.mae_feature_extractor_path
        )

        # Define the pretrained model
        self.vitmae_model = ViTModelCustomMasking.from_pretrained(self.mae_pretrained_path)

        # Move the model to CUDA
        self.vitmae_model.to(self.device)

    def store_images(self, image, image_name, img_type):
        folder_name_ = image_name
        if not os.path.exists(folder_name_):
            os.makedirs(folder_name_)

        torchvision.utils.save_image(
            torchvision.utils.make_grid(image, normalize=True, scale_each=True),
            os.path.join(folder_name_, f'{img_type}_.png'),
        )

    def visualize_and_store_images(self, pixel_values, model, noise, image_name):
        # Forward pass
        outputs = model(pixel_values, noise=noise)
        y = model.unpatchify(outputs.logits)
        # y = torch.einsum('nchw->nhwc', y).detach().cpu()

        # visualize the mask
        mask = outputs.mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * 3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)

        # masked image
        im_masked = pixel_values * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = pixel_values * (1 - mask) + y * mask

        # Original image
        self.store_images(pixel_values[0], image_name, 'Original')

        # Masked image
        self.store_images(im_masked[0], image_name, 'Masked')

        # Reconstruction
        self.store_images(y[0], image_name, 'Reconstruction')

        # Reconstruction + visible
        self.store_images(im_paste[0], image_name, 'Recons_Visible')

        return outputs

    def do_operations(self) -> None:
        """
        Function that does the following:
        1. Load each data from the dataloader.
        2. Pass it through the ViT MAE model (will have to resize this and convert this back)
        3. Output can be stored or not be stored based on choice.
        """
        # Check the time to see how long this took
        start_time = time.time()
        # Create an iterator to iterate over each object
        for _, feed_dict in enumerate(self.train_dataloader):
            # Get image name
            full_image_path = feed_dict['img_path']
            image_name = os.path.splitext(os.path.basename(full_image_path[0]))[0]
            # s_time = time.time()
            # Need to get a method to also get the name of the subfolder
            subfolder_name = os.path.dirname(full_image_path[0]).split(os.path.sep)[-1]
            # Get the image
            image = feed_dict['img']
            # Get the high scoring patches (local for now) from the lut
            # high_scoring_index = self.local_attn_scores[image_name]
            # Using the above information, we get the attn_scores using the LUT.
            attn_score_np = self.masked_attn_score_LUT[image_name]
            """
            From the attn score of size (64,128), we map this to the entire image.
            Lets consider a case where we are constructing each image, per patch
            For the first patch (0,0)->(256,256) : The attention score of (0,0)->(16,16)
            The easiest approach right now is to patchify using our approach
            """
            # Convert the numpy array to a torch tensor for patchifying
            attn_score_t = torch.from_numpy(attn_score_np).unsqueeze(0).unsqueeze(0)
            # Patchify image and attn_score_t
            patchified_imgs = self.patchifier_images(image.to(torch.float32)).squeeze()
            patchified_imgs = patchified_imgs.to(torch.uint8)
            patchified_attn_scores = self.patchifier_attnscores(attn_score_t).squeeze(0).squeeze(1)
            """
            Once we have patchified the score and the image, for each image,
            we construct the noise individually, by removing information, but also
            knowing its position, if not interpolation.
            """
            # Because we need to reconstruct the individual patches back together
            reconstr_patched_parts = []
            # Create a flag if the noise is random or bsaed on attention scores
            # random_noise_flag = False
            for index in range(patchified_imgs.shape[0]):
                # Each patch is of size (3, 256, 256) and needs to be preprocessed
                pixel_values = (
                    self.vit_image_processor(patchified_imgs[index], return_tensors='pt')
                    .to(self.device)
                    .pixel_values
                )
                # If the patch index in the high scoring patches, use random masking with a lower mask ratio
                # Therefore we are sending more information to the cloud.
                # if index in high_scoring_index:
                if True:
                    # The size of the corresponding attn score is (16,16)
                    # This has to be made into 192 sized vector (14*14).
                    # @NOTE: For now, simply interpolate it and flatten.
                    attn_score = torch.nn.functional.interpolate(
                        patchified_attn_scores[index].unsqueeze(0).unsqueeze(0),
                        size=(14, 14),
                        mode='bilinear',
                        align_corners=False,
                    )

                    # Now flatten this tensor to get a long vector of size (1, 192)
                    attn_score_noise = (
                        torch.flatten(attn_score.squeeze(0).squeeze(0)).unsqueeze(0).to(self.device)
                    )

                    # Pass this onto the forward method as noise
                    # self.vitmae_model.config.mask_ratio = 0.70
                    outputs = self.vitmae_model(pixel_values, noise=attn_score_noise)
                    # Make the masking ratio to a lower extent to send more info
                    """
                    The below function is used when we want to visualize each patch
                    Here we are also changing the masking ratio because the imp patches
                    have lesser information masked
                    """
                    """
                    outputs = self.visualize_and_store_images(
                        pixel_values,
                        self.vitmae_model,
                        attn_score_noise, #None
                        image_name
                    )
                    """
                else:
                    self.vitmae_model.config.mask_ratio = 0.80
                    outputs = self.vitmae_model(pixel_values)
                    # outputs = self.visualize_and_store_images(
                    #     pixel_values,
                    #     self.vitmae_model,
                    #     None,
                    #     image_name
                    # )
                # Because we want to investigate each patch, let us create a folder/image
                """
                folder_name = "reconstructed_data_"
                full_folder_path = os.path.join(folder_name, f"_{random_noise_flag}")
                if not os.path.exists(full_folder_path):
                    os.makedirs(full_folder_path)
                """
                # Get the unpatchified output
                unpatch_output = self.vitmae_model.unpatchify(outputs.logits)
                # Get the mask (0: keep and 1: remove)
                mask = outputs.mask.detach()
                mask = mask.unsqueeze(-1).repeat(1, 1, self.vitmae_model.config.patch_size**2 * 3)
                unpatch_mask = self.vitmae_model.unpatchify(mask)
                """
                Here we will have the masks, images and other information saved for ref
                """
                # Masked image
                # im_masked = pixel_values * (1 - unpatch_mask)
                # Reconstructed image
                patch_recons = pixel_values * (1 - unpatch_mask) + unpatch_output * unpatch_mask
                """
                # Save the masked info on disk
                self.store_images(
                    im_masked[0], f"{image_name}_mr0.5", f"{index}_masked_{self.vitmae_model.config.mask_ratio}"
                )
                # Save the reconstructed on disk
                self.store_images(
                    unpatch_output[0], f"{image_name}_mr0.5", f"{index}_recons_{self.vitmae_model.config.mask_ratio}"
                )
                # Save the original on disk
                self.store_images(
                    pixel_values[0], f"{image_name}_mr0.5", f"{index}_original_{self.vitmae_model.config.mask_ratio}"
                )
                # Save the original on disk
                self.store_images(
                    patch_recons[0], f"{image_name}_mr0.5", f"{index}_reconst_visible_{self.vitmae_model.config.mask_ratio}"
                )
                """
                # Because this image is going to be (224, 224) and we need (256,256)
                patch_recons = torch.nn.functional.interpolate(
                    patch_recons, size=(256, 256), mode='bilinear', align_corners=False
                )
                # Append this patch information to a list
                reconstr_patched_parts.append(patch_recons)

            """
            Once all the individual patches have been reconstructed, we are now going to
            build the images back again
            """
            reconstructed_img = torch.stack(reconstr_patched_parts, dim=0)
            reshaped_recons_img = reconstructed_img.view(4, 8, 3, 256, 256)
            reshaped_recons_img = reshaped_recons_img.permute(0, 3, 1, 4, 2)
            original_img_post_patchify = (
                reshaped_recons_img.contiguous().view(1024, 2048, 3).permute(2, 0, 1)
            )
            """
            # Pass the image through the preprocessor
            pixel_values = self.vit_image_processor(image, return_tensors="pt").pixel_values
            # Pass the pixel values through the model
            outputs = self.vitmae_model(pixel_values)
            # Get the unpatchified output
            # outputs.logits have the shape (batch_size, num_patches, patch_size**2 * num_channels)
            unpatch_output = self.vitmae_model.unpatchify(outputs.logits)
            # The mask is also important for reconstruction
            mask = outputs.mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, self.vitmae_model.config.patch_size**2 *3)
            # Unpatchify the mask as well
            mask = self.vitmae_model.unpatchify(mask)
            # Reconstructed image
            recons_image = pixel_values * (1 - mask) + unpatch_output * mask
            """

            # Detach the tensor to convert to numpy
            folder_name_ = '/data/reconstructed_cityscapes_trainds_attentionmask/leftImg8bit/train'
            # Create a folder to store and see the output
            # folder_name_ = image_name
            img_arr = original_img_post_patchify.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
            img = (img_arr * 255).astype(np.uint8)
            image = Image.fromarray(img)
            # full_folder_path = os.path.join(folder_name_, subfolder_name)
            # if not os.path.exists(full_folder_path):
            # os.makedirs(full_folder_path)
            image_path = os.path.join(folder_name_, subfolder_name, f'{image_name}.png')
            # image_path = os.path.join(folder_name_, "_fullyreconstructed_img.png")
            image.save(image_path)

        end_time = time.time()

        print(
            f'\nTotal time for the reconstruction of the images:'
            f'{end_time-start_time:.2f} seconds'
        )

    def go_do_your_magic(self) -> None:
        self.do_operations()


def main():
    # Main function to capture the image sizes
    parser = argparse.ArgumentParser()
    default_output_dir = Path.cwd()
    parser.add_argument(
        '--full_train_path',
        type=str,
        default='/data/datasets/PytorchDatasets/CityScapes-pytorch/leftImg8bit/train/',
        help='Path to image folder',
    )
    parser.add_argument(
        '--dataset_folder_path',
        type=str,
        default='/data/datasets/PytorchDatasets/CityScapes-pytorch/',
        help='Path to image folder',
    )

    parser.add_argument(
        '--vit_mae_pretrained_path',
        type=str,
        default='facebook/vit-mae-base',
        help='Path to the hf pre-trained model path',
    )

    parser.add_argument(
        '--dataset_name', type=str, default='CityScapes', help='Name of the dataset'
    )

    parser.add_argument(
        '--vit_mae_feature_extractor_path',
        type=str,
        default='facebook/vit-mae-base',
        help='Path to the hf pre-trained feature extractor',
    )

    parser.add_argument(
        '--output_folder_path',
        type=str,
        default=default_output_dir,
        help='Output path to where you want to store the size information',
    )

    parser.add_argument(
        '--masked_wt_attnscores',
        type=str,
        default='pscoreforvitmae_patchscoreforvitmae/patchscore_model_fullytrained_hf.pkl',
        help='Path to where all the attention scores of an image are stored',
    )

    parser.add_argument(
        '--local_attn_scorelut',
        type=str,
        default='patches_lut_patch_lut/attn_lut_from_hf/from_hf_pretrained_on_all_data_lb1_local_keepindx_0.3_patchsize_256_.pkl',
        help='Path to where all the attention scores of an image are stored',
    )

    args = parser.parse_args()

    # Check the path when loading the pkl file if everything is valid.
    if not os.path.isfile(args.masked_wt_attnscores):
        raise ValueError(f'Given pkl path {args.masked_wt_attnscores} is invalid.')
    if not args.masked_wt_attnscores.endswith('.pkl'):
        raise ValueError(f'Given path {args.masked_wt_attnscores} is not pkl file.')

    # Check if the lut path for attention scores are also valid.
    if not os.path.isfile(args.local_attn_scorelut):
        raise ValueError(f'Given pkl path {args.local_attn_scorelut} is invalid.')
    if not args.local_attn_scorelut.endswith('.pkl'):
        raise ValueError(f'Given path {args.local_attn_scorelut} is not pkl file.')

    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f'Current device: {torch.cuda.current_device()}')
        setup_cuda_env()
    else:
        device = 'cpu'
        logger.warning('No accelerator found, proceeding with CPU!')

    vit_mae = VITMAE(
        full_train_path=args.full_train_path,
        dataset_folder_path=args.dataset_folder_path,
        vit_mae_feature_extractor_path=args.vit_mae_feature_extractor_path,
        vit_mae_pretrained_path=args.vit_mae_pretrained_path,
        dataset_name=args.dataset_name,
        output_folder_path=args.output_folder_path,
        masked_attn_scores_path=args.masked_wt_attnscores,
        attn_lut_path=args.local_attn_scorelut,
        device=device,
    )

    vit_mae.go_do_your_magic()


if __name__ == '__main__':
    main()
