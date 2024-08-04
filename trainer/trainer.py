import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.hooks
from tqdm import tqdm
from transformers import Mask2FormerImageProcessor

from metric.average_metrics import AverageMeter, AverageMeterRelative
from metric.mean_iou import MeanIoU
from patchify.decoder_attn_map import get_cum_decoder_weights
from patchify.patchify import (
    Patchify,
    get_all_scores_of_patched_decoder,
    global_score_index_assignment,
    score_attention_patches,
    store_LUT_as_pkl,
)
from patchify.view_patchify import (
    view_high_score_indexes,
    view_patchify_attention_weight,
    view_patchify_original_image,
)


# Since we are training Mask2Former using hugging face, we need to also use preprocessor
class CustomMask2FormerImageProcessor(Mask2FormerImageProcessor):
    def post_process_semantic_segmentation(self, outputs, target_sizes=None) -> 'torch.Tensor':
        """
        Converts the output of [`Mask2FormerForUniversalSegmentation`]
            into semantic segmentation maps. Only supports
        PyTorch.
        Args:
            outputs ([`Mask2FormerForUniversalSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`)
                    corresponds to the requested
                final size (height, width) of each prediction. If left to None,
                    predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic
                    segmentation map of shape (height, width)
                corresponding to the target_sizes entry
                    (if `target_sizes` is specified). Each entry of each
                        `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = (
            outputs.class_queries_logits
        )  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = (
            outputs.masks_queries_logits
        )  # [batch_size, num_queries, height, width]

        # Original Imagepreprocessor - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits,
            size=target_sizes[0],
            mode='bilinear',
            align_corners=False,
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum('bqc, bqhw -> bchw', masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    'Make sure that you pass in as many target sizes as the'
                    'batch dimension of the logits'
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode='bilinear',
                    align_corners=False,
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [
                semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])
            ]

        return semantic_segmentation


@dataclass
class ScoreAttentionForPatchifyConfig:
    confidence_prob_threshold: float = 0.0
    patch_keep_ratio: float = 0.0
    local_attn_score_per_patch: bool = True
    patchsize: int = 256
    visualize: bool = False


# Define a trainer class that has training and validation functions
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        metric: MeanIoU,
        scoreattention_config: ScoreAttentionForPatchifyConfig = None,
    ) -> None:
        """Constructor for the trainer

        Args:
            model (nn.Module): The model the class needs
            device (str): The device that the model will be trained or validated on
        """
        self.model = model
        self.device = device
        self.metric = metric

        self.trainer_config = scoreattention_config

        # Call the image preprocessor
        self.mask2formerpreprocessor = CustomMask2FormerImageProcessor(
            do_resize=False,
            do_normalize=False,
            do_rescale=False,
            ignore_index=255,
            reduce_labels=False,
        )

        # Create an empty LUT that will be used later to store the indexes
        self.train_patches_LUT = {}
        # The dictionary here contains the name of the file and the associated scores
        self.training_imgs_score = {}

    def preprocessing_step(self, input_data: Dict[str, any]) -> Dict[str, any]:
        # Let the incoming data run through the preprocessor
        img_preprocessor_op = self.mask2formerpreprocessor(
            images=list(input_data['image']),
            segmentation_maps=list(input_data['mask']),
            return_tensors='pt',
        )
        # Ensuring the data is in the right structure for the further steps
        for key in img_preprocessor_op:
            if isinstance(img_preprocessor_op[key], list):
                img_preprocessor_op[key] = [i.to(self.device) for i in img_preprocessor_op[key]]
            elif isinstance(img_preprocessor_op[key], torch.Tensor):
                img_preprocessor_op[key] = img_preprocessor_op[key].to(self.device)
        # Ensure that the input data is also moved to the GPU
        for key in input_data:
            if isinstance(input_data[key], torch.Tensor):
                input_data[key] = input_data[key].to(self.device)

        return img_preprocessor_op

    def attach_decoder_hooks(self, model: nn.Module) -> Union[List]:
        """Function to attach hooks and return.

        Args:
            model (nn.Module): Torch model to attach the hooks.

        Returns:
            Union[List]: List of variables for with the hooks.
        """
        # List to store masked attention weights of last layer
        decoder_masked_attn_weight = []
        """
        # List to store all the decoder attention weights
        all_decoder_hook_handler = []
        all_decoder_masked_attn_weight = []
        for layer in model.model.transformer_module.decoder.layers:
            all_decoder_hook_handler.append(layer.cross_attn.register_forward_hook(
                lambda self,
                input,
                output,
                layer=layer:all_decoder_masked_attn_weight.append(output[1])
                )
            )
        """
        # List to store self attention weights in decoder
        decoder_self_attn_weight = []
        hooks = [
            # Ideally, we only take the last layer, but now, testing with all the layers
            model.model.transformer_module.decoder.layers[-1].cross_attn.register_forward_hook(
                lambda self, input, output: decoder_masked_attn_weight.append(output[1])
            ),
            model.model.transformer_module.decoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: decoder_self_attn_weight.append(output[1])
            ),
        ]

        return hooks, decoder_masked_attn_weight

    def remove_attached_hooks(self, hooks: Union[torch.utils.hooks.RemovableHandle]) -> None:
        for hook in hooks:
            hook.remove()

    def patchify_weights(self, outputs, decoder_masked_attn_weight, images, image_paths):
        """Patchify the decoder masked-attention weight layer.

        Args:
            outputs (_type_): Outputs of the Mask2Former model.
            decoder_masked_attn_weight (_type_): Masked attention weight layer.
            images (_type_): Original image
            image_paths (str): Path to the original image.

        Raises:
            ValueError: When Confidence probability threshold is invalid.
            ValueError: When Patches keep ratio is invalid.
            ValueError: When Patch size is invalid and not divisible by 32.
        """
        # decoder_masked_attn_weight has the output size of (1, 100, 32768)
        # The 32678 can be reshaped into (128, 256) using the .view()
        # Remove the list for decoder weights, shape would now be (100, 32678)
        decoder_masked_attn_weight = decoder_masked_attn_weight[0]
        # Remove null classes
        masks_classes = outputs.class_queries_logits.softmax(dim=-1)[0, :, :-1]

        # The length of the paths and images indicate the batch size
        # Since we right now use one batch, use the first one's path
        path = image_paths[0]
        # 0 is the 1st part of the name, 2nd part is .png
        file_name = os.path.splitext(os.path.basename(path))[0]

        # Image information
        target_height = images[0].shape[1]
        target_width = images[0].shape[2]

        if (
            self.trainer_config.confidence_prob_threshold > 1.0
            or self.trainer_config.confidence_prob_threshold < 0.0
        ):
            raise ValueError(
                f'Confidence probability threshold should be within 0-1. Received:'
                f'{self.trainer_config.confidence_prob_threshold}'
            )

        if self.trainer_config.patch_keep_ratio > 1.0 or self.trainer_config.patch_keep_ratio < 0.0:
            raise ValueError(
                f'Patches keep ratio should be within 0-1. Received:'
                f'{self.trainer_config.patch_keep_ratio}'
            )

        # Keep percentage when capturing the attention maps
        keep = masks_classes.max(-1).values > self.trainer_config.confidence_prob_threshold

        # Create a patchifier object
        if (
            self.trainer_config.patchsize > target_height
            or self.trainer_config.patchsize > target_width
            or (self.trainer_config.patchsize % 16 != 0)
            or self.trainer_config.patchsize < 0
        ):
            raise ValueError(
                f'Patch size should be valid and divisible by 16. Received:'
                f'{self.trainer_config.patchsize}'
            )
        patchifier = Patchify(patchsize=self.trainer_config.patchsize)

        # Cumulative the decoder attention weights
        # Another point to test if we do not get the normalized decoder attn wt
        # Use get_cum_unnormalized_decoder_weights for that
        cum_decoder_attn_weights = get_cum_decoder_weights(
            attn_weight=decoder_masked_attn_weight, keep_indexes=keep
        )
        """
        From the above logic, we get a tensor of size (1, 32768)
        where all the queries which are confident (0.9) have been added together
        Now we simply get the score and store it in LUT
        Thefore each image will have an affiliated (64, 128) matrix as score
        """
        decoder_attn_tensor = cum_decoder_attn_weights.view(128, 256).unsqueeze(0).unsqueeze(0)
        # Reshape this back to (64, 128)
        vitmaeshapedweights = (
            torch.nn.functional.interpolate(
                decoder_attn_tensor, size=(64, 128), mode='bilinear', align_corners=False
            )
            .squeeze(0)
            .squeeze(0)
        )
        # By default when creating LUT, create it based on global score
        if not self.trainer_config.local_attn_score_per_patch:
            # What can be done here is to get all the scores for individual patches
            # So instead of assigning high score indices, we would assign the score
            # And before storing this info in a new LUT, we end up storing only the
            # top 30% of the total patches as keep indices
            patch_score, patched_weights = get_all_scores_of_patched_decoder(
                patcher=patchifier,
                decoder_attention_weight=cum_decoder_attn_weights,
                height=target_height,
                width=target_width,
            )
        else:
            # Getting indexes of the patches with high score
            high_score_indexes, patched_weights = score_attention_patches(
                patcher=patchifier,
                decoder_attention_weight=cum_decoder_attn_weights,
                height=target_height,
                width=target_width,
                keep_percentage=self.trainer_config.patch_keep_ratio,
            )

        if file_name not in self.train_patches_LUT:
            if self.trainer_config.local_attn_score_per_patch:
                self.train_patches_LUT[file_name] = high_score_indexes.cpu().numpy()
            else:
                self.train_patches_LUT[file_name] = patch_score.cpu().numpy()

            self.training_imgs_score[file_name] = vitmaeshapedweights.cpu().numpy()

        # Currently, for visualization we are not using utils.visualize_attn()
        # @NOTE:
        # There are other functions in utils.visualize_attn_map.py to refer.
        # If the view flag is on, then make do the same
        if self.trainer_config.visualize:
            view_patchify_attention_weight(
                patcher=patchifier,
                decoder_attention_weight=cum_decoder_attn_weights,
                patch_size=self.trainer_config.patchsize,
                height=target_height,
                width=target_width,
                folder_name=file_name,
            )
            view_patchify_original_image(
                patcher=patchifier,
                img_tensor=images[0],
                height=target_height,
                width=target_width,
                patch_size=self.trainer_config.patchsize,
                folder_name=file_name,
            )
            view_high_score_indexes(
                patched_attention_weight=patched_weights,
                keep_indices=high_score_indexes,
                patchifier=patchifier,
                original_img=images[0],
                patch_size=self.trainer_config.patchsize,
                folder_name=file_name,
            )

    # Create a training function
    def train(
        self,
        data_loader,
        optimizer,
        # scaler,
        lr_scheduler,
        debug=False,
    ) -> Tuple[float, float]:
        print('------------Training!!------------')
        # Declare a variable to store the running loss
        loss_meter = AverageMeterRelative()
        # Declare a variable to store the miou meters
        # miou_meter = AverageMeter()
        # Training the model
        self.model.train()
        # instantiate a progress bar with the dataloader
        progress_bar = tqdm(
            data_loader, total=len(data_loader), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        for _, input_data in enumerate(progress_bar):
            # Preprocess the input for the forward pass
            img_preprocessor_op = self.preprocessing_step(input_data)
            # Clear the gradient
            optimizer.zero_grad()
            # Autocast helps in mixed precision training
            with torch.autocast(
                device_type='cuda',  # Changed here and hard coded
                dtype=torch.float16,
                enabled=False,
            ):
                # Forward pass the input through the model
                outputs = self.model(**img_preprocessor_op)
            # Get total loss to backpropagate
            total_loss = outputs.loss
            # Scale the loss backwards
            total_loss.backward()
            # scaler.scale(total_loss).backward()

            # Unscale the scaler
            # scaler.unscale_(optimizer)
            # Since norm clipping is a necessary
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.01, norm_type=2.0)
            # After propogating loss, we need to update the optimizer and scheduler
            # scaler.step(optimizer)
            optimizer.step()
            # scaler.update()

            # Learning rate scheduler is polynomial learning rate
            lr_scheduler.step()
            # Update the loss meter
            loss_meter.update(total_loss, delta_n=input_data['image'].shape[0])

            # For now also print the learning rate

            loss_out = loss_meter.avg

            # Update the progress bar with loss and learning rate
            progress_bar.set_postfix(loss=f'{loss_out:.4f}')

            if debug:
                print(f'Valid loss: {loss_out}')
                # print(f"Miou: {miou_out}")

        return loss_out

    def trainer_setup_for_lut_creation(
        self,
        confidence_prob_threshold: float = 0.0,
        patch_keep_ratio: float = 0.0,
        local_attn_score_per_patch: bool = True,
        patchsize: int = 256,
        visualize: bool = False,
    ):
        self.trainer_config.confidence_prob_threshold = confidence_prob_threshold
        self.trainer_config.patch_keep_ratio = patch_keep_ratio
        self.trainer_config.local_attn_score_per_patch = local_attn_score_per_patch
        self.trainer_config.patchsize = patchsize
        self.trainer_config.visualize = visualize

    # Create a validate function
    def validate(self, data_loader=None, debug=False) -> Tuple[float, float]:
        print('------------Validating!!------------')
        # Declare a variable to store the running loss
        loss_meter = AverageMeterRelative()
        # Declare a variable to store the miou meters
        miou_meter = AverageMeter()
        # Evaluate the model
        self.model.eval()
        # Since in validation we do not need gradients, set flag accordingly
        with torch.no_grad():
            # Use the tqdm library which helps in the progress bar
            progress_bar = tqdm(
                data_loader, total=len(data_loader), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            )
            for _, input_data in enumerate(progress_bar):
                # Preprocess the input for the forward pass
                img_preprocessor_op = self.preprocessing_step(input_data)

                # Use lists to store outputs of hooks
                # decoder_masked_attn_weight, self_attn_weights
                # Attaching hooks only if the variables are set in the config file.
                if (
                    self.trainer_config.confidence_prob_threshold > 0.0
                    and self.trainer_config.patch_keep_ratio > 0.0
                ):
                    hooks, decoder_masked_attn_weight = self.attach_decoder_hooks(model=self.model)

                # Forward pass the input through the model
                outputs = self.model(**img_preprocessor_op)

                # Remove the hooks as the variables are already stored
                # Removing hooks only if the variables are set in the config file.
                if (
                    self.trainer_config.confidence_prob_threshold > 0.0
                    and self.trainer_config.patch_keep_ratio > 0.0
                ):
                    self.remove_attached_hooks(hooks)

                # Post process the image output for semantic segmentation
                # Get the H, W and batch size
                batch_size, _, height, width = input_data['image'].shape
                predicted_segmentation_map = (
                    self.mask2formerpreprocessor.post_process_semantic_segmentation(
                        outputs=outputs, target_sizes=[(height, width)] * batch_size
                    )
                )

                # Call for patchify and store information only if config file is set.
                if (
                    self.trainer_config.confidence_prob_threshold > 0.0
                    and self.trainer_config.patch_keep_ratio > 0.0
                ):
                    self.patchify_weights(
                        outputs=outputs,
                        decoder_masked_attn_weight=decoder_masked_attn_weight,
                        images=input_data['image'],
                        image_paths=input_data['img_path'],
                    )

                # Acquire the loss from forward pass
                loss = outputs.loss

                # Calculate the mIoU required
                miou = self.metric(
                    output=torch.stack(predicted_segmentation_map), target=input_data['mask']
                )

                # Update the loss and miou metric using the meter
                loss_meter.update(loss, delta_n=batch_size)
                miou_meter.update(miou, delta_n=batch_size)

                loss_out = loss_meter.avg
                miou_out = miou_meter.avg

                if debug:
                    print(f'Valid loss: {loss_out}')
                    print(f'Miou: {miou_out}')

            """
            In order to store the LUT, we use the below method
            Here we also need to change cityscapes.py where img_path_val and
            img_path_train have to be swapped. This is because we want to have
            LUT for the training dataset
            Ex: img_path_val = "leftImg8bit/val" --> "leftImg8bit/train" etc.
            """
            # Actual code to store PKL file
            if (
                self.trainer_config.confidence_prob_threshold > 0.0
                and self.trainer_config.patch_keep_ratio > 0.0
            ):
                if not self.trainer_config.local_attn_score_per_patch:
                    self.train_patches_LUT = global_score_index_assignment(
                        self.train_patches_LUT,
                        patch_keep_ratio=self.trainer_config.patch_keep_ratio,
                    )
                store_LUT_as_pkl(
                    self.train_patches_LUT,
                    len(data_loader),
                    patch_keep_ratio=self.trainer_config.patch_keep_ratio,
                    patch_size=self.trainer_config.patchsize,
                    folder_name='patches_lut',
                )
                # store_image_patchscoreforvitmae(
                #     self.training_imgs_score,
                #     folder_name="pscoreforvitmae"
                # )

        print('------------End of validation!!------------')
        return loss_out, miou_out
