import argparse
import os
import sys

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from torch import nn
from torch.optim.lr_scheduler import PolynomialLR

from dataloader.CityScapeDataloader import LoadAndGetDataloader
from dataloader.custom_transforms.jpeg_compress_decompress import JPEGCompressDecompress
from dataloader.custom_transforms.patchifier import Patchifier
from dataloader.custom_transforms.random_fixed_resize import RandomFixedResize
from dataloader.custom_transforms.segmentation_crop import RandomSegmentationCrop
from dataloader.custom_transforms.todtype import ToDtype
from metric.mean_iou import MeanIoU
from model.huggingfacemodel import Mask2FormerHuggingFace
from trainer.trainer import Trainer
from utils.adaptive_weight_decay import map_weight_decay_to_params
from utils.config_loading import ModelTrainingConfigLoader
from utils.save_model_checkpoint import SaveBestModel
from utils.tensorboard_tracker import TensorboardTracker


def set_device():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # First check if there is a GPU or not. And if so, make that the default device
    # DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return DEVICE


def build_normal_training_pipeline() -> transforms.Compose:
    """
    Building the normal training transform pipeline.

    Returns:
        transforms.Compose: Transform pipeline applied to the training DS
    """
    train_transform = transforms.Compose(
        [
            RandomFixedResize(scale_range=np.arange(0.5, 2.1, 0.1).tolist(), antialias=True),
            RandomSegmentationCrop(
                size=[512, 1024],
                pad_if_needed=False,
                padding=None,
                max_attempts=100,
                cat_max_ratio=0.75,
            ).transform(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPhotometricDistort(),  # using default values here
            ToDtype(dtype_image='float32', dtype_mask='long'),
            transforms.Normalize(
                inplace=False, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)
            ),
        ]
    )

    return train_transform


def build_usecase_patchify_compress_pipeline(
    lut_path: str = 'None',
    patch_size: int = 256,
    imp_patch_quality: int = 15,
    other_quality: int = 5,
) -> transforms.Compose:
    """Build transform pipeline for patchify compress pipeline.

    Args:
        lut_path (str, optional): Path to LUT. Defaults to "None".
        patch_size (int, optional): Patch size. Defaults to 256.
        imp_patch_quality (int, optional): Image patch quality. Defaults to 15.
        other_quality (int, optional): Other patch quality. Defaults to 5.

    Returns:
        transforms.Compose: Transform pipeline applied to the training DS
    """
    train_transform = transforms.Compose(
        [
            Patchifier(
                lut=lut_path,
                patch=patch_size,
                imp_patches_compress_quality=imp_patch_quality,
                other_compress_quality=other_quality,
            ).transform(),
            RandomFixedResize(scale_range=np.arange(0.5, 2.1, 0.1).tolist(), antialias=True),
            RandomSegmentationCrop(
                size=[512, 1024],
                pad_if_needed=False,
                padding=None,
                max_attempts=100,
                cat_max_ratio=0.75,
            ).transform(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPhotometricDistort(),  # using default values here
            ToDtype(dtype_image='float32', dtype_mask='long'),
            transforms.Normalize(
                inplace=False, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)
            ),
        ]
    )

    return train_transform


def build_usecase_jpeg_compress_pipeline(
    lut_path: str = 'None',
    jpeg_quality: int = 15,
) -> transforms.Compose:
    """Build transform pipeline for jpeg compress pipeline.

    Args:
        lut_path (str, optional): Path to LUT. Defaults to "None".
        jpeg_quality (int, optional): Image patch quality. Defaults to 15.

    Returns:
        transforms.Compose: Transform pipeline applied to the training DS
    """
    train_transform = transforms.Compose(
        [
            JPEGCompressDecompress(quality=jpeg_quality, lut=lut_path).jpeg_transform(),
            RandomFixedResize(scale_range=np.arange(0.5, 2.1, 0.1).tolist(), antialias=True),
            RandomSegmentationCrop(
                size=[512, 1024],
                pad_if_needed=False,
                padding=None,
                max_attempts=100,
                cat_max_ratio=0.75,
            ).transform(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPhotometricDistort(),  # using default values here
            ToDtype(dtype_image='float32', dtype_mask='long'),
            transforms.Normalize(
                inplace=False, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)
            ),
        ]
    )

    return train_transform


def build_val_transform_pipeline():
    # Create a transform compose object for the validation dataset
    valid_transform = transforms.Compose(
        [
            ToDtype(dtype_image='float32', dtype_mask='long'),
            transforms.Normalize(
                inplace=False, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)
            ),
        ]
    )

    return valid_transform


def select_model(
    device: str,
    use_hf_pre_trained_model: bool = False,
    use_local_pretrained_model: bool = False,
):
    # First, we will simply run inference on a fully pre-trained Mask2Former model
    # For that we did use_hf_pre_trained_model=True
    mask2formermodel = Mask2FormerHuggingFace(
        use_hf_pre_trained_model=use_hf_pre_trained_model,
        use_local_pretrained_model=use_local_pretrained_model,
    )
    model = mask2formermodel.create_model()

    # Move the model to the device
    model.to(device=device)

    return model


def setup_optimizer(model: nn.Module, initial_lr: float, weight_decay: float):
    # Gradient scalers are responsible for taking care of mixed-precision scaling
    # of gradients. This would mean if there are gradients with lesser precision
    # for ex: float16, the understanding of how these would be handled is done by
    # Gradscaler.
    # Since this is only supported in GPU, we check for thisa and if it is fp16
    # fp16=False
    # scaler = torch.cuda.amp.GradScaler()

    netparams = map_weight_decay_to_params(
        model=model, initial_lr=initial_lr, weight_decay=weight_decay
    )
    optimizer = torch.optim.AdamW(params=netparams, lr=initial_lr)

    return optimizer


def score_and_create_lut_for_patchify_pipeline(
    device: str,
    root_dir: str,
    metric: MeanIoU,
    train_transform: transforms.Compose,
    valid_transform: transforms.Compose,
    dataloader: LoadAndGetDataloader,
    get_attention_map_for_unseen: bool = False,
    get_attention_map_for_train_ds: bool = False,
    seen_data_pkl_file_path: str = 'None',
    write_ds_LUT_to_disk: bool = False,
    train_ds_reduction: float = -1.0,
):
    # Select the model
    # The path of which pre-trained model can be made configurable
    model = select_model(device=device, use_local_pretrained_model=True)

    # Instantiate a trainer object
    trainer_object = Trainer(model=model, device=device, metric=metric)
    # Best design principle is to have this as well configured
    # Therefore a yaml file has these settings as input
    trainer_object.trainer_setup_for_lut_creation(
        confidence_prob_threshold=0.9,
        patch_keep_ratio=0.1,
        local_attn_score_per_patch=True,
        patchsize=256,
        visualize=False,
    )

    # This would mean, there is no need for training of a model
    # One needs to simply use a pretrained model, which will
    # extract attention scores and store info in LUT

    training_dataloader, valid_dataloader = dataloader.create_dataset_and_load(
        train_transform=train_transform,
        valid_transform=valid_transform,
        get_attention_map_for_training_ds=get_attention_map_for_unseen,
        get_attention_map_for_train_ds=get_attention_map_for_train_ds,
        train_ds_reduction=train_ds_reduction,
        seen_data_pkl_file_path=seen_data_pkl_file_path,
        write_ds_LUT_to_disk=write_ds_LUT_to_disk,
    )

    # Call the validate function of the trainer for the training dataset
    val_loss, val_miou = trainer_object.validate(data_loader=valid_dataloader, debug=False)

    return val_loss, val_miou


def main():
    # For now, there would be three options available
    # 1) Normal training
    # 2) Patchify-compress-pipeline (usecase)
    # 3) JPEG compress pipeline (usecase)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config_path',
        type=str,
        default='myworkspace/configs/model_training_configs/normal_training_config.yaml',
    )

    args = parser.parse_args()

    # Load the config file based on the path
    config = ModelTrainingConfigLoader(config_file_path=args.config_path)

    # Root directory:
    ROOT_DIRECTORY = config.dataset_root_path  # args.dataset_root_directory
    # Path to save the model and track experiment:
    SAVE_MODEL_PATH = config.save_best_model_directory  # args.save_model_path_directory

    # Number of epochs
    NUM_EPOCS = config.num_epochs  # args.num_epoch

    # Set the required device
    DEVICE = set_device()

    # Build the train transform pipeline
    if config.usecase_patchify_pipeline:
        train_transform = build_usecase_patchify_compress_pipeline(
            lut_path=config.patchifier_lut_path,
            patch_size=config.patchifier_patch_size,
            imp_patch_quality=config.patchifier_imp_patches_compress_quality,
            other_quality=config.patchifier_other_compress_quality,
        )
    elif config.usecase_jpeg_pipeline:
        train_transform = build_usecase_jpeg_compress_pipeline(
            lut_path=config.jpeg_lut_path, jpeg_quality=config.jpeg_compress_quality
        )
    else:
        train_transform = build_normal_training_pipeline()

    # Build the validation transform pipeline
    valid_transform = build_val_transform_pipeline()

    #### Define and declare the mIoU metrics using evaluate library
    # iou_metric = evaluate.load('mean_iou')
    iou_metric = MeanIoU(num_classes=19, ignore_index=255)

    # Get validation dataloader (validation_dataloader)
    dataloader = LoadAndGetDataloader(
        root_dir=ROOT_DIRECTORY, batch_size=4, num_workers=4, shuffle=True
    )

    # Check if the user has asked to score training images
    if config.get_lut_score_for_train_images:  # args.get_lut_score_for_train_images:
        val_loss, val_miou = score_and_create_lut_for_patchify_pipeline(
            device=DEVICE,
            root_dir=ROOT_DIRECTORY,
            metric=iou_metric,
            valid_transform=valid_transform,
        )

        print(f'The final val_loss: {val_loss} and mIoU: {val_miou}')

        print('Exiting from the lut creation cycle')

        sys.exit()

    # Select the model
    model = select_model(device=DEVICE)

    # Instantiate a trainer object
    trainer_object = Trainer(model=model, device=DEVICE, metric=iou_metric)

    # Print out the number of parameters and trainable parameters
    total_parameters = sum(param.numel() for param in model.parameters())
    print(f'Total number of parameters: {total_parameters}')
    # Print out total number of trainable parameters
    total_trainable_param = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    print(f'Total number of parameters: {total_trainable_param}')

    ### Create empty metrics and loss
    train_loss = []
    valid_loss, valid_miou = [], []

    # Create a directory if the directory does not exist
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    # Create an instance of the writer object
    tensorboard_tracker = TensorboardTracker(log_path=SAVE_MODEL_PATH)

    print(
        f'Tensorboard experiment is being tracked at: http://localhost:6006/'
        f' or tensorboard --logdir={SAVE_MODEL_PATH}'
    )

    # Get validation dataloader (validation_dataloader)
    dataloader = LoadAndGetDataloader(
        root_dir=ROOT_DIRECTORY, batch_size=4, num_workers=4, shuffle=True
    )

    training_dataloader, validation_dataloader = dataloader.create_dataset_and_load(
        train_transform=train_transform, valid_transform=valid_transform
    )

    #### Declare and define the optimizer
    INITIAL_LR = 1e-4
    WEIGHT_DECAY = 0.05

    optimizer = setup_optimizer(model=model, initial_lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)

    #### Declare and define a learning rate scheduler which should be adaptable
    # Instead of MultiStepLR, we need to use Polynomial LR
    lr_scheduler = PolynomialLR(
        optimizer=optimizer,
        power=0.9,
        total_iters=(120 * len(training_dataloader)),  # Total number of steps per epoch
    )

    # Empty the GPU cache before training starts
    torch.cuda.empty_cache()

    # Instantiate the save model object
    best_model_saver = SaveBestModel()

    for epoch in range(NUM_EPOCS):
        print(f'Epoch {epoch+1} / {NUM_EPOCS}')

        # Train the model
        epoch_loss = trainer_object.train(
            data_loader=training_dataloader,  # training_dataloader.get_dataloader(),
            optimizer=optimizer,
            # scaler=scaler,
            lr_scheduler=lr_scheduler,
        )

        train_loss.append(epoch_loss)

        # Call the validate function of the trainer
        val_loss, val_miou = trainer_object.validate(
            data_loader=validation_dataloader,  # .get_dataloader(),
            debug=False,
        )

        valid_loss.append(val_loss)
        valid_miou.append(val_miou)

        # Save the best model
        best_model_saver(
            current_valid_loss=val_loss,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            path_to_save=SAVE_MODEL_PATH,
        )

        # Pass arguments to track
        tensorboard_tracker.track(
            epoch=epoch,
            train_loss=epoch_loss,
            val_loss=val_loss,
            val_miou=val_miou,
            model=model,
            learning_rate=optimizer.param_groups[0]['lr'],
        )

        print(f'Valid Epoch loss: {val_loss: .4f},')
        print(f'Valid Epoch mIoU: {val_miou: .4f},')


if __name__ == '__main__':
    main()
