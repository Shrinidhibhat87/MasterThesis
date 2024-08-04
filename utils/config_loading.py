import torch
import yaml


class ModelTrainingConfigLoader:
    def __init__(self, config_file_path: str) -> None:
        # Based on the config path, check if the file is valid
        try:
            with open(config_file_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f'Config path specified does not have a config file: {config_file_path}'
            ) from e
        except yaml.YAMLError as e:
            raise ValueError(f'Error loading config yaml file from {config_file_path} : {e}') from e

        # Give the option to the user to specify CUDA or CPU
        if self.config['training']['device'] == 'cuda':
            self.device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
            print(f'CUDA specified and Device used: {self.device}')
        else:
            # Even if the user does not specify a device, try to get cuda
            self.device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
            print(f'No device specified, using: {self.device}')

        # 1) Choose a setting of which training to choose from
        self.usecase_patchify_pipeline = bool(self.config['training']['usecase_patchify_training'])
        self.usecase_jpeg_pipeline = bool(self.config['training']['usescase_jpeg_training'])
        self.num_epochs = int(self.config['training']['num_epochs'])

        # 2) Choose the dataset and the path specified
        self.dataset_name = str(self.config['dataset']['name'])
        self.dataset_root_path = str(self.config['dataset']['root'])

        # 3) Tensorboard and saving the model directory
        self.save_best_model_directory = str(self.config['model_output']['output_path'])

        # 4) Create LUT variables
        self.get_lut_score_for_train_images = bool(
            self.config['create_lut']['get_score_create_lut']
        )
        self.get_attention_map_for_unseen = bool(
            self.config['create_lut']['get_attention_map_for_unseen']
        )
        self.seen_data_pkl_path = str(self.config['create_lut']['seen_data_pkl_file_path'])
        self.write_ds_LUT_to_disk = bool(self.config['create_lut']['write_ds_LUT_to_disk'])
        self.train_ds_reduction = float(self.config['create_lut']['train_ds_reduction'])

        # 5) Patchify-Compress pipeline variables
        self.usecase_patchify_confidence_prob_threshold = float(
            self.config['usecase_patchify']['confidence_prob_threshold']
        )
        self.usecase_patchify_patch_keep_ratio = float(
            self.config['usecase_patchify']['patch_keep_ratio']
        )
        self.usecase_patchify_local_attn_score_per_patch = bool(
            self.config['usecase_patchify']['local_attn_score_per_patch']
        )
        self.usecase_patchify_patchsize = int(self.config['usecase_patchify']['patchsize'])
        self.usecase_patchify_visualize = bool(self.config['usecase_patchify']['visualize'])

        # 6) Patchifier variables
        self.patchifier_patch_size = int(self.config['patchifier']['patch_size'])
        self.patchifier_lut_path = str(self.config['patchifier']['lut_path'])
        self.patchifier_imp_patches_compress_quality = int(
            self.config['patchifier']['imp_patches_compress_quality']
        )
        self.patchifier_other_compress_quality = int(
            self.config['patchifier']['other_compress_quality']
        )

        # 7) JPEG Compress variables
        self.jpeg_lut_path = str(self.config['jpeg']['lut_path'])
        self.jpeg_compress_quality = int(self.config['jpeg']['compress_quality'])

        # 1) RandomFixedScale
        """
        self.start_scale = self.config['train_transform']['RandomFixedScale']['start_scale']
        self.end_scale = self.config['train_transform']['RandomFixedScale']['end_scale']
        self.step_size = self.config['train_transform']['RandomFixedScale']['step_size']

        # 2) RandomSegmentationCrop
        self.height = self.config['train_transform']['RandomSegmentationCrop']['height']
        self.width = self.config['train_transform']['RandomSegmentationCrop']['width']

        """

        # Configuration files for training transformations.
        # Use of nesting here
        # Hard coding these parameters here
        """
        The below logic has to change because hard coding like this would not make sense
        Think of a different logic when time permits.
        """


class BandwidthCalculatorConfigLoader:
    def __init__(self, config_file_path: str) -> None:
        # Based on the config path, check if the file is valid
        try:
            with open(config_file_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f'Config path specified does not have a config file: {config_file_path}'
            ) from e
        except yaml.YAMLError as e:
            raise ValueError(f'Error loading config yaml file from {config_file_path} : {e}') from e

        # 1) Dataset path
        self.dataset_name = str(self.config['dataset']['name'])
        self.dataset_root_path = str(self.config['dataset']['root'])
        self.dataset_full_train_path = str(self.config['dataset']['full_train_path'])

        # 2) Path where to store output
        self.output_path = str(self.config['output']['path'])

        # 3) Usecase-Patchify Compress bandwidth calculation
        self.patchify_bandwidth_calc = bool(self.config['usecase_patchifier']['calculate_bw'])
        self.patchify_patchsize = int(self.config['usecase_patchifier']['patch_size'])
        self.highscoring_compress_quality = int(
            self.config['usecase_patchifier']['imp_patch_quality']
        )
        self.others_compress_quality = int(self.config['usecase_patchifier']['other_patch_quality'])

        # 4) Usecase JPEG compress bandwidth calculation
        self.usecase_jpeg_compress = bool(self.config['usecase_jpeg']['calculate_bw'])
        self.jpeg_compress_quality = int(self.config['usecase_jpeg']['jpeg_quality'])

        # 5) Calculate the size of uncompressed image
        self.calculate_uncompressed_image_bw = bool(
            self.config['uncompressed_image']['calculate_bw']
        )

        # 6) The LUT bsaed on which the BW is calculated
        self.lut_path = str(self.config['lut_path'])

        # 7) MSSSIM, PSNR and BPP options
        # self.calculate_bpp = bool(self.config['bpp'])
        self.calculate_msssim = bool(self.config['msssim'])
        self.psnr = bool(self.config['psnr'])
