import argparse
import io
import math
import os
import pickle
import time
from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision.io import decode_jpeg, encode_jpeg

from dataloader.CityScapeDataloader import LoadAndGetDataloader
from patchify.patchify import Patchify
from utils.config_loading import BandwidthCalculatorConfigLoader


class BandwidthCalculator:
    """
    Class to calculate the size of images in different format.
    Possible to calculate with only jpegcompressdecompress or
    patchify, encode and decode based on parts.

    PSNR:
    Calculate PSNR using cv2 library.
    PSNR is calculated in decibels and is done between the images.
    https://www.educative.io/answers/what-is-peak-signal-to-noise-ratio-in-image-processing

    MM-SMIM:
    # data_info = self.load_data_from_file(data_info)
    https://lightning.ai/docs/torchmetrics/stable/image
    /multi_scale_structural_similarity.html
    """

    def __init__(
        self,
        full_train_path: str,
        dataset_folder_path: str,
        output_folder_path: str,
        patch_size: int,
        lut: str,
        highscoring_compress_quality: int,
        others_compress_quality: int,
        jpeg_compress_quality: int,
        config,
    ) -> None:
        self.dataset_folder_path = dataset_folder_path
        self.full_train_path = full_train_path
        self.output_folder_path = output_folder_path
        self.images_as_tensors = []

        self.config = config

        self.patchifier = Patchify(patchsize=patch_size)
        self.lut_path = lut
        self.highscoring_compress_quality = highscoring_compress_quality
        self.other_compress_quality = others_compress_quality

        self.transform = transforms.Compose([transforms.PILToTensor()])

        dataloader = LoadAndGetDataloader(
            root_dir=self.dataset_folder_path, batch_size=1, num_workers=8, shuffle=False
        )

        self.train_dataloader, _ = dataloader.create_dataset_and_load(
            train_transform=self.transform, valid_transform=self.transform
        )

        self.check_input()
        self.lut = {}
        with open(self.lut_path, 'rb') as file:
            self.lut = pickle.load(file)

        self.all_info = []
        self.total_patched_size = 0.0
        self.total_orig_size = 0.0
        self.pure_jpeg_size = 0.0
        self.encoded_patched_size = 0.0
        self.encoded_pure_jpeg = 0.0
        self.jpeg_compress_quality = jpeg_compress_quality

    def check_input(self) -> None:
        """Check if the inputs passed is valid.

        Raises:
            ValueError: Raised when the path of LUT is invalid.
            ValueError: Raised when file format of LUT is invalid.
            ValueError: Raised when compress quality is incorrect.
            ValueError: Raised when compress quality is incorrect.
        """
        # Check if the path given is valid
        if not os.path.exists(self.lut_path):
            raise ValueError('Invalid path for look-up-table')
        # Check if the file extension is .pkl file
        if not self.lut_path.endswith('.pkl'):
            raise ValueError('Received file should be a valid .pkl file')
        # Check for compress and decompress quality for the high scoring patches
        if self.highscoring_compress_quality > 100 or self.highscoring_compress_quality < 1:
            raise ValueError(
                f'Compress high quality should be between 1-100'
                f'Received: {self.highscoring_compress_quality}'
            )
        # Check for compress and decompress for the other patches
        if self.other_compress_quality > 100 or self.other_compress_quality < 1:
            raise ValueError(
                f'Compress low quality should be between 1-100'
                f'Received: {self.other_compress_quality}'
            )

    def calculate_bpp(self, encoded_jpeg_data) -> float:
        """
        Calculate the bits per pixel (bpp) for an encoded jpeg image data.
        This happens per patch and not for full reconstructed image.

        Args:
            encoded_jpeg_data (bytes): JPEG encoded image data

        Returns:
            float: The bpp of the encoded jpeg image
        """
        # Convert tensor to bytes
        jpeg_bytes = encoded_jpeg_data.detach().cpu().numpy().tobytes()

        # Create a BytesIO object from JPEG bytes
        jpeg_bytes_io = io.BytesIO(jpeg_bytes)

        # Open the JPEG image from the BytesIO object
        image = Image.open(jpeg_bytes_io)
        width, height = image.size
        file_size = len(jpeg_bytes)

        bpp = (file_size * 8) / (width * height)

        return bpp

    def compute_msssim(self, tensor_a, tensor_b):
        return ms_ssim(tensor_a, tensor_b, data_range=1.0).item()

    def compute_psnr(self, tensor_a, tensor_b):
        mse = torch.mean((tensor_a - tensor_b) ** 2).item()
        # The numerator is the maximum value (uint8) squared
        # And therefore 255
        return 10 * math.log10(torch.max(tensor_a) ** 2 / mse)

    """
    def load_images(self) -> None:
        # Loading images based on the folder path passed
        s_time = time.time()
        # Iterate over the folder path
        for root, dirs, files in os.walk(self.full_train_path):
            # Iterate over each file
            for file in files:
                if file.endswith("png") or file.endswith("jpg"):
                    # Get the path of the image
                    image_path = os.path.join(root, file)

                    # Open the image using PIL
                    image = Image.open(image_path)

                    # Convert the image to a torch tensor object
                    img_tensor = self.transform_pil_to_tt(image)

                    # Do not add images into the list if it is not present in the lut
                    # This way, if the LUT was created for 70% unseen, only the sizes
                    # there will be calculated
                    img_name = os.path.splitext(os.path.basename(image_path))[0]

                    if img_name not in self.lut:
                        continue

                    # Append this information to the list
                    self.images_as_tensors.append((img_name, img_tensor))
        e_time = time.time()
        print(f"Time taken to load images: {e_time-s_time:.2f} seconds")
    """

    def calculate_bytes_kb(self, image: torch.tensor) -> float:
        """Function to calculate the bytes if this image.

        Args:
            image (torch.tensor): Torch tensor image whose image we should
                calculate.

        Returns:
            float: Size of the image.
        """
        # Convert the image tensor to PIL Image object
        if image.ndim == 4:
            image = image.squeeze(0)
        pil_image = transforms.ToPILImage()(image)

        # Create a BytesIO object
        stream = io.BytesIO()

        # Save the image in the stream object as PNG
        pil_image.save(stream, format='PNG')

        # Get the file size of the stream/BytesIO object
        file_size = stream.getbuffer().nbytes

        # Close the BytesIO object
        stream.close()

        size_in_kb = file_size / 1024

        return size_in_kb

    def patch_comp_decomp(self, image_name: str, image: torch.tensor) -> torch.tensor:
        """Function to patchify and compress and decompress the image.

        Args:
            image_name (str): Name of the image.
            image (torch.tensor): Image in the torch tensor format.

        Returns:
            torch.tensor: Patchified image.
        """
        # Need to first create a batch dimension to pass via Patchifier
        if image.ndim == 3:
            img_tensor = image.unsqueeze(0)
        else:
            img_tensor = image
        # Create patches. Need to convert to float32 to avoid an issue
        # The output of this would be (N_p, C, H_p, W_p)
        patched_parts = self.patchifier(img_tensor.to(torch.float32)).squeeze()
        # Convert to uint8
        patched_parts = patched_parts.to(torch.uint8)
        image_patched = []
        total_patchbased_size = 0
        # To calculate the final bpp for the entire encoded image
        bpp_total = 0
        # Check if the file_name, indicating the key is present in the LUT
        if image_name in self.lut:
            # Get the high scoring index from the lut
            high_scoring_indexes = self.lut[image_name]
            # Simply replace the indices from the compressed_parts with orig
            image_patched = []
            for index in range(patched_parts.shape[0]):
                if index in high_scoring_indexes:
                    encoded_data = encode_jpeg(
                        patched_parts[index], quality=self.highscoring_compress_quality
                    )
                    image_patched.append(decode_jpeg(encoded_data))
                else:
                    encoded_data = encode_jpeg(
                        patched_parts[index], quality=self.other_compress_quality
                    )
                    image_patched.append(decode_jpeg(encoded_data))

                # Back in the for loop
                # Simply adding the length of the encoded vector as each entry is a uin8 value
                total_patchbased_size += len(encoded_data)
                bpp_total += self.calculate_bpp(encoded_data)

        # Stack the patches together
        patched_image = torch.stack(image_patched, dim=0)
        # Construct the patched parts to a full image
        reshaped_color_patches = patched_image.view(4, 8, 3, 256, 256)
        reshaped_color_patches = reshaped_color_patches.permute(0, 3, 1, 4, 2)
        original_img_post_patchify = (
            reshaped_color_patches.contiguous().view(1024, 2048, 3).permute(2, 0, 1)
        )

        # Convert this back to uint8
        # Note: When training the model, this should be converted to float32
        original_img_post_patchify = original_img_post_patchify.to(torch.uint8)

        return original_img_post_patchify, total_patchbased_size, bpp_total

    def write_info_in_file(self) -> None:
        """Function to write the information in the file."""
        s_time = time.time()
        if self.all_info:
            # Create directory if it does not exist
            if not os.path.exists(self.output_folder_path):
                os.makedirs(self.output_folder_path)
            # Create the file path
            # file_path = os.path.join(
            #     self.output_folder_path,
            #     f"Local_Usecase_70_unseen_patchkeep0.1_SizeComparison_"
            #     f"{self.highscoring_compress_quality}_{self.other_compress_quality}_"
            #     f"_JPEG_{self.jpeg_compress_quality}.txt",
            # )
            file_path = os.path.join(
                self.output_folder_path,
                'Reduced_training_kpall.txt',
            )
            # Open the file in write mode
            with open(file_path, 'w') as file:
                # Iterate over the elements and write the information to the file
                for all_info in self.all_info:
                    (
                        img_name,
                        size_patch,
                        size_jpeg,
                        size_orig,
                        patch_encoded,
                        pure_encoded,
                        psnr_patched,
                        psnr_jpeg,
                        msssim_patched,
                        msssim_jpeg,
                    ) = all_info
                    file.write(
                        f'{img_name}, '
                        f'PSNR Patched [db]: {psnr_patched},'
                        f'PSNR JPEG [db]: {psnr_jpeg}, '
                        f'MS-SSIM Patched: {msssim_patched}, '
                        f'MS-SSIM JPEG: {msssim_jpeg} \n'
                        # f"BPP patched: {bpp_patched}, "
                        # f"BPP JPEG: {bpp_jpeg}, \n"
                    )
                    file.write(f'Original size: {size_orig} KB\n')
                    file.write(f'JPEG Compress Decompress: {size_jpeg}\n')
                    file.write(f'Patchify JPEGCompDecomp: {size_patch} KB\n')
                    file.write(f'Patchify JPEG encoded: {patch_encoded} KB\n')
                    file.write(f'Full JPEG encoded: {pure_encoded} KB\n\n')

                file.write(f'Total orig size {self.total_orig_size / 1024} MB\n')
                file.write(f'Total patched size: {self.total_patched_size / 1024} MB\n')
                file.write(f'Total jpeg size: {self.pure_jpeg_size / 1024} MB\n')
                file.write(f'Total patched encoded size: {self.encoded_patched_size / 1024} MB\n')
                file.write(f'Total pure jpeg encoded size: {self.encoded_pure_jpeg / 1024} MB\n')

            e_time = time.time()
            print(f'Total time to write to file: {e_time-s_time:.2f} seconds')

    def calcuate_usecase_bandwidth_size(self) -> None:
        """Function to do the operations."""
        # Check the time to see how long this took
        start_time = time.time()
        size_patched_jpeg_compdecomp = 0
        pure_encoded_jpeg_kb = 0
        size_pure_jpeg = 0
        size_orig_img = 0
        patched_encoded_kb = 0
        psnr_patched = 0
        msssim_jpeg = 0
        psnr_jpeg = 0
        msssim_patched = 0
        # Create an iterator to iterate over each object
        for _, feed_dict in enumerate(self.train_dataloader):
            # Get image name
            full_image_path = feed_dict['img_path']
            image_name = os.path.splitext(os.path.basename(full_image_path[0]))[0]
            # s_time = time.time()

            # if image_name not in self.lut:
            #     continue

            image = feed_dict['image']

            # If user has asked to calculate the bandwidth of the patchify approach
            if self.config.patchify_bandwidth_calc:
                patched_comp_img, patched_encoded_size, patched_bpp = self.patch_comp_decomp(
                    image_name=image_name, image=image
                )
                # Get the size of the patched, comp and decomp img
                size_patched_jpeg_compdecomp = self.calculate_bytes_kb(image=patched_comp_img)
                patched_encoded_kb = patched_encoded_size / 1024  # Convert the bytes to kilobytes

                # If user has asked to calculate MSSSIM
                if self.config.calculate_msssim:
                    # Calculate the msssim for pure jpeg compression
                    msssim_patched = self.compute_msssim(
                        image.float(),  # Actual image
                        patched_comp_img.unsqueeze(0).float(),
                    )

                # If user has asked to calculate PSNR
                if self.config.psnr:
                    # Calculate the PSNR metrics between the actual image and pure jpeg
                    psnr_patched = self.compute_psnr(
                        image.float(),  # Actual image
                        patched_comp_img.unsqueeze(0).float(),
                    )

            # If the user has asked to calculate the jpeg compress approach
            if self.config.usecase_jpeg_compress:
                # pure_jpeg_comp_img = jpeg(image=image, quality=self.jpeg_compress_quality)
                pure_jpeg_encoded = encode_jpeg(image[0], self.jpeg_compress_quality)
                pure_jpeg_comp_img = decode_jpeg(pure_jpeg_encoded)

                pure_encoded_jpeg_kb = len(pure_jpeg_encoded) / 1024

                # Get the size of pure jpeg compress decompress image
                size_pure_jpeg = self.calculate_bytes_kb(image=pure_jpeg_comp_img)

                # If user has asked to calculate MSSSIM
                if self.config.calculate_msssim:
                    # Calculate the msssim for pure jpeg compression
                    msssim_jpeg = self.compute_msssim(
                        image.float(),  # Actual image
                        pure_jpeg_comp_img.unsqueeze(0).float(),
                    )

                # If user has asked to calculate PSNR
                if self.config.psnr:
                    # Calculate the PSNR metrics between the actual image and pure jpeg
                    psnr_jpeg = self.compute_psnr(
                        image.float(),  # Actual image
                        pure_jpeg_comp_img.unsqueeze(0).float(),
                    )

            # If the user has asked to calculate the bw of uncompressed image
            if self.config.calculate_uncompressed_image_bw:
                # Get the size of the original image
                size_orig_img = self.calculate_bytes_kb(image=image)

            # Store all the information in a list
            self.all_info.append(
                (
                    image_name,
                    size_patched_jpeg_compdecomp,
                    size_pure_jpeg,
                    size_orig_img,
                    patched_encoded_kb,
                    pure_encoded_jpeg_kb,
                    psnr_patched,
                    psnr_jpeg,
                    msssim_patched,
                    msssim_jpeg,
                    # patched_bpp,
                    # self.calculate_bpp(pure_jpeg_encoded)
                )
            )
            # Store the total original image size and patched size
            self.total_patched_size += size_patched_jpeg_compdecomp
            self.total_orig_size += size_orig_img
            self.pure_jpeg_size += size_pure_jpeg
            self.encoded_patched_size += patched_encoded_kb
            self.encoded_pure_jpeg += pure_encoded_jpeg_kb
            # e_time = time.time()
            # print(f"Time taken for {index} image is {e_time-s_time:.2f} secs")

        end_time = time.time()

        print(f'\nTotal time for the image operations:' f'{end_time-start_time:.2f} seconds')

    def calculate_bandwidth_pure_images(self) -> None:
        start_time = time.time()

        # Create an iterator to iterate over each object
        for _, feed_dict in enumerate(self.train_dataloader):
            # Get image name
            full_image_path = feed_dict['img_path']
            image_name = os.path.splitext(os.path.basename(full_image_path[0]))[0]
            if image_name not in self.lut:
                continue
            image = feed_dict['image']

            # Get the size of the original image
            size_orig_img = self.calculate_bytes_kb(image=image)
            # Get the size of pure jpeg compress decompress image
            # size_pure_jpeg = self.calculate_bytes_kb(image=pure_jpeg_comp_img)
            # patched_encoded_kb = (patched_encoded_size/1024) # Convert the bytes to kilobytes
            # pure_encoded_jpeg_kb = (len(pure_jpeg_encoded)/1024)
            # Store all the information in a list
            self.all_info.append(
                (
                    image_name,
                    0,
                    0,
                    size_orig_img,
                    0,
                    0,  # pure_encoded_jpeg_kb,
                    0,  # psnr_patched,
                    0,  # psnr_jpeg,
                    0,  # msssim_patched,
                    0,  # msssim_jpeg,
                    # patched_bpp,
                    # self.calculate_bpp(pure_jpeg_encoded)
                )
            )
            # Store the total original image size and patched size
            self.total_patched_size += 0
            self.total_orig_size += size_orig_img
            self.pure_jpeg_size += 0
            self.encoded_patched_size += 0
            self.encoded_pure_jpeg += 0
            # e_time = time.time()
            # print(f"Time taken for {index} image is {e_time-s_time:.2f} secs")

        end_time = time.time()

        print(f'\nTotal time for the image operations:' f'{end_time-start_time:.2f} seconds')

    def go_do_your_magic(self) -> None:
        self.calcuate_usecase_bandwidth_size()
        # self.calculate_bandwidth_pure_images()
        self.write_info_in_file()


def main():
    # Main function to capture the image sizes
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config_path',
        type=str,
        default='myworkspace/configs/bw_calculation_configs/usecase_bandwidth_calculation.yaml',
    )
    """
    default_output_dir = Path.cwd()
    parser.add_argument(
        "--full_train_path",
        type=str,
        default="/data/datasets/PytorchDatasets/CityScapes-pytorch/leftImg8bit/train/",
        help="Path to image folder",
    )
    parser.add_argument(
        "--dataset_folder_path",
        type=str,
        default="/data/datasets/PytorchDatasets/CityScapes-pytorch/",
        help="Path to image folder",
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default=default_output_dir,
        help="Output path to where you want to store the size information",
    )
    parser.add_argument(
        "--patch_size", type=int, default=256, help="Patch size to define"
    )
    parser.add_argument(
        "--lut_path",
        type=str,
        default="patches_lut_patch_lut/reduced_train_ds/actual_training_lb4_gb8_reduced_data_kp_0.5_30_.pkl",
        help="LUT .pkl path",
    )
    parser.add_argument(
        "--highscoring_compress_quality",
        type=int,
        default=15,
        help="Compress quality to verify",
    )
    parser.add_argument(
        "--others_compress_quality",
        type=int,
        default=15,
        help="Compress quality to verify",
    )
    # jpeg_compress_quality
    parser.add_argument(
        "--jpeg_compress_quality",
        type=int,
        default=10,
        help="JPEG compress and decompress",
    )
    """

    args = parser.parse_args()

    # Load the config file
    config = BandwidthCalculatorConfigLoader(config_file_path=args.config_path)

    img_calc = BandwidthCalculator(
        full_train_path=config.dataset_full_train_path,
        dataset_folder_path=config.dataset_root_path,
        output_folder_path=Path.cwd(),  # config.output_path,
        patch_size=config.patchify_patchsize,
        lut=config.lut_path,
        highscoring_compress_quality=config.highscoring_compress_quality,
        others_compress_quality=config.others_compress_quality,
        jpeg_compress_quality=config.jpeg_compress_quality,
        config=config,
    )

    img_calc.go_do_your_magic()


if __name__ == '__main__':
    main()
