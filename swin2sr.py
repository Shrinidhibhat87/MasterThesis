import os
import time

import numpy as np
import torch
import torchpack.distributed as dist
import torchvision
import torchvision.transforms.v2 as transforms
from PIL import Image
from torchvision.io import decode_jpeg, encode_jpeg
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

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


def main():
    # Read the original image
    img = Image.open('Aachen_125_OriginalImage.png').convert('RGB')

    # if torch.backends.mps.is_available():
    #     device = 'mps'
    # elif torch.cuda.is_available():
    #     device = 'cuda'
    #     print(f'Current device: {torch.cuda.current_device()}')
    #     setup_cuda_env()
    # else:
    #     device = 'cpu'
    #     logger.warning('No accelerator found, proceeding with CPU!')

    # Pass the original image through jpegencodedecode and save this.
    # Will have to convert this to a torch tensor
    transform_pil_to_tt = transforms.Compose([transforms.PILToTensor()])
    img_tensor = transform_pil_to_tt(img)
    img_jpeg = decode_jpeg(encode_jpeg(img_tensor, quality=25))
    torchvision.utils.save_image(
        torchvision.utils.make_grid(img_jpeg.float(), normalize=True, scale_each=True),
        'Aachen_125_JPEG25.png',
    )

    # Get the preprocessor
    img_processor = Swin2SRImageProcessor()

    # Define the model
    model = Swin2SRForImageSuperResolution.from_pretrained('caidas/swin2SR-classical-sr-x2-64')

    # Get the pixel values.
    pixel_values = img_processor(img_jpeg, return_tensors='pt').pixel_values
    print(f'Shape of the preprocessed image is: {pixel_values.shape}')
    s_time = time.time()
    with torch.no_grad():
        outputs = model(pixel_values)
    e_time = time.time()

    print(f'Time taken for the model to process the input: {e_time-s_time}')
    # The output shape after upscale and reconstruction
    print(f'The shape of the reconstructed output: {outputs.reconstruction.shape}')

    # Process the output to ensure we can also see the output
    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output_img = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    image = Image.fromarray(output_img)
    reconst_name = 'Aachen_125_reconstructed_img_25.png'
    image.save(reconst_name)


if __name__ == '__main__':
    main()
