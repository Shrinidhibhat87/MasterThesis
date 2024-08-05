# Master Thesis Topic: Bandwidth efficient learning on Vision Transformer for Semantic Segmentation

## NOTE:

The project was carried out in BMW Autonomous campus, Unterschleißheim from 1st March 2024 until July 31st.
WZL, a research institute at RWTH Aachen supervised my work and required that the code is made available
to demonstrate the work carried out for assessment of the thesis.
Please refrain from copying, duplicating or downloading the code without permission.

## What is the problem?

Data being uploaded to the server from the edge devices requires heavy bandwidth. A lot of data in this regard,
are actually redundant and dont help the model in training. As this is a continuous process of capturing images,
uploading to server, model training and then deployment, it is imperative to find ways to compress images and hence,
save bandwidth.


Problem statement

(![problem_definition_flowchart](https://github.com/user-attachments/assets/d0a69765-4d84-458a-be80-637d54041170))


## How are we trying to solve the problem?

The idea is to extract the most meaningful information from the image using attention weights from the Mask2Former model chosen. The decoder, which is part of the transformer decoder, is able to attend to what it thinks are the most important part of the image post masked attention. These parts of the images are then encoded or are allotted a higher bandwidth as compared to the others.

This way, we propose a differential encoding and decoding framework which assigns if the patch is important or not based on the attention score that it generates.


Our Framework

![Our_framework_flowchart](https://github.com/user-attachments/assets/535cbf31-347d-4e36-871c-b03bac54d8f5)



## What are the results?

Although, not beating the traditional JPEG compression is all aspects, there are certain points when having higher bandwidth capacity would result in better model training performance using our framework. Although the main idea of the research work was to reduce the bandwidth or the file sizes when uploading information, one interesting find was that our method beats jpeg method when storing the information on disk after decompressing. For file storage after decompression, we used the well known PNG format as this retains as much information and is a lossless way of storing images, unlike JPEG.


### Creating the different model usecases

![create_model_usecase_20240730_141402](https://github.com/user-attachments/assets/a56b6161-cfb0-42c1-947f-ad6691023806)


### Usecase 70

![Usecase_70unseen_Patchify_pipeline_20240730_145522](https://github.com/user-attachments/assets/697d27f3-b28d-4a93-b4d9-b6d26976200f)

![Usecase_70unseen_Patchify_pipeline_disk_memusage_20240730_144305](https://github.com/user-attachments/assets/a1fb77c3-cf59-4e4d-bbfe-f45fd30ef721)


### Usecase 50

![Usecase_50unseen_Patchify_pipeline_20240730_145522](https://github.com/user-attachments/assets/fa6e1261-f536-4607-ba8d-8d1b94d0649d)

![Usecase_50unseen_Patchify_pipeline_disk_memusage_20240730_144306](https://github.com/user-attachments/assets/2f9f616d-2e84-4ab2-a9d9-a692d6105e42)


### Usecase 30

![Usecase_30unseen_Patchify_pipeline_20240730_145523](https://github.com/user-attachments/assets/53f5b45d-73d3-4398-93a9-288e4464eac9)


## Miscellaneous images

### Patched Image

![patched_results](https://github.com/user-attachments/assets/40f8de74-f03e-4552-929a-889e7868f383)


### Attention map visualization

![attn_weights_patched_results](https://github.com/user-attachments/assets/be647734-1ce0-4279-b86f-abbfc1e4ee55)


### Most important patches

![kept_color_patches](https://github.com/user-attachments/assets/549981e8-72f9-44a0-a2d8-bb266c41a5c5)

![_kept_indices](https://github.com/user-attachments/assets/7479f07b-fdba-418e-9d9a-4f9c6a5fa491)



## Future TODO

1. Elaborate on the readme document on how to set the project up.
    - Create a docker setup where user would simply go the container and be able to run tests.
    - Have a subsequent requirements.txt file to download all the dependency.
    - Detail out on how to download the Cityscapes dataset.

2. Fix the bug w.r.t the subset allocation. Either use mmlab backend or create own method to be able to select a subset of the whole dataset.

3. Make a more configurable way of training models and running different usecases.
    - Make more dynamic config files.
    - Usage of hydra would also enhance the configurability.
