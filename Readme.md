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

![Color Image](images/problem_definition_flowchart.jpg)


## How are we trying to solve the problem?

The idea is to extract the most meaningful information from the image using attention weights from the Mask2Former model chosen. The decoder, which is part of the transformer decoder, is able to attend to what it thinks are the most important part of the image post masked attention. These parts of the images are then encoded or are allotted a higher bandwidth as compared to the others.

This way, we propose a differential encoding and decoding framework which assigns if the patch is important or not based on the attention score that it generates.


Our Framework

![Color Image](images/Our_framework_flowchart.jpg)


## What are the results?

Although, not beating the traditional JPEG compression is all aspects, there are certain points when having higher bandwidth capacity would result in better model training performance using our framework. Although the main idea of the research work was to reduce the bandwidth or the file sizes when uploading information, one interesting find was that our method beats jpeg method when storing the information on disk after decompressing. For file storage after decompression, we used the well known PNG format as this retains as much information and is a lossless way of storing images, unlike JPEG.


### Creating the different model usecases

![Color Image](images/create_model_usecase_20240730_141402.png)


### Usecase 70

![Color Image](images/Usecase_70unseen_Patchify_pipeline_20240730_145522.png)

![Color Image](images/Usecase_70unseen_Patchify_pipeline_disk_memusage_20240730_144305.png)


### Usecase 50

![Color Image](images/Usecase_50unseen_Patchify_pipeline_20240730_145522.png)

![Color Image](images/Usecase_50unseen_Patchify_pipeline_disk_memusage_20240730_144306.png)


### Usecase 30

![Color Image](images/Usecase_30unseen_Patchify_pipeline_20240730_145523.png)


## Miscellaneous images

### Patched Image
![Color Image](images/patched_results.png)


### Attention map visualization
![Color Image](images/attn_weights_patched_results.png)


### Most important patches
![Color Image](images/_kept_indices.png)

![Color Image](images/kept_color_patches.png)


## Future TODO

1. Elaborate on the readme document on how to set the project up.
    - Create a docker setup where user would simply go the container and be able to run tests.
    - Have a subsequent requirements.txt file to download all the dependency.
    - Detail out on how to download the Cityscapes dataset.

2. Fix the bug w.r.t the subset allocation. Either use mmlab backend or create own method to be able to select a subset of the whole dataset.

3. Make a more configurable way of training models and running different usecases.
    - Make more dynamic config files.
    - Usage of hydra would also enhance the configurability.
