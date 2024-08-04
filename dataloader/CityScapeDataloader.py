import os
import pickle
import random
import time

import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

from .dataset.cityscapes import CityScapesDataset


class LoadAndGetDataloader:
    def __init__(
        self, root_dir: str, batch_size: int, num_workers: int = 4, shuffle: bool = True
    ) -> None:
        """Class meant to load and get the dataloader.

        Args:
            root_dir (str): Root directory to where the data is located.
            batch_size (int): The size of the batch
            split (str, optional): How the data should be split. Defaults to train
            transform (None, optional): transforms specified. Defaults to 'train'.
            num_workers (int, optional): Num of workers. Defaults to 4.
            shuffle (bool, optional): Whether the DS should be shuffled. Defaults to True.
        """
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def create_dataset_and_load(
        self,
        train_transform: transforms.Compose,
        valid_transform: transforms.Compose,
        get_attention_map_for_unseen: bool = False,
        get_attention_map_for_training_ds: bool = False,
        train_ds_reduction: float = -1.0,
        write_ds_LUT_to_disk: bool = False,
        seen_data_pkl_file_path: str = 'None',
    ):
        if not get_attention_map_for_unseen and not get_attention_map_for_training_ds:
            train_dataset = CityScapesDataset(
                root=self.root_dir, transform=train_transform, split='train'
            )
            # There are cases when one wants to train the model on reduced dataset
            # This would mean trained on 30%, 50% or 70% of the dataset
            # These scenarios are used for usecases when these models score the remaining
            # training dataset for the patchify compress pipeline
            if train_ds_reduction != -1.0:
                if train_ds_reduction < 0.0 and train_ds_reduction > 1.0:
                    raise ValueError(
                        'Expected value is between 0-1. Received:' f'{train_ds_reduction}'
                    )
                # Set a seed number
                # Call the function
                seed_number = 30
                train_dataset = self.reduce_dataset(
                    dataset=train_dataset,
                    percentage=train_ds_reduction,
                    seed=seed_number,
                    write_ds_LUT_to_disk=write_ds_LUT_to_disk,
                )
            valid_dataset = CityScapesDataset(
                root=self.root_dir, transform=valid_transform, split='val'
            )

        else:
            # In this case here, we want to make our remaining training dataset
            # to validation dataset with those transforms, because we have a
            # pre-trained model that simply extracts the attention maps
            train_dataset = CityScapesDataset(
                root=self.root_dir, transform=train_transform, split='val'
            )
            valid_dataset = CityScapesDataset(
                root=self.root_dir, transform=train_transform, split='train'
            )
            # Use the pkl file now to find out the other files or paths
            if get_attention_map_for_unseen:
                valid_dataset = self.get_unseen_data(valid_dataset, seen_data_pkl_file_path)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

        return train_dataloader, valid_dataloader

    """
    def get_dataloader(self) -> DataLoader:
        Simply returns the dataloader

        Returns:
            DataLoader: Dataloader object to return

        return self.dataloader
    """

    def reduce_dataset(
        self,
        dataset: CityScapesDataset,
        percentage: float,
        seed=None,
        write_ds_LUT_to_disk: bool = False,
    ):
        """
        Reduces the size of the dataset by the percentage specified

        Args:
            dataset (CityScapesDataset): The dataset to reduce
            percentage (float): The percentage of data that needs to be kept.
            seed (int): The seed number to specify. Defaults to None.
        """
        # Calculate the number of samples
        num_samples = len(dataset)
        num_samples_to_keep = int(num_samples * percentage)

        # Check if the seed is not none
        if seed is not None:
            # Global setting and any calls to random.sample() will use
            # same sequence of random numbers, when the seed is the same.
            random.seed(seed)

        # Randomly select the samples to keep
        indices = random.sample(range(num_samples), num_samples_to_keep)

        # Create a reduces dataset bsaed on mmlabs dataset
        # Since this is not overriden here, we will have to add/change
        # code here
        reduced_dataset = dataset.get_subset(indices)

        # Write the reduced dataset information to a LUT only if flag is set
        if write_ds_LUT_to_disk:
            print('Start of getting the dictionary')
            s_time = time.time()
            # Once we have the reduced dataset, create a LUT for the image info
            img_name_to_img_idx = {
                os.path.splitext(os.path.basename(dataset[i]['img_path']))[0]: i for i in indices
            }
            e_time = time.time()
            print(f'Total time taken for loading dictionary: {e_time-s_time}')

            # Save the dictionary to a pickle file
            folder_name_ = 'patches_lut_patch_lut/'
            file_path = os.path.join(
                folder_name_,
                f'actual_training_lb4_gb8_reduced_data_kp_{percentage}_{seed}_.pkl',
            )
            s_time = time.time()
            print('Start dump')
            with open(file_path, 'wb') as file:
                pickle.dump(img_name_to_img_idx, file)
            e_time = time.time()

            print(f'Total time taken for dumping the file: {e_time-s_time}')

        return reduced_dataset

    def get_unseen_data(self, full_dataset: CityScapesDataset, pkl_file_path: str):
        """
        Extracts the unseen dataset from the full dataset using the pkl file.

        Args:
            full_dataset (CityScapesDataset): The full dataset.
            pkl_file_path (str): The path to the pkl file.

        Raises:
            ValueError: If provided pkl file path is invalid.
            ValueError: If provided path is not a pkl file.
        """

        if not os.path.isfile(pkl_file_path):
            raise ValueError(f'provided pkl_path {pkl_file_path} is invalid.')
        if not pkl_file_path.endswith('.pkl'):
            raise ValueError(f'Provided file {pkl_file_path} is not a valid pkl file')

        # Load the dictionary from the pkl file
        with open(pkl_file_path, 'rb') as file:
            seen_data = pickle.load(file)

        # Create the unseen indices list
        unseen_indices = [i for i in range(len(full_dataset)) if i not in seen_data.values()]
        # Create a reduces dataset bsaed on mmlabs dataset
        # Since this is not overriden here, we will have to add/change
        # code here
        unseen_dataset = full_dataset.get_subset(unseen_indices)

        return unseen_dataset
