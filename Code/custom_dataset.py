"""
@author: Lena Schwertmann
@objective: load the Island Conservation camera traps dataset
@timespan: November, 11th 2020 -
"""
# %% Imports

# for interaction with the system
import os
import socket

if socket.gethostname() == 'Schlepptop':
    os.chdir('/home/lena/git/research_project/')
elif socket.gethostname() == 'ml3-gpu2':
    os.chdir('/home/lena.schwertmann/git/camera_traps_CNN')
else:
    print("Please specify the working directory manually!")

# neural network stuff
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

# data processing
import pandas as pd
import numpy as np

# visualization

# utility functions
import time
from tqdm import tqdm


# %% Create a custom PyTorch dataset


class IslandConservationDataset(Dataset):
    """
    Creates a PyTorch Dataset from the raw island conservation dataset files.



    from class doc: "All datasets that represent a map from keys to data samples should subclass it."
    """

    def __init__(self, img_base_dir, images_metadata_dataframe, list_of_categories,
                 transformations = None, samples_per_class = None):
        """
        :type samples_per_class: int
        :type list_of_categories: list of tuples
        :param img_base_dir (string): absolute path to the image directory
        :param images_metadata_dataframe (pandas.DataFrame): dataframe containing file paths + labels
        :param transformations (torchvision.transforms.Compose): object that summarizes all transformations
        """

        ### create one-hot encoding for class_selection that is used in __getitem__
        self.class_encoding = []
        # one-hot encode the labels
        class_encoder_label = 0
        for encoder_index in list_of_categories:
            self.class_encoding.append((class_encoder_label, encoder_index[0], encoder_index[1]))
            class_encoder_label += 1

        ### create a subset of the dataset containing only the specified category IDs
        global class_selection_indices
        self.class_ID_selection = []
        for i in list_of_categories:
            self.class_ID_selection.append(i[1])

        # collect list of row indices of type pandas.Index

        i = 0
        first_iter = True
        for value in self.class_ID_selection:
            indices_class = images_metadata_dataframe[images_metadata_dataframe['category_id'] == value].index
            #print(indices_class.__len__())
            #print(f'Class {self.class_encoding[i][1]} has {indices_class.__len__()} images.')
            self.class_encoding[i] = self.class_encoding[i]+(indices_class.__len__(),)
            i += 1
            # uniformly sample a given amount of samples per class from the indices
            if samples_per_class is not 'all':
                indices_class = pd.Int64Index(np.random.choice(indices_class.values,
                                                               size = samples_per_class, replace = False))
            if first_iter:
                class_selection_indices = indices_class
                first_iter = False
            else:
                class_selection_indices = class_selection_indices.union(indices_class, sort = None)

            ### code used once for storing the ordered class count
            # class_count_sorted = sorted(self.class_encoding, key = lambda tup: tup[3], reverse = True)
            # # save list of tuples to CSV
            # import csv
            # with open('Lenas_sorted_class_count.csv', 'w') as out:
            #     csv_out = csv.writer(out)
            #     csv_out.writerow(['my_encoded_value', 'species_name', 'original_encoding', 'image count'])
            #     csv_out.writerows(class_count_sorted)

        self.images_metadata_dataframe_subset = images_metadata_dataframe.loc[
            class_selection_indices]

        # reset the index, such that the BatchSampler in DataLoader is not confused!
        self.images_metadata_dataframe_subset.reset_index(inplace = True)
        # important that this happens AFTER REINDEXING, such that the row indices are the same!

        # encode the label to create tensor of suitable size (must be same size as prediction
        # output by the network to calculate cross entropy)
        self.class_encoding_lookup_tensor = torch.zeros(
            size = (self.images_metadata_dataframe_subset.shape[0], len(self.class_encoding)))

        print('Accessing the labels...')
        i = 0
        for label in tqdm(self.images_metadata_dataframe_subset['category_id']):
            temporary_row_tensor = torch.zeros(len(self.class_encoding))
            for entry in self.class_encoding:
                if entry[2] == label:
                    true_label_look_up = entry[0]
                    temporary_row_tensor[true_label_look_up] = 1.
            self.class_encoding_lookup_tensor[i, :] = temporary_row_tensor
            i += 1

        ### do remaining assignments required for correct initiation
        self.img_base_dir = img_base_dir
        self.transforms = transformations

        # specifically use the PIL image backend (accimage has too few transforms implemented)
        torchvision.set_image_backend('PIL')

    def __len__(self) -> int:
        """
        This method returns the total number of images in the dataset.
        It accesses the metadata dataframe, and requires that the entries are unique!

        from class doc: "is expected to return the size of the dataset by many sampler implementations"
        """
        img_file_count = self.images_metadata_dataframe_subset.shape[0]

        return img_file_count

    def __getitem__(self, item):
        """
        Provides an image, given an item/index within the dataset.
        If applicable, does the required transformations

        from class doc: "supports fetching a data sample for a given key"

        :param item: int specifies the index in the entire dataset (is never of multiple dimensions,
        the batch sampler simply calls this method multiple times and then concats the results)
        :return: tuple of the transformed image and its label as integer
        """
        ### copied from microsoft/CameraTraps/train_classifier.py
        # access the relative file path for the requested 'item'
        img_rel_path = self.images_metadata_dataframe_subset['file_name'][item]
        # if applicable, create the absolute file path using img_base_dir
        if self.img_base_dir is not None:
            img_abs_path = os.path.join(self.img_base_dir, img_rel_path)
        else:
            img_abs_path = img_rel_path
        # load the image itself using Torch's default_loader
        img = default_loader(img_abs_path)  # will be loaded as PIL.Image

        if img.mode != 'RGB':
            raise TypeError(
                f"The loaded image is not in RGB mode. Doublecheck image {img_rel_path}.")

        # access the image's one-hot encoded label
        true_label_encoded = self.class_encoding_lookup_tensor[item]

        if self.transforms is not None:
            img_transformed = self.transforms(img)
        else:
            img_transformed = img

        # TODO depending on what train() requires, maybe change returned dtype, e.g. to dict
        return img_transformed, true_label_encoded

