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

        first_iter = True
        for value in self.class_ID_selection:
            indices_class = images_metadata_dataframe[
                images_metadata_dataframe['category_id'] == value].index
            # uniformly sample a given amount of samples per class from the indices
            if samples_per_class is not None:
                indices_class = pd.Int64Index(np.random.choice(indices_class.values,
                                                               size = samples_per_class, replace = False))
            if first_iter:
                class_selection_indices = indices_class
                first_iter = False
            else:
                class_selection_indices = class_selection_indices.union(indices_class, sort = None)

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


if __name__ == "__main__":

    # %% Load the dataframe with all metadata/labels for each valid image

    images_metadata = pd.read_pickle(
        os.path.join(os.getcwd(), 'Code/images_metadata_preprocessed.pkl'))

    image_directory = "/home/lena/git/research_project/image_data/"
    ## %% Specify image transformations

    transformations_simple = transforms.Compose([transforms.RandomCrop((1024, 1280)),
                                                 # (height, width) resize all images to smallest common image size
                                                 transforms.ToTensor(),
                                                 # creates FloatTensor scaled to the range [0,1]
                                                 ])

    transformations_simple_ResNet18 = transforms.Compose([transforms.RandomCrop((1024, 1280)),
                                                          # (height, width) resize all images to smallest common image size
                                                          transforms.ToTensor(),
                                                          # creates FloatTensor scaled to the range [0,1]
                                                          transforms.Normalize(
                                                              mean = [0.485, 0.456, 0.406],
                                                              std = [0.229, 0.224, 0.225])])

    # %% Fulfulling requirements for PyTorch's ResNet implementation

    top_5_categories = {'empty': 0, 'rat': 7, 'rabbit': 22, 'petrel': 21, 'iguana': 3}

    dataset = IslandConservationDataset(img_base_dir = image_directory,
                                        images_metadata_dataframe = images_metadata,
                                        list_of_categories = top_5_categories,
                                        transformations = transformations_simple_ResNet18)

    # test_img = dataset.__getitem__(3)[0]
    # test_img.shape  # torch.Size([3, 1024, 1280])
    #
    # # create a batch as is required by ResNet???
    # type(test_img)
    # test_img.unsqueeze(0).shape

    # %% Creating a dataloader

    # check that the data loader works as expected

    # very dangerous code, uses too much memory
    # for X_batch in data_loader:
    #     print(X_batch)

    # access data in batches using the DataLoader
    batch_size = 32
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = 0)

    # start_time_batch = time.perf_counter()
    # next(iter(data_loader))
    # end_time_batch = time.perf_counter()
    # print(f"Loading batch of size {batch_size} took {round(end_time_batch-start_time_batch, 2)} seconds.")

    # try to iterate through the dataloader once
    start_time_data_loader_loop = time.perf_counter()
    iteration = 0
    for batch_index, (image_batch, label_batch) in tqdm(enumerate(data_loader)):
        # print(image_batch)
        # print(label_batch)
        iteration += 1
        if iteration % 100 == 0:
            print(
                f'Loop is already running for {round(time.perf_counter() - start_time_data_loader_loop, 2) / 60} minutes.')
            print(f'We are at batch {batch_index} of {data_loader.__len__()}')
    end_time_data_loader_loop = time.perf_counter()
    print(
        f"Iterating through the dataloader with batch size {batch_size} took {round(end_time_data_loader_loop - start_time_data_loader_loop, 2)} seconds.")

    ### Look at the returned image tensor
    # source: https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch
    # TODO look at the returned image tensor

    ### Look at an image batch

    from IslandConservationDataset_utilities import show_images

    image_batch, label_batch = next(iter(data_loader))

    image_batch_torchvision = torchvision.utils.make_grid(image_batch)

    show_images(image_batch_torchvision, title = label_batch)

# %% Look at images with matplotlib/tensorboard

# writer = SummaryWriter('runs/visualize_images')


# %% Old hacky code

# makes sure that the code below is not executed, when functions are imported
# gets only executed if .py is executed in terminal

### Using Microsoft CameraTraps tools
#
# ### using data_management/databases
#
# # easy access to class count
# from MicrosoftCameraTraps.data_management.databases.sanity_check_json_db import \
#     sanity_check_json_db
#
# sanity_check_json_db("/home/lena/git/research_project/island_conservation.json")
#
# from MicrosoftCameraTraps.data_management.cct_json_to_filename_json import \
#     convertJsonToStringList
#
# rel_file_path_list, out_file_name = convertJsonToStringList(
#     inputFilename = "/home/lena/git/research_project/island_conservation.json",
#     outputFilename = "island_conservation_all_rel_file_paths.json")
#
# ### using data_management/cct_json_utils.py
# from MicrosoftCameraTraps.data_management.cct_json_utils import CameraTrapJsonUtils
#
# # def order_db_keys(db: JSONObject) -> OrderedDict:
# json_ordered = CameraTrapJsonUtils.order_db_keys(json_loaded)
#
# from MicrosoftCameraTraps.data_management.cct_json_utils import IndexedJsonDb
#
# json_db_island_conserv = IndexedJsonDb(
#     "/home/lena/git/research_project/island_conservation.json")
#
