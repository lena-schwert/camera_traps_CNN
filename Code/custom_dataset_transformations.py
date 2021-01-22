"""
@author: Lena Schwertmann
@objective: load the Island Conservation camera traps dataset
@timespan: November, 11th 2020 -
"""
# %% Imports

# for interaction with the system
import os
import platform
import subprocess

if platform.system() == 'Linux':
    os.chdir('/home/lena/git/research_project/')
elif platform.system() == 'Windows':
    os.chdir('L:\\Dokumente\\git\\research_project')
else:
    print("Please specify the working directory manually!")

# neural network stuff
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torchvision.datasets.folder import default_loader

# data processing
import pandas as pd
import json
from PIL import Image

# visualization
import matplotlib.pyplot as plt
from skimage import io
import seaborn as sns

# load custom utility functions
from IslandConservationDataset_utilities import show_image_given_its_ID, show_labels_given_image_ID
from IslandConservationDataset_utilities import return_image_batch_given_category_ID

# Global Settings
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 15)


# %% Load the dataframe with all metadata/labels for each valid image

images_metadata = pd.read_pickle(os.path.join(os.getcwd(), 'Code/images_metadata_preprocessed.pkl'))

image_directory = "/home/lena/git/research_project/image_data/"


# %% Create a custom PyTorch dataset

# example: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class IslandConservationDataset(Dataset):
    """
    Creates a PyTorch Dataset from the raw island conservation dataset files.



    from class doc: "All datasets that represent a map from keys to data samples should subclass it."
    """

    def __init__(self, img_base_dir, images_metadata_dataframe, dict_of_categories, transformations = None):
        """
        :type dict_of_categories: dict
        :param img_base_dir (string): absolute path to the image directory
        :param images_metadata_dataframe (pandas.DataFrame): dataframe containing file paths + labels
        :param transformations (torchvision.transforms.Compose): object that summarizes all transformations
        """
        ### create a subset of the dataset containing only the specified category IDs
        global class_selection_indices
        self.class_ID_selection = dict_of_categories.values()
        self.class_selection_dict = dict_of_categories

        # create a string that is used for filtering the dataset by arbitrarily many categories
        # collect list of row indices

        first_iter = True
        for value in self.class_ID_selection:
            indices_class = images_metadata_dataframe[images_metadata_dataframe['category_id'] == value].index

            if first_iter:
                class_selection_indices = indices_class
                first_iter = False
            else:
                class_selection_indices = class_selection_indices.union(indices_class, sort = None)

        self.images_metadata_dataframe_subset = images_metadata_dataframe.loc[class_selection_indices]
        # reset the index, such that the BatchSampler in DataLoader is not confused!
        self.images_metadata_dataframe_subset.reset_index(inplace = True)

        # TODO: specify the total number of samples/number of samples per class and sample rows!

        ### do remaining assignments required for correct initiation
        self.img_base_dir = img_base_dir
        #self.all_img_file_paths = images_metadata_dataframe_c['file_name']
        #self.labels = images_metadata_dataframe['category_id']
        self.transforms = transformations

        # specifically use the PIL image backend (accimage has too few transforms implemented)
        torchvision.set_image_backend('PIL')

    def __len__(self) -> int:
        """
        This method returns the total number of images in the dataset.
        It accesses the metadata dataframe, and requires that the entries are unique!

        from class doc: "is expected to return the size of the dataset by many sampler implementations"
        """
        # # find length by counting files in Linux
        # if platform.system() != 'Linux':
        #     raise NotImplementedError("Implementation of __len__ only exists for Linux.")
        # img_file_count_linux = int(
        #     subprocess.getoutput(f'find {self.img_base_dir} -type f | wc -l'))
        img_file_count_linux = images_metadata.shape[0]

        return img_file_count_linux

    def __getitem__(self, item):
        """
        Provides an image, given an item/index within the dataset.
        If applicable, does the required transformations

        from class doc: "supports fetching a data sample for a given key"

        :param item: specifies the index in the entire dataset?
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

        # access the image's label
        true_label = self.images_metadata_dataframe_subset['category_id'][item]

        if self.transforms is not None:
            img_transformed = self.transforms(img)
        else:
            img_transformed = img

        # TODO depending on what train() requires, maybe change returned dtype, e.g. to dict
        return img_transformed, true_label


# %% Specify image transformations

transformations_simple = transforms.Compose([
    transforms.RandomCrop((1024, 1280)),  # (height, width) resize all images to smallest common image size
    transforms.ToTensor(),  # creates FloatTensor scaled to the range [0,1]
])

transformations_simple_ResNet18 = transforms.Compose([
    transforms.RandomCrop((1024, 1280)),  # (height, width) resize all images to smallest common image size
    transforms.ToTensor(),  # creates FloatTensor scaled to the range [0,1]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# %% Fulfulling requirements for PyTorch's ResNet implementation

top_5_categories = {
    'empty': 0,
    'rat': 7,
    'rabbit': 22,
    'petrel': 21,
    'iguana': 3
}

dataset = IslandConservationDataset(img_base_dir = image_directory,
                                    images_metadata_dataframe = images_metadata,
                                    dict_of_categories = top_5_categories,
                                    transformations = transformations_simple_ResNet18)

test_img = dataset.__getitem__(3)[0]
test_img.shape  # torch.Size([3, 1024, 1280])

# create a batch as is required by ResNet???
type(test_img)
test_img.unsqueeze(0).shape


# %% Creating a dataloader

# access data in batches using the DataLoader
data_loader = DataLoader(dataset, batch_size = 32, shuffle = False)

# check that the data loader works as expected

# very dangerous code, uses too much memory
# for X_batch in data_loader:
#     print(X_batch)

next(iter(data_loader))

### Look at the returned image tensor
# source: https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch
# TODO look at the returned image tensor












# %% Look at images with matplotlib/tensorboard
image_directory = "/home/lena/git/research_project/image_data/"

# writer = SummaryWriter('runs/visualize_images')




# %% Old hacky code

# makes sure that the code below is not executed, when functions are imported
# gets only executed if .py is executed in terminal
if __name__ == "__main__":
    pass
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


