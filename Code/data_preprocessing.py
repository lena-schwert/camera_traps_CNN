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

# Global Settings
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 15)


# %% Load the dataframe with all metadata/labels for each valid image

images_metadata = pd.read_pickle(os.path.join(os.getcwd(), 'images_metadata_preprocessed.pkl'))


# %% Create a custom PyTorch dataset

# example: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class IslandConservationDataset(Dataset):
    """
    Creates a PyTorch Dataset from the raw island conservation dataset files.

    from class doc: "All datasets that represent a map from keys to data samples should subclass it."
    """

    def __init__(self, img_base_dir, all_img_file_paths, label_file_path, label_dataframe,
                 transformations = None):
        """
        :param img_base_dir (string):
        :param all_img_file_paths (pandas.Series): list of relative file paths for all images
        :param label_file_path (string):
        :param label_dataframe (pandas.DataFrame):
        :param transformations (torchvision.transforms.Compose):
        """
        self.img_base_dir = img_base_dir
        self.all_img_file_paths = all_img_file_paths
        self.label_file_path = label_file_path
        self.transforms = transformations
        self.labels = label_dataframe

        # TODO pass list of class labels to use in __getitem__ to self
        self.class_selection = None

        # specifically use the PIL image backend (accimage has too few transforms implemented)
        torchvision.set_image_backend('PIL')

    def __len__(self) -> int:
        """
        This method returns the total number of images in the dataset.

        from class doc: "is expected to return the size of the dataset by many sampler implementations"
        """
        # TODO clarify whether to use image or label count for return
        # find length by counting files in Linux
        if platform.system() != 'Linux':
            raise NotImplementedError("Implementation of __len__ only exists for Linux.")
        img_file_count_linux = int(
            subprocess.getoutput(f'find {self.img_base_dir} -type f | wc -l'))

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
        img_rel_path = self.all_img_file_paths[item]
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

        # TODO access the label given the image
        true_label = 2
        if len(true_label) > 1:
            raise ValueError("Image has more than one label. Classifier can only handle 1 class per image.")
        # TODO make sure that multiple labels can be handled if this can technically happen


        if self.transforms is not None:
            img_transformed = self.transforms(img)
        else:
            img_transformed = img

        # TODO depending on what train() requires, maybe change returned dtype, e.g. to dict
        return img_transformed, true_label


# %% Specify image transformations

# TODO choose image transformation types
transformations_simple = transforms.Compose([transforms.RandomCrop((1024, 1280)),
                                             # (height, width) resize all images to smallest common image size
                                             transforms.PILToTensor()]
                                            # creates FloatTensor in range [0,1]
                                            )

# RandomHorizontalFlip
# Normalize , but needs to be applied on tensor not PILImage stype

# %% Creating a dataloader
image_directory = "/home/lena/git/research_project/image_data/"
path_to_label_json = "/home/lena/git/research_project/island_conservation.json"

all_data = IslandConservationDataset(img_base_dir = image_directory,
                                     label_file_path = path_to_label_json,
                                     all_img_file_paths = images['file_name'],
                                     transformations = transformations_simple,
                                     label_dataframe = annotations)

all_data.__len__()
all_data.__getitem__(2)

# access data in batches using the DataLoader
data_loader = DataLoader(all_data, batch_size = 256, shuffle = False)

for (image, label) in data_loader:
    print(image)

### Look at the returned image tensor
# source: https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch
# TODO look at the returned image tensor












# %% Look at images with matplotlib/tensorboard
image_directory = "/home/lena/git/research_project/image_data/"

plt.ion()
plt.imshow(io.imread(os.path.join(image_directory, json_loaded['images'][72000]['file_name'])))
plt.show()

# writer = SummaryWriter('runs/visualize_images')




# %% Old hacky code

# makes sure that the code below is not executed, when functions are imported
# gets only executed if .py is executed in terminal
if __name__ == "__main__":

    # %% Using Microsoft CameraTraps tools

    ### using data_management/databases

    # easy access to class count
    from MicrosoftCameraTraps.data_management.databases.sanity_check_json_db import \
        sanity_check_json_db

    sanity_check_json_db("/home/lena/git/research_project/island_conservation.json")

    from MicrosoftCameraTraps.data_management.cct_json_to_filename_json import \
        convertJsonToStringList

    rel_file_path_list, out_file_name = convertJsonToStringList(
        inputFilename = "/home/lena/git/research_project/island_conservation.json",
        outputFilename = "island_conservation_all_rel_file_paths.json")
    # TODO figure out whether the string list of relative file paths can be of use

    ### using data_management/cct_json_utils.py
    from MicrosoftCameraTraps.data_management.cct_json_utils import CameraTrapJsonUtils

    # def order_db_keys(db: JSONObject) -> OrderedDict:
    json_ordered = CameraTrapJsonUtils.order_db_keys(json_loaded)

    from MicrosoftCameraTraps.data_management.cct_json_utils import IndexedJsonDb

    json_db_island_conserv = IndexedJsonDb(
        "/home/lena/git/research_project/island_conservation.json")



