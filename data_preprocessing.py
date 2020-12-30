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


# %% Load JSON label file as dataframe

def object_to_string_dtype(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtype == 'object':
            dataframe[i] = dataframe[i].astype('string')
        else:
            pass
    return dataframe


# import with json as dict
data = open('island_conservation.json')
labels = json.load(data)  # loads json as dict

# convert dict to data frame
images = pd.json_normalize(labels['images'])  # works
annotations = pd.json_normalize(labels['annotations'])  # works
categories = pd.json_normalize(labels['categories'])  # works
info = pd.json_normalize(labels['info'])  # works

images = object_to_string_dtype(images)
annotations = object_to_string_dtype(annotations)
info = object_to_string_dtype(info)

# add column to images that includes width_x_height
# both columns have dtype int64
# apply function to each row
images = images.assign(width_x_height = lambda x: (x['width'].map(str)+"x"+x['height'].map(str)))

# %% Using Microsoft CameraTraps tools

### using data_management/databases

# easy access to class count
from MicrosoftCameraTraps.data_management.databases.sanity_check_json_db import sanity_check_json_db
sanity_check_json_db("/home/lena/git/research_project/island_conservation.json")

from MicrosoftCameraTraps.data_management.cct_json_to_filename_json import convertJsonToStringList
rel_file_path_list, out_file_name = convertJsonToStringList(
    inputFilename = "/home/lena/git/research_project/island_conservation.json",
    outputFilename = "island_conservation_all_rel_file_paths.json")
# TODO figure out whether the string list of relative file paths can be of use


### using data_management/cct_json_utils.py
from MicrosoftCameraTraps.data_management.cct_json_utils import CameraTrapJsonUtils
# def order_db_keys(db: JSONObject) -> OrderedDict:
json_ordered = CameraTrapJsonUtils.order_db_keys(json_loaded)


from MicrosoftCameraTraps.data_management.cct_json_utils import IndexedJsonDb

json_db_island_conserv = IndexedJsonDb("/home/lena/git/research_project/island_conservation.json")

# %% Create a custom PyTorch dataset

# example: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
image_directory = "/home/lena/git/research_project/image_data/"
path_to_label_json = "/home/lena/git/research_project/island_conservation.json"

class IslandConservationDataset(Dataset):
    """
    Creates a PyTorch Dataset from the raw island conservation dataset files.

    from class doc: "All datasets that represent a map from keys to data samples should subclass it."
    """
    def __init__(self, img_base_dir, label_file_path, all_img_file_paths, transformations = None):
        """

        :param all_img_file_paths: comes from the JSON with labels
        :param img_base_dir (string):
        :param label_file_path (string):
        :param transformations (torchvision.transforms.Compose):
        """
        self.img_base_dir = img_base_dir
        self.all_img_file_paths = all_img_file_paths
        self.label_file_path = label_file_path
        self.transforms = transformations

        # TODO pass list of class labels to use in __getitem__ to self
        self.class_labels = None

        # specifically use the PIL image backend (accimage has too few transforms implemented)
        torchvision.set_image_backend('PIL')

    def __len__(self):
        """
        This method returns the total number of images in the dataset.

        from class doc: "is expected to return the size of the dataset by many sampler implementations"
        """
        # TODO clarify whether to use image or label count for return
        # find length by counting files in Linux
        if platform.system() != 'Linux':
            raise NotImplementedError("Implementation of __len__ only exists for Linux.")
        img_file_count_linux = int(subprocess.getoutput(f'find {self.img_base_dir} -type f | wc -l'))

        return img_file_count_linux

    def __getitem__(self, item) -> (torch.Tensor, int):
        """
        Provides an image, given an item/index within the dataset.
        If applicable, does the required transofrmations

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
            raise TypeError(f"The loaded image is not in RGB mode. Doublecheck image {img_rel_path}.")

        # TODO access the label given the image
        true_label = None

        # TODO apply transformations here
        if self.transforms is not None:
            img_transformed = self.transforms(img)
        else:
            img_transformed = img

        # TODO depending on what train() requires, maybe change returned dtype, e.g. to dict
        return img_transformed, true_label


# %% Specify image transformations

# TODO choose image transformation types
transformations = transforms.Compose(
    [transforms.RandomCrop((1024, 1280)),   # (height, width) resize all images to smallest common image size
     transforms.PILToTensor()]  # creates FloatTensor in range [0,1]
)
# RandomHorizontalFlip
# Normalize , but needs to be applied on tensor not PILImage stype

# %% Creating a dataloader
all_data = IslandConservationDataset(img_base_dir = image_directory,
                                     label_file_path = path_to_label_json,
                                     all_img_file_paths = images['file_name'],
                                     transformations = transformations)

all_data.__len__()
all_data.__getitem__(2)

data_loader = DataLoader(all_data, batch_size = 256, shuffle = True)

### Look at the returned image tensor



# %% Answering small questions about the dataset?

### Are there more/less labels than images?
# How many labels are there in total?
# dataframe annotations has 142,341 rows
annotations.columns
unique_annotation_IDs = set(annotations['id'])
len(unique_annotation_IDs)  # 142341
unique_annotation_image_ids = set(annotations['image_id'])
len(unique_annotation_image_ids)  # 127,410

# How many images are there in total?
#... acccording to the image path file
# dataframe images has 127,410 rows
# How many unique IDs?
unique_image_IDs = set(images['id'])
len(unique_image_IDs)  # 127,410
unique_image_file_names = set(images['file_name'])
len(unique_image_file_names)  # 127,410
print(f"There are {unique_image_IDs} unique image IDs.")

#... according to Linux?
image_directory = "/home/lena/git/research_project/image_data/"
image_file_count_linux = int(subprocess.getoutput(f'find {image_directory} -type f | wc -l'))
print(f"Linux finds {image_file_count_linux} image files.")  # 122,602

# Do all image IDs appear in the annotations? (= set difference)
unique_annotation_image_ids.difference(unique_image_IDs)  # empty set
unique_image_IDs.difference(unique_annotation_image_ids)  # empty set

# Where does the count difference come from?

### How do images look like that have multiple labels? Do they always hve bboxes?

# aggregate over annotations dataframe


### What do images with the 'unknown' label look like?


### Are there any images that have the 'human' label? (they said they removed those)
human_id = int(categories['id'][categories['name'] == 'human'])





### What are the image sizes (width/height) like?
images['width_x_height'].unique()
images['width_x_height'].value_counts()
# What are the most/least common image sizes?
# plot as histogram
plot_dist_image_sizes = sns.displot(images['width_x_height'])
plot_dist_image_sizes.set_xticklabels(rotation = 45, horizontalalignment='right')
plt.tight_layout()
plt.show()
# There are 8 different image formats


# %% Look at images with matplotlib/tensorboard
image_directory = "/home/lena/git/research_project/image_data/"

plt.ion()
plt.imshow(io.imread(os.path.join(image_directory,
    json_loaded['images'][72000]['file_name'])))
plt.show()

#writer = SummaryWriter('runs/visualize_images')



# %% Old hacky code

# makes sure that the code below is not executed, when functions are imported
# gets only executed if .py is executed in terminal
if __name__ == "__main__":
    pass
