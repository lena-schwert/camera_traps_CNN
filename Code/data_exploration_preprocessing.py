
# %% imports

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

# %% Loading the annotations + file paths

def object_to_string_dtype(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtype == 'object':
            dataframe[i] = dataframe[i].astype('string')
        else:
            pass
    return dataframe


# import with json as dict
data = open('island_conservation.json')
labels_json = json.load(data)  # loads json as dict

# convert dict to data frame
images = pd.json_normalize(labels_json['images'])  # works
annotations = pd.json_normalize(labels_json['annotations'])  # works
categories = pd.json_normalize(labels_json['categories'])  # works
info = pd.json_normalize(labels_json['info'])  # works

images = object_to_string_dtype(images)
annotations = object_to_string_dtype(annotations)
info = object_to_string_dtype(info)

# add column to images that includes width_x_height
# both columns have dtype int64
# apply function to each row
images = images.assign(
    width_x_height = lambda x: (x['width'].map(str) + "x" + x['height'].map(str)))


# %% The Dataset class

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
        # TODO make sure that multiple labels can be handled if necessary

        # TODO apply transformations here
        if self.transforms is not None:
            img_transformed = self.transforms(img)
        else:
            img_transformed = img

        # TODO depending on what train() requires, maybe change returned dtype, e.g. to dict
        return img_transformed, true_label

transformations_simple = transforms.Compose([transforms.RandomCrop((1024, 1280)),
                                             # (height, width) resize all images to smallest common image size
                                             transforms.PILToTensor()]
                                            # creates FloatTensor in range [0,1]
                                            )





# %% Utility functions

image_directory = "/home/lena/git/research_project/image_data/"

def show_image_given_its_ID(image_identifier: str, image_dataframe):
    """
    Given a image ID (words separated by _), accesses the file path (separated by /)
    to plot and open the respective image using PIL.

    Requires the GLOBAL image_directory to create the absolute file path (on my Laptop).
    Requires the dataframe images.

    :param image_identifier: image ID as string (words separated by _)
    :return: opens the image in an external program
    """
    image_rel_path = image_dataframe['file_name'][image_dataframe['image_id'] == image_identifier].iloc[0]
    image = Image.open(os.path.join(image_directory, image_rel_path))
    return image.show()


def show_labels_given_image_ID(image_identifier: str):
    """
    Given a image ID (words separated by _), provides the number of labels
    and which animals are found on the image.

    Requires the dataframes: annotations, categories

    :param image_identifier: image ID as string (words separated by _)
    :return: dictionary with label count + animal name(s)
    """
    labels = annotations[annotations['image_id'] == image_identifier]
    image_categories = labels['category_id'].unique()
    image_categories_names = []
    for i in image_categories:
        image_categories_names.append(categories['name'][categories['id'] == i].iloc[0])
        return {'number_of_labels': labels['category_id'].count(),
                'animals_on_this_image': image_categories_names,
                'numerical_categories': labels.unique()
                }

def return_tensorboard_image_batch_given_category_ID(category_ID, how_many_images):
    """
    Requires access to global dataframe image_metadata

    :param category_ID:
    :param how_many_images:
    :return:
    """
    # filter the dataframe w.r.t. the category ID
    indices_class = images_metadata[images_metadata['category_id'] == category_ID].index

    # sample given number from the category
    # use
    transformation = transforms.RandomCrop((1024, 1280))
    images = transformation()
    # tensor must be 4D mini-batch Tensor of shape (B x C x H x W) or a list of images all of the same size.
    torchvision.utils.make_grid(tensor = None,
                                nrow = 10)
    pass

# %% Why are there more/less labels than images?

# How many labels are there in total?
# dataframe annotations has 142,341 rows
unique_annotation_IDs = set(annotations['id'])
len(unique_annotation_IDs)  # 142,341
unique_annotation_image_ids = set(annotations['image_id'])
len(unique_annotation_image_ids)  # 127,410

# How many images are there in total?
# ... acccording to the image path file
# dataframe images has 127,410 rows
# How many unique IDs?
unique_image_IDs = set(images['id'])
len(unique_image_IDs)  # 127,410
unique_image_file_names = set(images['file_name'])
len(unique_image_file_names)  # 127,410
print(
    f"There are {len(unique_image_IDs)} unique image IDs and {len(unique_image_file_names)} unique file names.")

# ... according to Linux?
image_directory = "/home/lena/git/research_project/image_data/"
image_file_count_linux = int(subprocess.getoutput(f'find {image_directory} -type f | wc -l'))
print(f"Linux finds {image_file_count_linux} image files.")  # 122,602

# Do all image IDs appear in the annotations? (= set difference)
unique_annotation_image_ids.difference(unique_image_IDs)  # empty set
unique_image_IDs.difference(unique_annotation_image_ids)  # empty set

# Where does the count difference come from?
# see below
# 1. There are 4808 images that are missing. This corresponds to images with the "human" label.
# 2. The 127,420 unique images have mutiple annotations!
#


# %% Is the difference between number of annotations + images within JSON file explainable? - Yes!

# Find images that have multiple annotations (identify using the image ID)
images_multiple_annotations = annotations[annotations.duplicated(subset = 'image_id', keep = False)]  # show all duplicates
images_multiple_annotations.shape  # 20,858 rows

# safety check: any multiple annotations w.r.t. empty annotation? (ID = 0)
images_multiple_annotations[images_multiple_annotations['category_id'] == 0]

# How many images with multiple annotations are there?
images_multiple_annotations['image_id'].unique().shape  # 5927

# 20,858 >> 5927 (mean: 14 annotations per image)!!! Does it make sense? How can this be explained?
duplicate_annotation_count = images_multiple_annotations.pivot_table(index = 'image_id', aggfunc = 'size')

# How many images have more than 2 annotations?
duplicate_annotation_count[duplicate_annotation_count > 2]  # 1922

# How many images have more than 10 annotations?
duplicate_annotation_count[duplicate_annotation_count > 10]  # 436
# What does an exemplary image look like?
# appears 12 times: micronesia_cam06_cam06june2019_pig_micronesia_cam06_cam06june2019_pig_20190629205325_1390
show_image_given_its_ID(
    'micronesia_cam06_cam06june2019_pig_micronesia_cam06_cam06june2019_pig_20190629205325_1390', )
# there are 12 rats on the image....
show_image_given_its_ID(
    'micronesia_cam06_cam06june2019_pig_micronesia_cam06_cam06june2019_pig_20190629192048_11310', )

duplicate_annotation_count.sum()  # 20,858
# subtract the number of unique images from that number

print \
    (f"There are {images_multiple_annotations['image_id'].unique().shape[0]} images with multiple annotations.")
print \
    (f"This explains the difference between {len(set(annotations['id']))} annotations and {len(set(images['id']))} images according to the original JSON file,")
print \
    (f"because there are {images_multiple_annotations.shape[0 ] -5927} annotations that are duplicates,")
print(f"and 142341-14931 = {142341 -14931}.")

# %% Are there ever images with multiple annotations that have more than one class label?

# use: images_multiple_annotations

# TODO Are there ever images with multiple annotations that have more than one class label?

# idea 1: loop through the dataframe repeatedly, whenever a row already eqist

# idea 2: use an aggregate function: aggregate by unique image_ids and extract each category_id!
# index: "column from df that we want to appear as a unique value in each row"
# values: "the column we want to apply some aggregate operation on"
# aggfunc: "the function that is/are applied"
# images_multiple_annotations.pivot_table(index = 'image_id', values = 'category_id',
#                                         aggfunc = append())
# id

# ideas from skype call

### 1 using group by
images_multiple_annotations_unique_ID = images_multiple_annotations.drop_duplicates \
    (subset = 'image_id', keep = 'first')
# access row indices
images_multiple_annotations_unique_ID.index  # 5927 entries
# this could then be intersectd with
images_multiple_annotations.index  # 20858 entries


bla = images_multiple_annotations.groupby('image_id')['category_id'].count()


### using unique


images_multiple_annotations[images_multiple_annotations.image_id.unique(), 'image_id']
['image_id' == images_multiple_annotations.image_id.unique()]

temp = images_multiple_annotations['image_id'] + "_" + images_multiple_annotations['category_id'].map(str)
images_multiple_annotations.loc[:, 'image_id_and_category_id'] = temp

images_multiple_annotations['image_id+category_id'].unique()
images_multiple_annotations['image_id_and_category_id'].unique()

print()

images_multiple_annotations['image_id_and_category_id']

# show the image with the maximal animal count
bla[bla == bla.max()].index
show_image_given_its_ID(
    "micronesia_cam06_cam06june2019_pig_micronesia_cam06_cam06june2019_pig_20190629195303_1926", )



# %% ### What do images with the 'unknown' label look like?

unknown_category_id = int(categories['id'][categories['name'] == 'unknown'])

images_unknown = annotations.loc[annotations['category_id'] == unknown_category_id]

images_unknown.info()
images_unknown['image_id'].unique()  # there are 941 unique images with that label

# look at an example
show_image_given_its_ID(images_unknown['image_id'].iloc[100], )




# %% ### What do images with the 'human' label look like?

### Are there any images that have the 'human' label? (they said they removed those)
human_category_id = int(categories['id'][categories['name'] == 'human'])

# How many unique images with the human label are there?
images_humans = annotations.loc[annotations['category_id'] == human_category_id]

# How many uniwue images with humans on them are there?
images_humans['image_id'].unique()  # 4808


# %% Which image sizes exist and how many of each are there in the dataset?

images['width_x_height'].unique()
images['width_x_height'].value_counts()
# What are the most/least common image sizes?
# plot as histogram
plot_dist_image_sizes = sns.displot(images['width_x_height'])
plot_dist_image_sizes.set_xticklabels(rotation = 45, horizontalalignment = 'right')
plt.tight_layout()
plt.show()
# There are 8 different image formats

# %% Create a merged dataframe with all key information

### drop all 20,858 (corresponding to 5,927 unique images) rows of images with more than 1 label
images_multiple_annotations = annotations[annotations.duplicated(subset = 'image_id', keep = False)]  # show all duplicates
images_multiple_annotations.info()
# use pandas drop_duplicates to exclude all those rows

annotations_preprocessed = annotations.drop_duplicates(subset = 'image_id', keep = False)
# leaves 121,483 rows  (121,483 + 20,858 = 142,341)

### drop annotation ID column to avoid confusion with image_ID
annotations_preprocessed = annotations_preprocessed.drop('id', axis = 1)

### drop bbox column, because bounding boxes will not be used
annotations_preprocessed = annotations_preprocessed.drop('bbox', axis = 1)

### drop all rows with the label "human"
human_category_id = int(categories['id'][categories['name'] == 'human'])
# How many unique images with the human label are there? - 3767
images_humans = annotations_preprocessed.loc[annotations_preprocessed['category_id'] == human_category_id]

annotations_preprocessed = annotations_preprocessed.drop(index = images_humans.index)
# 117,716 rows are left (121,483 - 3,767 = 117,716)
annotations_preprocessed.info()

### create the new dataframe (= add information from images dataframe to annotations_preprocessed)

# It might be necessary to synchronize the index/column names of both dataframes?
images_preprocessed = images.rename(columns = {'id': 'image_id'})
# The indexes are not the same, but this is not relevant for merge!

# only add the rows from images dataframe that are still contained in annotations_preprocessed
# --> use left-join/merge
images_metadata_preprocessed = pd.merge(left = annotations_preprocessed, right = images_preprocessed,
                                        how = 'left', left_index = False, right_index = False)

### TODO add the category name as string as a new column

### one-time thing: check for existence of each image still contained in the dataframe
images_metadata_preprocessed.info()

non_existing_image_IDs = []
for path in images_metadata_preprocessed['file_name']:
    current_abs_img_path = os.path.join(image_directory, path)
    if os.path.isfile(current_abs_img_path) is not True:
        # get image ID of current image
        current_img_ID = images_metadata_preprocessed['image_id'][images_metadata_preprocessed['file_name'] == path].iloc[0]
        non_existing_image_IDs.append(current_img_ID)

non_existing_image_IDs  # is empty!

# save the dataframe on disk for import by other files

images_metadata_preprocessed.to_pickle(os.path.join(os.getcwd(), 'Code/images_metadata_preprocessed.pkl'))

