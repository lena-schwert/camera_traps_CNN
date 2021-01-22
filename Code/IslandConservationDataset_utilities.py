# %% Imports

import os
import torch
import torchvision
from PIL import Image
from torchvision import transforms, utils


# %% Utilities

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

def return_image_batch_given_category_ID(category_ID, how_many_images):
    """
    Requires access to global dataframe image_metadata

    :param category_ID:
    :param how_many_images:
    :return:
    """
    # filter the dataframe w.r.t. the category ID
    indices_class = images_metadata[images_metadata['category_id'] == category_ID].index
    images_metadata['']

    # TODO implement with or without the Dataset class

    # sample given number from the category
    # use
    transformation = transforms.RandomCrop((1024, 1280))
    images = transformation()
    # tensor must be 4D mini-batch Tensor of shape (B x C x H x W) or a list of images all of the same size.
    torchvision.utils.make_grid(tensor = None,
                                nrow = 10)
    pass