

import os
import pandas as pd

# neural network stuff
import torch
import torch.nn as nn



# %% Load the metadata dataframe

images_metadata = pd.read_pickle(os.path.join(os.getcwd(), 'Code/images_metadata_preprocessed.pkl'))

image_directory = " "  # TODO specify correct path on server

# %% check whether all the files exist in the local directory

# set current working directory

# loop through all file paths
non_existing_image_IDs = []
for path in images_metadata['file_name']:
    current_abs_img_path = os.path.join(image_directory, path)
    if os.path.isfile(current_abs_img_path) is not True:
        # get image ID of current image
        current_img_ID = images_metadata['image_id'][images_metadata['file_name'] == path].iloc[0]
        non_existing_image_IDs.append(current_img_ID)
