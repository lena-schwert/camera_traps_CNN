"""
@author: Lena Schwertmann
@objective: load the Island Conservation camera traps dataset
@timespan: November, 11th 2020 -
"""
# %% Imports

# set working directory OS-specific
import os
import platform

if platform.system() == 'Linux':
    os.chdir('/home/lena/git/research_project/')
elif platform.system() == 'Windows':
    os.chdir('L:\\Dokumente\\git\\research_project')
else:
    print("Please specify the working directory manually!")

# data processing
import pandas as pd
import json

# Global Settings
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 15)

# %% Load JSON label file

# import with json as dict
data = open('Data/island_conservation.json')
labels = json.load(data)  # loads json as dict

# convert dict to data frame
images = pd.json_normalize(labels['images'])  # works
annotations = pd.json_normalize(labels['annotations'])  # works
categories = pd.json_normalize(labels['categories'])  # works
info = pd.json_normalize(labels['info'])  # works


# %% Inspect labels

def object_to_string_dtype(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtype == 'object':
            dataframe[i] = dataframe[i].astype('string')
        else:
            pass
    return dataframe


images = object_to_string_dtype(images)
annotations = object_to_string_dtype(annotations)
categories = object_to_string_dtype(categories)
info = object_to_string_dtype(info)

# %%


# %% Old hacky code

# makes sure that the code below is not executed, when functions are imported
# gets only executed if .py is executed in terminal
if __name__ == "__main__":
    pass
