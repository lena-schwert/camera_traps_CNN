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
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_dataset_transformations import IslandConservationDataset

import pandas as pd
import time
from tqdm import tqdm

# %% Loop through all batches of a dataset

images_metadata = pd.read_pickle(os.path.join(os.getcwd(), 'Code/images_metadata_preprocessed.pkl'))
image_directory = os.path.join(os.getcwd(), 'image_data')

transformations_simple_ResNet18 = transforms.Compose([transforms.RandomCrop((1024, 1280)),
                                                      # (height, width) resize all images to smallest common image size
                                                      transforms.ToTensor(),
                                                      # creates FloatTensor scaled to the range [0,1]
                                                      transforms.Normalize(
                                                          mean = [0.485, 0.456, 0.406],
                                                          std = [0.229, 0.224, 0.225])])

top_5_categories = {'empty': 0, 'rat': 7, 'rabbit': 22, 'petrel': 21, 'iguana': 3}

dataset = IslandConservationDataset(img_base_dir = image_directory,
                                    images_metadata_dataframe = images_metadata,
                                    dict_of_categories = top_5_categories,
                                    transformations = transformations_simple_ResNet18)

batch_size = 32

data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = 0)

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
