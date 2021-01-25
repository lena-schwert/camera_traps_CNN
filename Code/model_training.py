# %% IMPORTS

#
import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
import ray
from ray import tune

# Data stuff
import os
import platform
import socket
import time
import tqdm
import copy
import pandas as pd
from custom_dataset_transformations import IslandConservationDataset

if socket.gethostname() == 'Schlepptop':
    os.chdir('/home/lena/git/research_project/')
elif socket.gethostname() == 'ml3-gpu2':
    os.chdir('/home/lena.schwertmann/git/camera_traps_CNN')
else:
    print("Please specify the working directory manually!")

print(f'Working directory changed to: {os.getcwd()}')

image_directory = os.path.join(os.getcwd(), 'image_data')

# %% LOAD PRETRAINED RESNET-18

# torchvision version: 0.8.1
model_resnet18_pytorch = torch.hub.load(repo_or_dir = 'pytorch/vision:v0.8.1', model = 'resnet18',
                                        pretrained = True)

# %% LOAD THE METADATA/LABELS

images_metadata = pd.read_pickle(os.path.join(os.getcwd(), 'Code/images_metadata_preprocessed.pkl'))


# %% TRAIN FUNCTION


def train(train_loader, model, optimizer, device, criterion):
    model.train()
    epoch_loss = 0
    batch_time = []
    for batch_index, (image_batch, label_batch) in enumerate(train_loader):
        start_batch = time.perf_counter()
        # transfer data to active device (not sure whether necessary)
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        # reset the gradients to zero, so they don't accumulate
        optimizer.zero_grad()
        # calculate model output on batch
        prediction_logit = model(image_batch)  # TODO make prediction using the model
        # TODO make sure that the labels are one-hot encoded (= probability distribution)
        # calculate the loss, input in shape of (prediction, labels)
        batch_error = criterion(prediction_logit, label_batch)
        # do backpropagation using the error
        batch_error.backward()
        # update the weights using the calculated gradients
        optimizer.step()
        # sum up the loss values over all the batches
        epoch_loss += batch_error.data
        # batch_error = batch_error.detach()
        # print(f'Cumulative loss at batch {batch_index} in current epoch: {epoch_loss}')
        end_batch = time.perf_counter()
        print(f'Error of current batch is: {batch_error}')
        print(f'Runtime of batch {batch_index} is {round(end_batch - start_batch, 2)}')
        batch_time.append(end_batch - start_batch)

    epoch_loss = epoch_loss / train_loader.__len__()

    return epoch_loss  # , batch_time


# %% VALIDATE FUNCTION

def validate(data, model, criterion):
    model.eval()
    with torch.no_grad():

        # data: torch.utils.data.dataset.Subset
        X_validate = data.dataset.tensors[0][data.indices]
        y_validate = data.dataset.tensors[1][data.indices]

        # calculate loss all at once (no batches needed)
        # TODO check input with function declaration
        validate_loss = None

    return validate_loss


# %% SPECIFY TRANSFORMATIONS

# noinspection DuplicatedCode
transformations_simple_ResNet18 = transforms.Compose([transforms.RandomCrop((1024, 1280)),
                                                      # (height, width) resize all images to smallest common image size
                                                      transforms.ToTensor(),
                                                      # creates FloatTensor scaled to the range [0,1]
                                                      transforms.Normalize(
                                                          mean = [0.485, 0.456, 0.406],
                                                          std = [0.229, 0.224, 0.225])])

# %% MAIN CODE SET UP THE DATA

### decide for a class selection dict

top_5_categories = {'empty': 0, 'rat': 7, 'rabbit': 22, 'petrel': 21, 'iguana': 3}

### decide on the number of instances that are used

# TODO has yet to be implemented

### load the subset of the Island Conservation Dataset

dataset_top_5_categories = IslandConservationDataset(img_base_dir = image_directory,
                                                     images_metadata_dataframe = images_metadata,
                                                     dict_of_categories = top_5_categories,
                                                     transformations = transformations_simple_ResNet18)
print(f'Categories used are: {dataset_top_5_categories.class_selection_dict}')

### do a train, validate, test split

train_size, validate_size, test_size = (0.6, 0.3, 0.1)
# test_size = 1 - (train_size + validate_size)
# if train_size + validate_size + test_size != 1:
#     raise ValueError("Train, validate, test proportions do not sum to 1!")
# use PyTorch's random_split
train_length = int(validate_size * dataset_top_5_categories.__len__())
val_length = int(validate_size * (dataset_top_5_categories.__len__() - train_length))
test_length = dataset_top_5_categories.__len__() - (train_length + val_length)

# the data subsets are of type torch.utils.data.dataset.Subset
train_data, temp = random_split(dataset_top_5_categories, [train_length, val_length + test_length])
validate_data, test_data = random_split(temp, [val_length, test_length])

# %% EXPLORE + ADAPT THE MODEL

### adapt model architecture to IslandConservationDataset

# clone the model with deepcopy!
model_resnet18_adapted = copy.deepcopy(model_resnet18_pytorch)
# adapt the output dimensions according to the class selection!
# fully connected layer out-of-the-box: Linear(in_features=512, out_features=1000, bias=True)
model_resnet18_adapted.fc = nn.Linear(in_features = 512, out_features = len(top_5_categories),
                                      bias = True)

# check whether dimensionality of input is suitable

# debug_loader_all = DataLoader(dataset_top_5_categories, batch_size = BATCH_SIZE, shuffle = True,
#                               num_workers = 0)
#
# image_batch_debug, labels_batch_debug = next(iter(debug_loader_all))
# output = model_resnet18_adapted(image_batch_debug)
# output.size()  # torch.Size([16, 5])
# --> prediction over 5 classes for each sample

# %% INCLUDE RAY FOR MINIMAL HYPERPARAMETER TUNING

search_space = {'learning_rate': tune.grid_search([0.1, 0.01, 0.001, 0.0001]), # 0.1-0.001 (powers of 10
                'batch_size': tune.grid_search([64, 128, 256])}


                # %% MAIN CODE TRAINING LOOP + TESTING

### create the data_loader for training
BATCH_SIZE = 16
print(f'Batch size used: {BATCH_SIZE}')
train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)
validate_loader = DataLoader(validate_data, batch_size = BATCH_SIZE, shuffle = True,
                             num_workers = 0)

### set up everything for the training loop
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"CUDA is used: {torch.cuda.is_available()}")
print(f'CUDA detects {torch.cuda.device_count()} devices.')
print(f'CUDA device currently used: {torch.cuda.current_device()}')
learning_rate = 0.01
print(f'Learning rate: {learning_rate}')
model_resnet18_adapted.to(device)  # TODO send model to device
optimizer_adam = torch.optim.Adam(model_resnet18_adapted.parameters(), betas = (0.9, 0.999),
                                  # suggested default
                                  eps = 1e-08,  # suggested default, for numerical stability
                                  lr = learning_rate, amsgrad = False)

cross_entropy_loss = nn.CrossEntropyLoss(reduction = 'mean')

writer_tb = SummaryWriter()

N_EPOCHS = 10

runtime_epochs = []

for i in range(N_EPOCHS):
    start_time_epoch = time.perf_counter()
    print(f'Epoch {i} has started.')
    # call train function
    train(train_loader = train_loader, model = model_resnet18_adapted, device = device,
          criterion = cross_entropy_loss, optimizer = optimizer_adam)
    end_time_epoch = time.perf_counter()
    print(f'Epoch {i} of {N_EPOCHS} ran for {round(end_time_epoch - start_time_epoch, 2)} seconds.')
    runtime_epochs.append(round(end_time_epoch - start_time_epoch, 2))  # TODO collect metadata about the training)

# TODO save the model state dict


# TODO validate/test the trained model for accuracy
