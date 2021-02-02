# %% IMPORTS

#
import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import ray
from ray import tune
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Data stuff
import os
import platform
import socket
import time
from datetime import datetime
from tqdm import tqdm
import copy
import pandas as pd
import numpy as np
from custom_dataset import IslandConservationDataset
import warnings

if socket.gethostname() == 'Schlepptop':
    os.chdir('/home/lena/git/research_project/')
elif socket.gethostname() == 'ml3-gpu2':
    os.chdir('/home/lena.schwertmann/git/camera_traps_CNN')
else:
    print("Please specify the working directory manually!")

print(f'Working directory changed to: {os.getcwd()}')

image_directory = os.path.join(os.getcwd(), 'image_data')
start_script = time.perf_counter()

warnings.filterwarnings("ignore")

# %% LOAD PRETRAINED RESNET-18

RESNET_PRETRAINED = True

# torchvision version: 0.8.1
model_resnet18_pytorch = torch.hub.load(repo_or_dir = 'pytorch/vision:v0.8.1', model = 'resnet18',
                                        pretrained = RESNET_PRETRAINED)

# %% LOAD THE METADATA/LABELS

images_metadata = pd.read_pickle(os.path.join(os.getcwd(), 'Code/images_metadata_preprocessed.pkl'))


# %% TRAIN FUNCTION


def train(train_loader, model, optimizer, device, criterion):
    print('Training...')
    epoch_loss = 0
    batch_time = []
    for batch_index, (image_batch, label_batch) in tqdm(enumerate(train_loader)):
        start_batch = time.perf_counter()
        # transfer data to active device (not sure whether necessary)
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        # reset the gradients to zero, so they don't accumulate
        optimizer.zero_grad()
        # calculate model output on batch
        prediction_logit = model(image_batch)
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
        # print(f'Error of current batch is: {batch_error}')
        # print(f'Runtime of batch {batch_index} is {round(end_batch - start_batch, 2)}')
        batch_time.append(end_batch - start_batch)

    epoch_loss = epoch_loss / train_loader.__len__()

    return epoch_loss


# %% VALIDATE FUNCTION

def validate(data_loader, model, criterion):
    print('Validating...')
    validate_loss = 0
    mean_accuracy = []
    with torch.no_grad():
        for batch_index, (image_batch, label_batch) in tqdm(enumerate(data_loader)):
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            prediction_logit = model(image_batch)
            batch_error = criterion(prediction_logit, label_batch)
            validate_loss += batch_error.data

            # calculate classification accuracy
            prediction_prob = F.softmax(prediction_logit, dim = 1)
            prediction_class_1D = torch.argmax(prediction_prob, dim = 1).cpu().numpy()
            true_class_1D = torch.argmax(label_batch, dim = 1).cpu().numpy()

            # confusion_matrix_5classes = confusion_matrix(y_true = true_class_1D,
            #                                              y_pred = prediction_class_1D,
            #                                              normalize = "all").ravel()

            #classific_report = classification_report(y_true = true_class_1D, y_pred = prediction_class_1D)

            accuracy = accuracy_score(y_true = true_class_1D, y_pred = prediction_class_1D)
            mean_accuracy.append(accuracy)

    mean_accuracy = sum(mean_accuracy) / len(mean_accuracy)
    validate_loss = validate_loss / data_loader.__len__()

    return validate_loss, mean_accuracy


# %% SPECIFY TRANSFORMATIONS

# noinspection DuplicatedCode
transformations_simple_ResNet18 = transforms.Compose([transforms.Resize((1000, int(1000 * 1.4))),
                                                      # (height, width) resize all images to same size with median image ratio 1.4
                                                      transforms.ToTensor(),
                                                      # creates FloatTensor scaled to the range [0,1]
                                                      transforms.Normalize(
                                                          mean = [0.485, 0.456, 0.406],
                                                          std = [0.229, 0.224, 0.225])])

# %% SET UP THE DATA
global BATCH_SIZE_TRAIN
global BATCH_SIZE_VALIDATE
global LEARNING_RATE
global CLASS_SELECTION
global SAMPLES_PER_CLASS

if socket.gethostname() == 'Schlepptop':
    BATCH_SIZE_TRAIN = 16
    BATCH_SIZE_VALIDATE = 16
    LEARNING_RATE = 0.01
    CLASS_SELECTION = "top_5_categories"
    SAMPLES_PER_CLASS = 10
elif socket.gethostname() == 'ml3-gpu2':
    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_VALIDATE = 64
    LEARNING_RATE = 0.01
    CLASS_SELECTION = "top_5_categories"
    SAMPLES_PER_CLASS = 1000
else:
    print("Error, error!")

### decide for a class selection dict

if CLASS_SELECTION == "top_3_categories":
    class_selection_ID_list = [('empty', 0), ('rat', 7), ('rabbit', 22)]
elif CLASS_SELECTION == "top_5_categories":
    class_selection_ID_list = [('empty', 0), ('rat', 7), ('rabbit', 22), ('petrel', 21),
                               ('iguana', 3)]
else:
    raise ValueError("Class selection not recognized. Specify a valid option")

### load the subset of the Island Conservation Dataset

dataset_top_5_categories = IslandConservationDataset(img_base_dir = image_directory,
                                                     images_metadata_dataframe = images_metadata,
                                                     list_of_categories = class_selection_ID_list,
                                                     transformations = transformations_simple_ResNet18,
                                                     samples_per_class = SAMPLES_PER_CLASS)
print(f'Categories used are: {dataset_top_5_categories.class_encoding}')

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

# %% ADAPT THE MODEL

### adapt model architecture to IslandConservationDataset
# clone the model with deepcopy!
model_resnet18_adapted = copy.deepcopy(model_resnet18_pytorch)

# set all gradients to zero = FREEZE THE MODEL STATE FOR FINETUNING
for name, param in model_resnet18_adapted.named_parameters():
    if param.requires_grad is True:
        param.requires_grad = False

# adapt the output dimensions according to the class selection!
# fully connected layer out-o
model_resnet18_adapted.fc = nn.Linear(in_features = model_resnet18_pytorch.fc.in_features,
                                      out_features = len(class_selection_ID_list), bias = True)

# doublecheck the requires_grad attribute
print('These are the layers that are trained:')
for name, param in model_resnet18_adapted.named_parameters():
    if param.requires_grad is True:
        print(name, param.requires_grad)

# %% INCLUDE RAY FOR MINIMAL HYPERPARAMETER TUNING

# search_space = {'LEARNING_RATE': tune.grid_search([0.1, 0.01, 0.001, 0.0001]),  # 0.1-0.001 (powers of 10
#                 'batch_size': tune.grid_search([64, 128, 256])}


# %% PREPARE FOR TRAINING

### create the data_loader for training
print(f'{SAMPLES_PER_CLASS} samples per class are used.')
print(f'Batch size used for training: {BATCH_SIZE_TRAIN}')
print(f'Batch size used for validation: {BATCH_SIZE_VALIDATE}')
print(f'Learning rate: {LEARNING_RATE}')

### set up everything for the training loop
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"CUDA is used: {torch.cuda.is_available()}")
print(f'CUDA detects {torch.cuda.device_count()} devices.')
# print(f'CUDA device currently used: {torch.cuda.current_device()}')

model_resnet18_adapted.to(device)
optimizer_adam = torch.optim.Adam(model_resnet18_adapted.parameters(), betas = (0.9, 0.999),
                                  # suggested default
                                  eps = 1e-08,  # suggested default, for numerical stability
                                  lr = LEARNING_RATE, amsgrad = False)

cross_entropy_multi_class_loss = nn.BCEWithLogitsLoss()

N_EPOCHS = 10
print(f'Number of epochs: {N_EPOCHS}')

results_dataframe = pd.DataFrame(
    columns = ['epoch_number', 'epoch_runtime_min', 'train_loss', 'validate_loss',
               'validate_accuracy', 'validate_runtime_min', 'batch_size_train',
               'batch_size_validate', 'learning_rate', 'pretrained', 'class_selection',
               'samples_per_class', 'transformations'], index = np.arange(10))
results_dataframe['batch_size_train'] = results_dataframe['batch_size_train'].fillna(
    value = BATCH_SIZE_TRAIN)
results_dataframe['batch_size_validate'] = results_dataframe['batch_size_validate'].fillna(
    value = BATCH_SIZE_VALIDATE)
results_dataframe['learning_rate'] = results_dataframe['learning_rate'].fillna(
    value = LEARNING_RATE)
results_dataframe['pretrained'] = results_dataframe['pretrained'].fillna(value = RESNET_PRETRAINED)
results_dataframe['class_selection'] = results_dataframe['class_selection'].fillna(
    value = str(class_selection_ID_list))
results_dataframe['samples_per_class'] = results_dataframe['samples_per_class'].fillna(
    value = SAMPLES_PER_CLASS)
results_dataframe['transformations'] = results_dataframe['transformations'].fillna(
    value = str(transformations_simple_ResNet18.transforms))

# %% TRAINING LOOP

start_datetime = datetime.now()
experiment_identifier = f'{start_datetime.strftime("%d_%m_%Y_%H:%M:%S")}_SYS={socket.gethostname()}_BS={BATCH_SIZE_TRAIN}_LR={LEARNING_RATE}_EPOCHS={N_EPOCHS}_{CLASS_SELECTION}_SPC={SAMPLES_PER_CLASS}'
writer_tb = SummaryWriter(comment = experiment_identifier + "_transform_resize")

for i in tqdm(range(N_EPOCHS)):
    start_time_epoch = time.perf_counter()
    print(f'Epoch {i} has started.')
    results_dataframe.epoch_number[i] = i
    ############--------- DATA LOADERS ---------------#################
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE_TRAIN, shuffle = True,
                              num_workers = 0)
    validate_loader = DataLoader(validate_data, batch_size = BATCH_SIZE_VALIDATE, shuffle = True,
                                 num_workers = 0)
    ############------------- TRAINING ---------------#################
    model_resnet18_adapted.train()
    start_time_epoch_train = time.perf_counter()
    epoch_train_loss = train(train_loader = train_loader, model = model_resnet18_adapted,
                             device = device, criterion = cross_entropy_multi_class_loss,
                             optimizer = optimizer_adam)
    end_time_epoch_train = time.perf_counter()
    results_dataframe.epoch_number[i] = i
    results_dataframe.epoch_runtime_min[i] = round(
        (end_time_epoch_train - start_time_epoch_train) / 60, 2)
    results_dataframe.train_loss[i] = float(epoch_train_loss.cpu())
    ############------------- VALIDATION ---------------#################
    # validate the trained model for loss + accuracy
    model_resnet18_adapted.eval()
    start_time_epoch_validate = time.perf_counter()
    epoch_validate_loss, epoch_mean_accuracy = validate(data_loader = validate_loader,
                                                        model = model_resnet18_adapted,
                                                        criterion = cross_entropy_multi_class_loss)
    end_time_epoch_validate = time.perf_counter()
    results_dataframe.validate_loss[i] = float(epoch_validate_loss.cpu())
    results_dataframe.validate_accuracy[i] = epoch_mean_accuracy
    results_dataframe.validate_runtime_min[i] = round(
        (end_time_epoch_validate - start_time_epoch_validate) / 60, 2)

    # add current metrics to tensorboard
    writer_tb.add_scalars('loss',
                          {'epoch_train_loss': epoch_train_loss,
                           'epoch_validate_loss': epoch_validate_loss}, i)
    writer_tb.add_scalar('epoch_mean_accuracy', epoch_mean_accuracy, i)

    # save the results dataframe to disk with current results
    file_name = f'{experiment_identifier}.csv'
    results_dataframe.to_csv(path_or_buf = os.path.join(os.getcwd(), 'results', file_name),
                             index = False)
    print(f'Saved results of epoch {i} to disk.')
    end_time_epoch = time.perf_counter()
    print(
        f'Epoch {i} of {N_EPOCHS} ran for {round((end_time_epoch - start_time_epoch_train) / 60, 2)} minutes.')

writer_tb.close()
# save the model (after transferring to CPU)
file_name_model = f'{start_datetime.strftime("%d_%m_%Y_%H:%M:%S")}_SYS={socket.gethostname()}_BS={BATCH_SIZE_TRAIN}_LR={LEARNING_RATE}_N_EPOCHS={N_EPOCHS}_{CLASS_SELECTION}_SPC={SAMPLES_PER_CLASS}.pt'
torch.save(model_resnet18_adapted, os.path.join(os.getcwd(), 'results', file_name_model))
print('Final model saved to disk.')
end_script = time.perf_counter()
print(f'Total script ran for {round((end_script - start_script) / 60, 2)} minutes.')
