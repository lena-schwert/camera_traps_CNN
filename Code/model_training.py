# %% IMPORTS

#
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support, top_k_accuracy_score, plot_confusion_matrix
import fire
import types

# Data stuff
import os
import socket
import time
from datetime import datetime
from tqdm import tqdm
import copy
import pandas as pd
import numpy as np
from custom_dataset import IslandConservationDataset
import pickle
import json
import warnings

# visualization stuff
import matplotlib.pyplot as plt

if socket.gethostname() == 'Schlepptop':
    os.chdir('/home/lena/git/research_project/')
elif socket.gethostname() == 'ml3-gpu2':
    os.chdir('/home/lena.schwertmann/git/camera_traps_CNN')
else:
    print("Please specify the working directory manually!")

print(f'Working directory changed to: {os.getcwd()}')

start_script = time.perf_counter()

warnings.filterwarnings("ignore")


# %% FOR DEBUGGING
# RESNET_PRETRAINED = True
# model_resnet18_pytorch = torch.hub.load(repo_or_dir = 'pytorch/vision:v0.8.1', model = 'resnet18',
#                                         pretrained = True)
# # load the dataframe containing all label information
# images_metadata = pd.read_pickle(os.path.join(os.getcwd(), 'Code/images_metadata_preprocessed.pkl'))
# image_directory = os.path.join(os.getcwd(), 'image_data')


# %% THE BIG FUNCTION PASSED TO FIRE CALLING EVERYTHING ELSE

# in case you want to CONTINUE TRAINING BY LOADING A MODEL FILE
# check the current device!
# checkpoint = torch.load(None)   # enter a path to model file
# model_resnet18_adapted.load_state_dict(checkpoint['model'], map_location = 'cuda:0')
# optimizer_adam.load_state_dict(checkpoint['optimizer'])
# last_epoch = checkpoint['epoch']
# last_training_loss = checkpoint['training_loss']
# model_resnet18_adapted.to(device)

def train_and_validate(debugging = False, smoke_test = True, image_directory = os.path.join(os.getcwd(), 'image_data'),
                       transformations = 'transformations_simple_ResNet18', train_proportion = 0.6,
                       validate_proportion = 0.2, test_proportion = 0.2, pretrained = True,
                       finetuning_all_layers = False, finetuning_last_layer = True, batch_size = 16,
                       learning_rate = 0.01, class_selection = "top_5_categories",
                       samples_per_class = 100, n_epochs = 10, adam_betas = (0.9, 0.999),
                       # suggested default
                       adam_eps = 1e-08  # suggested default, for numerical stability
                       ):
    """
    # TODO specify all parameter types
    Parameters
    ----------
    adam_betas:
    adam_eps
    batch_size
    class_selection : str
    image_directory
    learning_rate
    n_epochs
    finetuning_all_layers
    finetuning_last_layer
    pretrained
    samples_per_class
    smoke_test : object
    test_proportion
    train_proportion
    transformations : str
    validate_proportion
    """
    if DEBUGGING:
        debugging = True
    samples_per_class = 32 if smoke_test else samples_per_class
    n_epochs = 3 if smoke_test else n_epochs

    ############------------- SET UP LOGGING ---------------#################
    # save all arguments specified for this function in an object
    my_args = types.SimpleNamespace(**locals())
    print(my_args)
    # create directories and loggers
    start_datetime = datetime.now()
    global experiment_identifier
    experiment_identifier = f'{start_datetime.strftime("%d_%m_%Y_%H:%M:%S")}_SYS={socket.gethostname()}_BS={batch_size}_LR={learning_rate}_EPOCHS={n_epochs}_{transformations}_{class_selection}_SPC={samples_per_class}'
    if pretrained:
        experiment_identifier = experiment_identifier + '_PRETRAINED'
    if finetuning_last_layer:
        experiment_identifier = experiment_identifier + '_FINETUNING_LAST'
    if finetuning_all_layers:
        experiment_identifier = experiment_identifier + '_FINETUNING_ALL'
    if smoke_test or debugging:
        experiment_identifier = experiment_identifier + '_DEBUGGING'
    print(f'Saving results to folder: {experiment_identifier}')
    # create Tensorboard writer
    writer_tb = SummaryWriter(log_dir = "runs/" + experiment_identifier, flush_secs = 30)

    # save all function attributes to disk in the trial folder
    path_for_json = os.path.join(os.getcwd(), 'runs/', experiment_identifier, 'options.json')
    with open(path_for_json, 'w') as file:
        json.dump(vars(my_args), file)

    # create dataframe for collecting results per epoch
    results_dataframe = create_logging_dataframe(my_args)

    #############------------- PREPARE FOR TRAINING ---------------#################

    # specify the class selection
    if class_selection == "top_3_categories":
        class_selection_ID_list = [('empty', 0), ('rat', 7), ('rabbit', 22)]
    elif class_selection == "top_5_categories":
        class_selection_ID_list = [('empty', 0), ('rat', 7), ('rabbit', 22), ('petrel', 21),
                                   ('iguana', 3)]
    elif class_selection == "top_10_categories":
        raise NotImplementedError(
            "You still have to extract the categories from the original dataframe images_metadata.")
    else:
        raise ValueError("Class selection not recognized. Specify a valid option")

    # load the resnet model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_resnet18 = load_and_adapt_model(my_args,
                                          class_selection_ID_list = class_selection_ID_list)
    model_resnet18.to(device)

    optimizer_adam = torch.optim.Adam(model_resnet18.parameters(), betas = my_args.adam_betas,
                                      eps = my_args.adam_eps, lr = my_args.learning_rate,
                                      amsgrad = False)
    # loss = multi-class cross-entropy
    cross_entropy_multi_class_loss = nn.BCEWithLogitsLoss()

    ### load the data

    train_validation_data, test_data, train_length, val_length = load_IslandConservation_subset(
        transformations = transformations, image_directory = image_directory,
        class_selection_ID_list = class_selection_ID_list, samples_per_class = samples_per_class,
        data_splits = (train_proportion, validate_proportion, test_proportion))

    ############------------- TRAINING ---------------#################
    start_datetime = datetime.now()
    last_epoch = False

    print('######### TRAINING LOOP ###########')
    for i in tqdm(range(my_args.n_epochs)):
        start_time_epoch = time.perf_counter()
        print(f'Epoch {i + 1} has started.')
        results_dataframe.epoch_number[i] = i
        if i + 1 == my_args.n_epochs:
            last_epoch = True
        ############--------- DATA LOADERS ---------------#################
        train_data, validate_data = random_split(train_validation_data, [train_length, val_length])
        train_loader = DataLoader(train_data, batch_size = my_args.batch_size, num_workers = 0)
        validate_loader = DataLoader(validate_data, batch_size = my_args.batch_size,
                                     num_workers = 0)
        ############------------- TRAINING ---------------#################
        model_resnet18.train()
        start_time_epoch_train = time.perf_counter()
        epoch_train_loss = train(train_loader = train_loader, model = model_resnet18,
                                 device = device, criterion = cross_entropy_multi_class_loss,
                                 optimizer = optimizer_adam, my_args = my_args)
        end_time_epoch_train = time.perf_counter()
        results_dataframe.epoch_number[i] = i
        results_dataframe.train_runtime_min[i] = round(
            (end_time_epoch_train - start_time_epoch_train) / 60, 2)
        results_dataframe.training_loss[i] = float(epoch_train_loss.cpu())
        ############------------- VALIDATION ---------------#################
        # validate the trained model for loss + accuracy
        model_resnet18.eval()
        start_time_epoch_validate = time.perf_counter()
        epoch_validate_loss, classification_metrics = validate(data_loader = validate_loader,
                                                               model = model_resnet18,
                                                               criterion = cross_entropy_multi_class_loss,
                                                               device = device,
                                                               last_epoch = last_epoch)
        end_time_epoch_validate = time.perf_counter()
        # TODO adapat logging with new metrics
        results_dataframe.validation_loss[i] = float(epoch_validate_loss.cpu())
        results_dataframe.mean_accuracy[i] = classification_metrics.get('mean_accuracy')
        results_dataframe.validation_runtime_min[i] = round(
            (end_time_epoch_validate - start_time_epoch_validate) / 60, 2)

        end_time_epoch = time.perf_counter()

        # save the results dataframe to disk with current results
        file_name = os.path.join(experiment_identifier, 'results.csv')
        results_dataframe.to_csv(path_or_buf = os.path.join(os.getcwd(), 'runs', file_name),
                                 index = False)
        # log everything to Tensorboard
        for parameter_name, values in model_resnet18.named_parameters():
            if values.requires_grad is True:
                writer_tb.add_histogram(parameter_name, values, i)
                writer_tb.add_histogram(f'{parameter_name}.grad', values.grad, i)
        metric_dict = {'loss/training_loss': epoch_train_loss,
                       'loss/validation_loss': epoch_validate_loss,
                       'classification_metrics/top1_accuracy': classification_metrics.get(
                           'top1_accuracy'), 'timings/total_epoch_runtime_min': round(
                (end_time_epoch - start_time_epoch) / 60, 2), 'timings/train_runtime_min': round(
                (end_time_epoch_train - start_time_epoch_train) / 60, 2),
                       'timings/validation_runtime_min': round(
                           (end_time_epoch_validate - start_time_epoch_validate) / 60, 2)}

        for key, value in metric_dict.items():
            writer_tb.add_scalar(key, value, i)

        writer_tb.flush()
        print(f'Saved results of epoch {i + 1} to disk.')
        print(
            f'Epoch {i + 1} of {my_args.n_epochs} ran for {round((end_time_epoch - start_time_epoch_train) / 60, 2)} minutes.')

    ############------------- AFTER COMPLETING ALL EPOCHS ---------------#################
    writer_tb.add_hparams(
        hparam_dict = {'batch_size': my_args.batch_size, 'learning_rate': my_args.learning_rate,
                       'class_selection': my_args.class_selection,
                       'samples_per_class': my_args.samples_per_class,
                       'pretrained': my_args.pretrained,
                       'finetuning_last_layer': my_args.finetuning_last_layer, },
        metric_dict = metric_dict)
    writer_tb.flush()
    writer_tb.close()
    # save the model + optimizer such that training could be resumed (after transferring to CPU)
    # source: https://pytorch.org/tutorials/beginner/saving_loading_models#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
    file_name_model = os.path.join(experiment_identifier, 'model_checkpoint.tar')
    torch.save(
        {'epoch': i, 'model': model_resnet18.state_dict(), 'optimizer': optimizer_adam.state_dict(),
         'training_loss': epoch_train_loss}, os.path.join(os.getcwd(), 'runs', file_name_model))
    print('Final model saved to disk.')
    end_script = time.perf_counter()
    print(f'Total script ran for {round((end_script - start_script) / 60, 2)} minutes.')
    print(f'Local time is {datetime.now().strftime("%d.%m.%Y %H:%M:%S")}.')


# %% FUNCTION TO LOAD THE DATA

def load_IslandConservation_subset(transformations, image_directory, class_selection_ID_list,
                                   samples_per_class, data_splits):
    print('############ DATA USED ################')
    # load the dataframe containing all label information
    images_metadata = pd.read_pickle(
        os.path.join(os.getcwd(), 'Code/images_metadata_preprocessed.pkl'))

    # specify the transformations
    if transformations == 'transformations_simple_ResNet18':
        image_transformations = transforms.Compose([transforms.Resize((1000, int(1000 * 1.4))),
                                                    # (height, width) resize all images to same size with median image ratio 1.4
                                                    transforms.ToTensor(),
                                                    # creates FloatTensor scaled to the range [0,1]
                                                    transforms.Normalize(
                                                        mean = [0.485, 0.456, 0.406],
                                                        std = [0.229, 0.224, 0.225])])
    elif transformations == 'transformations_like_literature':
        image_transformations = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor(),
             # creates FloatTensor scaled to the range [0,1]
             transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    else:
        raise ValueError("Please specify a valid string to select the image transformations.")

    # load the dataset using the custom dataset class
    dataset_subset = IslandConservationDataset(img_base_dir = image_directory,
                                               images_metadata_dataframe = images_metadata,
                                               list_of_categories = class_selection_ID_list,
                                               transformations = image_transformations,
                                               samples_per_class = samples_per_class)

    # do train, validation and test splits
    number_of_instances = dataset_subset.__len__()
    train_size, validate_size, test_size = data_splits
    train_length = int(train_size * number_of_instances)
    val_length = int(validate_size * number_of_instances)
    test_length = number_of_instances - (train_length + val_length)
    if train_length + val_length + test_length != number_of_instances:
        raise ValueError("Dataset split lengths incorrect. Doublecheck!")

    train_validate_data, test_data = random_split(dataset_subset,
                                                  [train_length + val_length, test_length])

    print(f'Categories used are {dataset_subset.class_encoding}')
    print(f'Training on {number_of_instances} samples.')

    return train_validate_data, test_data, train_length, val_length


# %% FUNCTION TO LOAD AND ADAPT THE MODEL

def load_and_adapt_model(my_args, class_selection_ID_list):
    if my_args.finetuning_last_layer + my_args.finetuning_all_layers != 1:
        raise ValueError("Check finetuning options. Only one should be set to True at the time.")

    model_resnet18_adapted = torch.hub.load(repo_or_dir = 'pytorch/vision:v0.8.1',
                                            model = 'resnet18', pretrained = my_args.pretrained)
    if my_args.finetuning_last_layer:
        # set all gradients to zero = FREEZE THE MODEL STATE FOR FINETUNING
        for parameter_name, param in model_resnet18_adapted.named_parameters():
            if param.requires_grad is True:
                param.requires_grad = False

    # adapt the output dimensions according to the class selection!
    # fully connected layer has as many dimensions as classes selected
    model_resnet18_adapted.fc = nn.Linear(in_features = model_resnet18_adapted.fc.in_features,
                                          out_features = len(class_selection_ID_list), bias = True)
    # TODO change, needs to come from somewhere else!
    # doublecheck the requires_grad attribute
    print('######## MODEL ##########')
    print('These are the layers that are trained:')
    for parameter_name, param in model_resnet18_adapted.named_parameters():
        if param.requires_grad is True:
            print(parameter_name, param.requires_grad)
    print(f"CUDA is used: {torch.cuda.is_available()}")
    print(f'CUDA detects {torch.cuda.device_count()} device(s).')
    if torch.cuda.is_available():
        print(f'CUDA device currently used: {torch.cuda.current_device()}')

    return model_resnet18_adapted


# %% CREATE PANDAS DATAFRAME FOR LOGGING TRAINING

def create_logging_dataframe(my_args):
    results_dataframe = pd.DataFrame(
        columns = ['epoch_number', 'train_runtime_min', 'training_loss', 'validation_loss',
                   'mean_accuracy', 'validation_runtime_min', 'batch_size', 'learning_rate',
                   'pretrained', 'finetuning_all_layers', 'finetuning_last_layer',
                   'class_selection', 'samples_per_class', 'transformations'],
        index = np.arange(my_args.n_epochs))
    results_dataframe['batch_size'] = results_dataframe['batch_size'].fillna(
        value = my_args.batch_size)
    results_dataframe['learning_rate'] = results_dataframe['learning_rate'].fillna(
        value = my_args.learning_rate)
    results_dataframe['pretrained'] = results_dataframe['pretrained'].fillna(
        value = my_args.pretrained)
    results_dataframe['finetuning_all_layers'] = results_dataframe['finetuning_all_layers'].fillna(
        value = my_args.finetuning_all_layers)
    results_dataframe['finetuning_last_layer'] = results_dataframe['finetuning_last_layer'].fillna(
        value = my_args.finetuning_last_layer)
    results_dataframe['class_selection'] = results_dataframe['class_selection'].fillna(
        value = my_args.class_selection)
    results_dataframe['samples_per_class'] = results_dataframe['samples_per_class'].fillna(
        value = my_args.samples_per_class)
    results_dataframe['transformations'] = results_dataframe['transformations'].fillna(
        value = my_args.transformations)

    return results_dataframe


# %% TRAIN FUNCTION PER EPOCH

def train(train_loader, model, optimizer, device, criterion, my_args):
    print('\nTraining...')
    epoch_loss = 0
    batch_time = []
    for batch_index, (image_batch, label_batch) in enumerate(train_loader):
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
        if my_args.debugging:
            print(f'Error of current batch is: {batch_error}')
            print(f'Runtime of batch {batch_index}/{train_loader.__len__()} is {round(end_batch - start_batch, 2)}')
        batch_time.append(end_batch - start_batch)

    epoch_loss = epoch_loss / train_loader.__len__()

    return epoch_loss


# %% VALIDATE FUNCTION

def validate(data_loader, model, criterion, device, last_epoch: bool):
    print('\nValidating...')
    validate_loss = 0
    predictions = torch.Tensor().long().to(device)
    prediction_probabilities = torch.Tensor().long().to(device)
    true_labels = torch.Tensor().long().to(device)
    with torch.no_grad():
        for batch_index, (image_batch, label_batch) in enumerate(data_loader):
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            prediction_logit = model(image_batch)
            batch_error = criterion(prediction_logit, label_batch)
            validate_loss += batch_error.data

            # calculate class predictions for the batch
            prediction_prob = F.softmax(prediction_logit, dim = 1)
            prediction_class_1D = torch.argmax(prediction_prob, dim = 1)
            true_class_1D = torch.argmax(label_batch, dim = 1)

            # add class predictions and true labels to epoch-long tensor
            predictions = torch.cat((predictions, prediction_class_1D))
            true_labels = torch.cat((true_labels, true_class_1D))
            prediction_probabilities = torch.cat((prediction_probabilities, prediction_prob), dim = 0)

    ### calculate all metrics and the loss using scikit-learn
    if torch.cuda.is_available():
        predictions = predictions.cpu()
        true_labels = true_labels.cpu()
        prediction_probabilities = prediction_probabilities.cpu()

    classification_metrics = calculate_evaluation_metrics(labels = true_labels,
                                                          predictions = predictions,
                                                          prediction_probs = prediction_probabilities)

    # save all objects to disk for easy access after training
    if last_epoch:
        path = os.path.join(os.getcwd(), 'runs', experiment_identifier)
        torch.save(predictions, os.path.join(path, 'predictions.pt'))
        torch.save(true_labels, os.path.join(path, 'labels.pt'))
        torch.save(prediction_probabilities, os.path.join(path, 'prediction_probabilities.pt'))
        with open(os.path.join(path, 'confusion_matrix.pickle'), 'wb') as file:
            pickle.dump(classification_metrics.get('confusion_matrix'), file, pickle.HIGHEST_PROTOCOL)

    # with open(os.path.join(path, 'confusion_matrix.pickle'), 'rb') as f:
    #     # The protocol version used is detected automatically, so we do not
    #     # have to specify it.
    #     data = pickle.load(f)

    # mean validation loss
    validate_loss = validate_loss / data_loader.__len__()

    return validate_loss, classification_metrics


def calculate_evaluation_metrics(labels, predictions, prediction_probs):
    """

    Parameters
    ----------
    labels : torch.Tensor
    predictions : torch.Tensor
    prediction_probs : torch.Tensor
    """
    if predictions.size() != labels.size():
        raise ValueError("Labels and prediction tensors must be of same length!")

    # top-1 accuracy
    top1_accuracy = accuracy_score(y_true = labels, y_pred = predictions)
    # top-5 accuracy
    top5_accuracy = top_k_accuracy_score(y_true = labels, y_score = prediction_probs, k = 5, normalize = True)
    # precision, recall, f1 score and support per class
    precision, recall, f1_score, support = precision_recall_fscore_support(y_true = labels, y_pred = predictions, average = None, beta = 1)
    # confusion matrix
    confuse_matrix = confusion_matrix(y_true = labels, y_pred = predictions)

    # calculations from: https://stackoverflow.com/questions/50666091/true-positive-rate-and-false-positive-rate-tpr-fpr-for-multi-class-data-in-py
    # true positives = diagonal of the confusion matrix (as many elements as classes)
    TP = np.diag(confuse_matrix)
    # predictions per class = confuse_matrix.sum(axis = 0)
    # samples per class (true labels) = confuse_matrix.sum(axis = 1)

    # false positives = samples predicted wrongly predicted as class X
    # --> subtract true positives from samples per class predicted as that class
    FP = confuse_matrix.sum(axis = 0) - TP
    # false negatives = the number of samples that truly belong to a specific class, but were not classifies as that class
    # --> subtract the true positives from the number of true samples (= support) per class
    FN = confuse_matrix.sum(axis = 1) - TP
    # true negatives = all samples that are left over
    # all elements of the -ith position in the four vectors will sum up to the number of samples
    # e.g. TP[4]+FP[4]+FN[4]+TN[4] = number_of_samples
    TN = confuse_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # false negative rate (fraction of false negatives of all truly positive examples)
    FNR = FN / (FN + TP)
    # false positive rate/fall-out (fraction of false positives of all truly negative examples)
    FPR = FP / (FP + TN)

    return {'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'support': support,
            'confusion_matrix': confuse_matrix,
            'false_negative_rate': FNR,
            'false_positive_rate': FPR}


DEBUGGING = False

if __name__ == '__main__':
    if DEBUGGING:
        train_and_validate()
    else:
        fire.Fire({'train+validate': train_and_validate,
                   'test': validate})
