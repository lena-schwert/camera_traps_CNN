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

def train_and_validate(debugging = False, smoke_test = True,
                       train_validate_data_path = None, test_data_path = None,
                       image_directory = os.path.join(os.getcwd(), 'image_data'),
                       transformations = 'transformations_like_literature', train_proportion = 0.6,
                       validate_proportion = 0.2, test_proportion = 0.2, pretrained = True,
                       finetuning_all_layers = False, finetuning_last_layer = True, batch_size = 16,
                       learning_rate = 0.01, weight_decay = 0, class_selection = 'top_5_categories',
                       samples_per_class = 100, n_epochs = 10, adam_betas = (0.9, 0.999),
                       # suggested default
                       adam_eps = 1e-08  # suggested default, for numerical stability
                       ):
    """
    # TODO specify all parameter types
    Parameters
    ----------
    debugging
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
    samples_per_class: specify 'all' or an integer
    smoke_test : object
    test_proportion
    train_proportion
    transformations : str
    validate_proportion
    weight_decay
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
    experiment_identifier = f'{start_datetime.strftime("%d_%m_%Y_%H:%M:%S")}_SYS={socket.gethostname()}_BS={batch_size}_LR={learning_rate}_WD={weight_decay}_EPOCHS={n_epochs}_{transformations}_{class_selection}_SPC={samples_per_class}'
    if pretrained:
        experiment_identifier = experiment_identifier + '_PRETRAINED'
    if finetuning_last_layer:
        experiment_identifier = experiment_identifier + '_FINETUNING_LAST'
    if finetuning_all_layers:
        experiment_identifier = experiment_identifier + '_FINETUNING_ALL'
    if smoke_test or debugging:
        experiment_identifier = experiment_identifier + '_DEBUGGING'
    print(f'Saving results to folder: {experiment_identifier}')
    global path_for_saving
    path_for_saving = os.path.join('runs', experiment_identifier)
    # instantiating the tensorboard writer also creates the respective directory
    writer_tb = SummaryWriter(log_dir = "runs/" + experiment_identifier, flush_secs = 30)

    # save all function attributes to disk in the trial folder
    path_for_json = os.path.join(os.getcwd(), 'runs/', experiment_identifier, 'options.json')
    with open(path_for_json, 'w') as file:
        json.dump(vars(my_args), file)

    # create dataframe for collecting results per epoch
    results_dataframe = create_logging_dataframe(my_args)
    # look at Lenas_sorted_class_count.csv for extracting the top_k_categories with encoding
    if class_selection == "top_3_categories":
        class_selection_ID_list = [('empty', 0), ('rabbit', 22), ('petrel', 21)]
    elif class_selection == "top_5_categories":
        class_selection_ID_list = [('empty', 0), ('rabbit', 22), ('petrel', 21), ('iguana', 3),
                                   ('rat', 7)]
    elif class_selection == "top_10_categories":
        class_selection_ID_list = [('empty', 0), ('rabbit', 22), ('petrel', 21), ('iguana', 3),
                                   ('rat', 7), ('cat', 5), ('pig', 37), ('goat', 36),
                                   ('shearwater', 26), ('petrel_chick', 23)]
    elif class_selection == "3_small_categories":
        class_selection_ID_list = [('raven', 4), ('donkey', 2), ('monitor_lizard', 46)]
    elif class_selection == "all_49_classes":
        import csv
        with open('category_encoding_all_classes.csv', mode = 'r') as infile:
            reader = csv.reader(infile)
            with open('test.csv', mode = 'w') as outfile:
                writer = csv.writer(outfile)
                all_category_encodings = {rows[0]: rows[1] for rows in reader}
                # drop all classes that have zero images (identified by trial and error)
                all_category_encodings.pop('name', None)
                all_category_encodings.pop('human', None)
                all_category_encodings.pop('unknown', None)
                all_category_encodings.pop('dove', None)
                all_category_encodings.pop('moth', None)
        class_selection_ID_list = [(k, int(v)) for k, v in all_category_encodings.items()]
    else:
        raise ValueError("Class selection not recognized. Specify a valid option")

    #############------------- PREPARE FOR TRAINING ---------------#################

    # specify the class selection

    # load the resnet model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_resnet18 = load_and_adapt_model(my_args,
                                          class_selection_ID_list = class_selection_ID_list)
    model_resnet18.to(device)

    optimizer_adam = torch.optim.Adam(model_resnet18.parameters(), betas = my_args.adam_betas,
                                      eps = my_args.adam_eps, lr = my_args.learning_rate,
                                      amsgrad = False, weight_decay = my_args.weight_decay)

    # checkpoint loading: this code can be used to change the parameters of the optimizer manually!
    # for param_group in optimizer_adam.param_groups:
    #     print(param_group['lr'])
    # loss = multi-class cross-entropy
    cross_entropy_multi_class_loss = nn.BCEWithLogitsLoss()

    ### load the data
    if train_validate_data_path is None and test_data_path is None:
        train_validation_data, test_data, train_length, val_length = load_IslandConservation_subset(
            transformations = transformations, image_directory = image_directory,
            class_selection_ID_list = class_selection_ID_list, samples_per_class = samples_per_class,
            data_splits = (train_proportion, validate_proportion, test_proportion))

    # if data is loaded from disk, it will not be saved to disk in the folder!
    train_validation_data = torch.load(os.path.join(os.getcwd(), train_validate_data_path))
    test_data = torch.load(os.path.join(os.getcwd(), test_data_path))
    # make sure that the correct image base dir is set
    train_validation_data.dataset.img_base_dir = os.path.join(os.getcwd(), 'image_data')

    # calculate train+ val length for train/validation split in the training loop
    number_of_instances = train_validation_data.indices.__len__() + test_data.indices.__len__()
    train_length = int(train_proportion * number_of_instances)
    val_length = int(validate_proportion * number_of_instances)

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
        end_time_epoch = time.perf_counter()
        ############------------- LOGGING ---------------#################
        # TODO make this more efficient, e.g. reuse the metric_dict below
        # solution: https://stackoverflow.com/questions/42632470/how-to-add-dictionaries-to-a-dataframe-as-a-row
        results_dataframe.epoch_number[i] = i
        results_dataframe.training_loss[i] = float(epoch_train_loss.cpu())
        results_dataframe.validation_loss[i] = float(epoch_validate_loss.cpu())
        results_dataframe.train_runtime_min[i] = round(
            (end_time_epoch_train - start_time_epoch_train) / 60, 2)
        results_dataframe.validation_runtime_min[i] = round(
            (end_time_epoch_validate - start_time_epoch_validate) / 60, 2)
        results_dataframe.total_epoch_runtime_min[i] = round(
            (end_time_epoch - start_time_epoch) / 60, 2)
        results_dataframe.top1_accuracy[i] = classification_metrics.get('top1_accuracy')
        results_dataframe.top3_accuracy[i] = classification_metrics.get('top3_accuracy')
        results_dataframe.top5_accuracy[i] = classification_metrics.get('top5_accuracy')
        results_dataframe.precision[i] = classification_metrics.get('precision')
        results_dataframe.recall[i] = classification_metrics.get('recall')
        results_dataframe.f1_score[i] = classification_metrics.get('f1_score')
        results_dataframe.support[i] = classification_metrics.get('support')
        results_dataframe.false_negative_rate[i] = classification_metrics.get('false_negative_rate')
        results_dataframe.false_positive_rate[i] = classification_metrics.get('false_positive_rate')

        # save the results dataframe to disk with current results
        file_name = os.path.join(experiment_identifier, 'results_training.csv')
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
                           'top1_accuracy'),
                       'classification_metrics/top3_accuracy': classification_metrics.get(
                           'top3_accuracy'),
                       'classification_metrics/top5_accuracy': classification_metrics.get(
                           'top5_accuracy'),
                       'classification_metrics/precision_macro_average': np.mean(
                           classification_metrics.get('precision')),
                       'classification_metrics/recall_macro_average': np.mean(
                           classification_metrics.get('recall')),
                       'classification_metrics/f1_score_macro_average': np.mean(
                           classification_metrics.get('f1_score')),
                       'classification_metrics/false_negative_rate_macro_average': np.mean(
                           classification_metrics.get('false_negative_rate')),
                       'classification_metrics/false_positive_rate_macro_average': np.mean(
                           classification_metrics.get('false_positive_rate')),
                       'timings/total_epoch_runtime_min': round(
                           (end_time_epoch - start_time_epoch) / 60, 2),
                       'timings/train_runtime_min': round(
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
                       'weight_decay': my_args.weight_decay,
                       'class_selection': my_args.class_selection,
                       'samples_per_class': my_args.samples_per_class,
                       'transformations': my_args.transformations,
                       'pretrained': my_args.pretrained,
                       'finetuning_last_layer': my_args.finetuning_last_layer,
                       'finetuning_all_layers': my_args.finetuning_all_layers},
        metric_dict = metric_dict)

    writer_tb.flush()
    writer_tb.close()
    # save the model + optimizer such that training could be resumed (after transferring to CPU)
    # source: https://pytorch.org/tutorials/beginner/saving_loading_models#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
    file_name_model = os.path.join(experiment_identifier, 'final_model_checkpoint.tar')
    torch.save(
        {'epoch': i, 'model': model_resnet18.state_dict(), 'optimizer': optimizer_adam.state_dict(),
         'training_loss': epoch_train_loss}, os.path.join(os.getcwd(), 'runs', file_name_model))
    print('Final model saved to disk.')
    end_script = time.perf_counter()
    print(f'Total script ran for {round((end_script - start_script) / 60, 2)} minutes.')
    with open(os.path.join('runs', experiment_identifier, 'total_runtime_min.txt'), 'w') as f:
        f.write(str(round((end_script - start_script) / 60, 2)))
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

    # save dataset_subset to disk, might

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

    torch.save(train_validate_data, os.path.join(path_for_saving, 'train_validate_data.pt'))
    torch.save(test_data, os.path.join(path_for_saving, 'test_data.pt'))

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
        columns = ['epoch_number', 'training_loss', 'validation_loss', 'train_runtime_min',
                   'validation_runtime_min', 'total_epoch_runtime_min', 'top1_accuracy', 'top3_accuracy',
                   'top5_accuracy', 'precision', 'recall', 'f1_score', 'support',
                   'false_negative_rate', 'false_positive_rate', 'batch_size', 'learning_rate',
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
            print(
                f'Runtime of batch {batch_index + 1}/{train_loader.__len__()} is {round(end_batch - start_batch, 2)} seconds.')
        batch_time.append(end_batch - start_batch)

    epoch_loss = epoch_loss / train_loader.__len__()

    return epoch_loss


# %% VALIDATE FUNCTION

def validate(data_loader, model, criterion, device, last_epoch: bool = False):
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
            prediction_probabilities = torch.cat((prediction_probabilities, prediction_prob),
                                                 dim = 0)

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
        torch.save(predictions, os.path.join(path_for_saving, 'predictions_validation_last_epoch.pt'))
        torch.save(true_labels, os.path.join(path_for_saving, 'labels_validation_last_epoch.pt'))
        torch.save(prediction_probabilities, os.path.join(path_for_saving, 'prediction_probabilities_validation_last_epoch.pt'))
        with open(os.path.join(path_for_saving, 'confusion_matrix_validation_last_epoch.pickle'), 'wb') as file:
            pickle.dump(classification_metrics.get('confusion_matrix'), file)

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
    # top-3 accuracy
    top3_accuracy = top_k_accuracy_score(y_true = labels, y_score = prediction_probs, k = 3,
                                         normalize = True)
    # top-5 accuracy
    top5_accuracy = top_k_accuracy_score(y_true = labels, y_score = prediction_probs, k = 5,
                                         normalize = True)
    # precision, recall, f1 score and support per class
    precision, recall, f1_score, support = precision_recall_fscore_support(y_true = labels,
                                                                           y_pred = predictions,
                                                                           average = None, beta = 1)
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

    return {'top1_accuracy': top1_accuracy, 'top3_accuracy': top3_accuracy,  'top5_accuracy': top5_accuracy,
            'precision': precision, 'recall': recall, 'f1_score': f1_score, 'support': support,
            'confusion_matrix': confuse_matrix, 'false_negative_rate': FNR,
            'false_positive_rate': FPR}


def test(path_to_experiment, test_data_filename = 'test_data.pt', test_batch_size = 16):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load the checkpoint to the respective device
    checkpoint = torch.load(os.path.join(path_to_experiment, 'final_model_checkpoint.tar'),
                            map_location = torch.device(device))

    ### load the trained model
    # load the number of classes
    pickle_opener = open(os.path.join(path_to_experiment, 'confusion_matrix_validation_last_epoch.pickle'), 'rb')
    confusion_matrix_trained = pickle.load(pickle_opener)
    class_count = confusion_matrix_trained.shape[0]
    # instantiate model and optimizer without any meaningful parameters
    model_resnet_trained = torch.hub.load(repo_or_dir = 'pytorch/vision:v0.8.1', model = 'resnet18',
                                          pretrained = False)
    model_resnet_trained.fc = nn.Linear(in_features = model_resnet_trained.fc.in_features,
                                        out_features = class_count, bias = True)
    # write the parameters from the checkpoint to model
    model_resnet_trained.load_state_dict(checkpoint['model'])
    model_resnet_trained.eval()
    model_resnet_trained.to(device)

    # # only needed if training should be continued
    # optimizer_adam_trained = torch.optim.Adam(model_resnet_trained.parameters())
    # optimizer_adam_trained.load_state_dict(checkpoint['optimizer'])
    # last_training_epoch = checkpoint['epoch']
    # last_training_loss = checkpoint['training_loss']

    # access the data
    test_data = torch.load(os.path.join(path_to_experiment, test_data_filename))
    # ensure the image base dir is correct
    test_data.dataset.img_base_dir = os.path.join(os.getcwd(), 'image_data')
    test_loader = DataLoader(test_data, batch_size = test_batch_size, shuffle = False, num_workers = 0)

    # calculate all the classification metrics using the test data
    cross_entropy_multi_class_loss = nn.BCEWithLogitsLoss()
    test_loss, test_classification_metrics = validate(data_loader = test_loader, model = model_resnet_trained,
                                                      criterion = cross_entropy_multi_class_loss,
                                                      device = device)
    # write test results to disk
    temp = {k: v.tolist() for k, v in test_classification_metrics.items()}
    with open(os.path.join(path_to_experiment, 'results_test_dataset.json'), 'w') as file:
        json.dump(temp, file)

    print('Successfully wrote results from test dataset to disk.')


DEBUGGING = False

if __name__ == '__main__':
    if DEBUGGING:
        print('CAREFUL, YOU ARE STILL IN DEBUGGING MODE!')
        train_and_validate(smoke_test = False, class_selection = "top_5_categories",
                           train_validate_data_path = 'data_splits/train_validate_data_experiment_2.pt',
                           test_data_path = 'data_splits/test_data_experiment_2.pt',
                           samples_per_class = 20, n_epochs = 3)
        #test(test_batch_size = 16, path_to_experiment = os.path.join('runs', '19_02_2021_14:27:13_SYS=Schlepptop_BS=16_LR=0.01_WD=0_EPOCHS=3_transformations_like_literature_top_5_categories_SPC=20_PRETRAINED_FINETUNING_LAST_DEBUGGING'))
    else:
        fire.Fire({'train+validate': train_and_validate, 'test': test})
