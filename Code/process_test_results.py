import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

path_to_final_results = '/home/lena/Nextcloud-Butterbread/Documents/Master_Data_Science/5_Semester/Research_Project/final_results/'

# %% SPECIFY FILE PATHS TO ALL EXPERIMENTS

def load_test_json(file_path):
    absolute_path = os.path.join(file_path, 'results_test_dataset.json')
    with open(absolute_path, 'r') as f:
        dictidict = json.load(f)
    # df = pd.DataFrame.from_dict(dictidict, orient = 'columns')
    # type(df.mean())

    return dictidict


experiment_1_balanced_path = '/home/lena/Nextcloud-Butterbread/Documents/Master_Data_Science/5_Semester/Research_Project/final_results/experiment_1/balanced_19_02_2021_19:51:31_SYS=ml3-gpu2_BS=128_LR=0.01_WD=0_EPOCHS=50_transformations_like_literature_top_5_categories_SPC=5300_PRETRAINED_FINETUNING_LAST/'
experiment_1_imbalanced_path = '/home/lena/Nextcloud-Butterbread/Documents/Master_Data_Science/5_Semester/Research_Project/final_results/experiment_1/imbalanced_19_02_2021_19:56:12_SYS=ml3-gpu2_BS=128_LR=0.01_WD=0_EPOCHS=50_transformations_like_literature_top_5_categories_SPC=all_PRETRAINED_FINETUNING_LAST/'

experiment_2_feature_extractor_path = '/home/lena/Nextcloud-Butterbread/Documents/Master_Data_Science/5_Semester/Research_Project/final_results/experiment_2/feature_extractor_24_02_2021_18:56:19_SYS=ml3-gpu2_BS=128_LR=0.01_WD=0_EPOCHS=50_transformations_like_literature_top_5_categories_SPC=1000_PRETRAINED_FINETUNING_LAST/'
experiment_2_from_scratch_path = '/home/lena/Nextcloud-Butterbread/Documents/Master_Data_Science/5_Semester/Research_Project/final_results/experiment_2/from_scratch_22_02_2021_12:10:07_SYS=ml3-gpu2_BS=128_LR=0.01_WD=0_EPOCHS=50_transformations_like_literature_top_5_categories_SPC=1000_FINETUNING_ALL/'

experiment_3_ten_classes_path = '/home/lena/Nextcloud-Butterbread/Documents/Master_Data_Science/5_Semester/Research_Project/final_results/experiment_3/22_02_2021_19:29:58_SYS=ml3-gpu2_BS=128_LR=0.01_WD=0_EPOCHS=50_transformations_like_literature_top_10_categories_SPC=1000_PRETRAINED_FINETUNING_LAST/'

# load all test results from JSON and add experiment name to dict
experiment_1_balanced = load_test_json(experiment_1_balanced_path)
experiment_1_balanced['experiment_name'] = 'experiment_1_balanced'

experiment_1_imbalanced = load_test_json(experiment_1_imbalanced_path)
experiment_1_imbalanced['experiment_name'] = 'experiment_1_imbalanced'

experiment_2_feature = load_test_json(experiment_2_feature_extractor_path)
experiment_2_feature['experiment_name'] = 'experiment_2_feature'

experiment_2_from_scratch = load_test_json(experiment_2_from_scratch_path)
experiment_2_from_scratch['experiment_name'] = 'experiment_2_from_scratch'

experiment_3_ten_classes = load_test_json(experiment_3_ten_classes_path)
experiment_3_ten_classes['experiment_name'] = 'experiment_3_ten_classes'

# %% create one big object from the results

# create a large dict of the smaller ones
list_of_dicts = [experiment_1_balanced, experiment_1_imbalanced, experiment_2_feature,
                 experiment_2_from_scratch, experiment_3_ten_classes]

d = {}
for k in experiment_1_imbalanced.keys():
    d[k] = tuple(d[k] for d in list_of_dicts)

# make a dataframe from the dict
# problem: How to treat the elements that are a list with multiple values?
test_dataframe = pd.DataFrame.from_dict(d).drop(['confusion_matrix', 'support'], axis = 1)

# calculate mean values out of each list
# select columns that are list entries

# works for replacing list with its mean
for column in test_dataframe.columns[3:8]:
    for i in test_dataframe[column].iteritems():
        #test_dataframe[column].iloc[i[0]] = np.mean(i[1])  # throws Settingwithcopwarning
        test_dataframe.loc[i[0], column] = np.mean(i[1]).astype('float64')
    # cast column to float, otherwise it will not be rounded!
    test_dataframe[column] = test_dataframe[column].astype('float64')

# calculate percentage = multiply by 100 (except the rates)
test_dataframe = test_dataframe[0:5].apply(lambda x: x*100 if x.name in test_dataframe.columns[0:3] else x,
                                 axis = 0)

# round all values
test_dataframe = test_dataframe.round(2)

# write to csv on disk
test_dataframe.to_csv(path_or_buf = path_to_final_results + 'all_test_results_class_specific.csv', index = False)

# %% plot all confusion matrices

# source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea

NORMALIZE = False
SAVE = True


def plot_save_confusion_matrix(confusion_matrix, list_of_category_names, figure_name, show = True,
                               save = False, normalized = False, show_colorbar = True):
    """

    Parameters
    ----------
    confusion_matrix: array or list containing the confusion matrix
    list_of_category_names: list of strings, if the categories should not be displayed as numbers
    figure_name: string specifying the file name
    """
    # fontdict = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 16, }
    if normalized:
        # confusion_matrix = np.round(confusion_matrix/np.sum(confusion_matrix), 2)
        # with np.nditer(confusion_matrix, op_flags = ['readwrite']) as iter:
        #     for element in iter:
        #         #print(element)
        #         if element == 0:
        #             element[...] = np.round(element, 0)
        # TODO make 0.00 entries to 0 for a better overview!
        plot = sns.heatmap(confusion_matrix / np.sum(confusion_matrix), annot = True,
                           cmap = 'Blues', cbar = show_colorbar, fmt = '.2f', vmin = 0, vmax = 1,
                           xticklabels = list_of_category_names,
                           yticklabels = list_of_category_names)
    else:
        plot = sns.heatmap(confusion_matrix, annot = True, cmap = 'Blues', fmt = 'd',
                           cbar = show_colorbar,
                           xticklabels = list_of_category_names,
                           yticklabels = list_of_category_names)
    plt.xlabel('Predicted Class', labelpad = 10, fontweight = 'bold')
    plt.ylabel('Actual Class', labelpad = 10, fontweight = 'bold')
    plt.tight_layout()  # make sure that no labels are cut off
    if show:
        plt.show()
    if save:
        figure_object = plot.get_figure()
        figure_object.savefig(os.path.join('/home/lena/git/research_project/results_plotted/final',
                                           figure_name + '.png'), dpi = 600)
    return plot


top_5_categories = ['empty', 'rabbit', 'petrel', 'iguana', 'rat']
top_10_categories = ['empty', 'rabbit', 'petrel', 'iguana', 'rat', 'cat', 'pig', 'goat',
                     'shearwater', 'petrel chick']

plot_save_confusion_matrix(experiment_1_balanced.get('confusion_matrix'),
                           list_of_category_names = top_5_categories,
                           figure_name = 'experiment_1_balanced_conf_matrix', save = SAVE,
                           normalized = NORMALIZE, show_colorbar = True)

plot_save_confusion_matrix(experiment_1_imbalanced.get('confusion_matrix'),
                           list_of_category_names = top_5_categories,
                           figure_name = 'experiment_1_imbalanced_conf_matrix', save = SAVE,
                           normalized = NORMALIZE, show_colorbar = True)

plot_save_confusion_matrix(experiment_2_feature.get('confusion_matrix'),
                           list_of_category_names = top_5_categories,
                           figure_name = 'experiment_2_feature_conf_matrix', save = SAVE,
                           normalized = NORMALIZE, show_colorbar = True)

plot_save_confusion_matrix(experiment_2_from_scratch.get('confusion_matrix'),
                           list_of_category_names = top_5_categories,
                           figure_name = 'experiment_2_from_scratch_conf_matrix', save = SAVE,
                           normalized = NORMALIZE, show_colorbar = True)

plot_save_confusion_matrix(experiment_3_ten_classes.get('confusion_matrix'),
                           list_of_category_names = top_10_categories,
                           figure_name = 'experiment_3_ten_classes_conf_matrix', save = SAVE,
                           normalized = NORMALIZE)
