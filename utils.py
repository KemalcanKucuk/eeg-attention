import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn as sk
import xgboost as xgb
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import itertools

from training import *
from selection import *

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def data_loader(path, ds=False, ds_rate=1):
    '''
    Load the data and apply the necessary transformations, downsample if necessary.

    Arguments:
    path -- Path to the dataset .csv
    ds -- Downsampling boolean
    ds_rate -- Rate of the downsampling

    return -- Dataset DataFrame
    '''
    # load the feature dataset as a dataframe
    df = pd.read_csv(path, float_precision='round_trip')

    if ds:
        df = downsampling(df, sr=ds_rate)

    df = df.drop('Unnamed: 0', axis=1)

    return df


def channel_selection(dataset_df, channel_list):
    '''
    Select the desired channels from the total feature dataset

    Arguments:
    dataset_df -- Output of the data_loader()
    channel_list -- List of channels to be extracted from the dataset

    return -- The reduced dataset with selected channels
    '''
    
    # split the dataset to features and labels
    features = dataset_df.drop('label', axis=1)
    labels = dataset_df.iloc[:, -1:]

    selected_channels = []
    for channel in channel_list:
        selected_channels.append(features.loc[features['channels'] == channel])
    # return the corresponding labels for the selected channels
    selected_labels = labels[0:2022*len(channel_list)].to_numpy()
    result_df = pd.concat(selected_channels).drop('channels', axis=1)
    result_df['label'] = selected_labels
    return result_df


def feature_selection(dataset, feature_subset):
    ''' 
    Select the desired subset of features to prepare training data on.

    Arguments:
    dataset -- Reduced dataset with selected channels, output of the channel_selection
    feature_subset -- List of features to be extracted from the dataset 

    return -- The reduced dataset with selected features (no labels)
    '''
    selected_features = pd.DataFrame()
    for feature in feature_subset:
        selected_features[feature] = dataset[feature]
    return selected_features


def incremental_training(dataset, channel_list, feature_subset, models, mode='feature', figure=False, save=False):
    '''
    Incrementally train channels or features to see individual performances
    '''

    if mode == 'feature':
        iterable = feature_subset
        filename = 'outs/feat_inc.csv'
        figname = 'outs/feat_fig.png'
    elif mode == 'channel':
        iterable = channel_list
        filename = 'outs/chnl_inc.csv'
        figname = 'outs/chnl_fig.png'
    
    incrementation = []
    results = {}
    for inc_var in tqdm(iterable):
        incrementation.append(inc_var)
        if mode == 'channel':
            reduced_dataset = channel_selection(dataset, incrementation)
            data = data_preparation(dataset=reduced_dataset,  feature_subset=feature_subset)
        else:
            reduced_dataset = channel_selection(dataset, channel_list)
            data = data_preparation(dataset=reduced_dataset,  feature_subset=incrementation)
        model_acc = {}
        for model in models:
            model_result = model_training(data, model, stats=False, cm=False)
            model_acc[model] = model_result['test_acc']
            # for better readiability of the csv
            # results[str(len(incrementation))] = model_acc
            results[str(incrementation)] = model_acc
        # plateau part comes here if its implemented
    results_df = pd.concat({
                        k: pd.DataFrame.from_dict(v, 'index') for k, v in results.items()
                    }, 
                    axis=0)
    results_df.columns = ['test_accuracy']
    if figure:
        import matplotlib.pyplot as plt
        # this model accuracy dictionary can be extended for further applications
        model_accs = {model: [] for model in models}
        for i in range(len(results_df)):
            model_accs[models[i%len(models)]].append(results_df.iloc[i].iloc[0])
        # take an arbitrary accuracy list for the x-axis
        for key in model_accs:
            x = np.linspace(1,len(model_accs[key]), len(model_accs[key]))
            plt.plot(x, model_accs[key])
        plt.grid()
        plt.legend(models)
        if save:
            plt.savefig(figname, bbox_inches='tight', dpi=300)
            # also save to a csv for plotting somewhere else
            
        plt.show()
    if save:
        results_df.to_csv(filename)

    return results_df

def downsampling(df, sr=0.5):
    '''
    Returns the downsampled version of a dataframe, conserving the class ratios.
    '''
    ds_df = pd.DataFrame()
    row_count = len(df.index)
    ds_sample_n = row_count * sr
    s0 = df.label[df.label.eq(0)].sample(
        int(ds_sample_n/2), random_state=447).index
    s1 = df.label[df.label.eq(1)].sample(
        int(ds_sample_n/2), random_state=447).index
    ds_df = df.loc[s0.union(s1)]
    return ds_df