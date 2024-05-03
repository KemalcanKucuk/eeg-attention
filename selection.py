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
from tqdm import tqdm
import utils


warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def feature_combination(feature_subset, dataset, models, training=False, min_n=1, max_n=5):
    '''
    Go through a feature subset and calculate the combinations of different features on the subsets.
    '''
    result_df = pd.DataFrame(
        columns=['model', 'train_acc', 'test_acc', 'combination'])
    for i in range(min_n, max_n):
        for comb in list(itertools.combinations(feature_subset, i)):
            if training:
                data = data_preparation(
                    dataset=dataset, feature_subset=comb)
                for model in models:
                    train_acc, test_acc = model_training(
                        data, model, verbose=False, stats=False, cm=False)
                    result_df.loc[len(result_df)] = {
                        'model': model, 'train_acc': train_acc, 'test_acc': test_acc, 'combination': comb}
    return result_df


def combination_evaluation(result_df, target, filename, acc_threshold=0.75, write=True):
    '''
    Return and display the desired performance results of the combination process.
    '''
    finds = ((result_df.where((result_df[target] >= acc_threshold))).dropna(
    )).sort_values(target, ascending=False)
    if write:
        result_df.to_csv('outs/' + filename)
        finds.to_csv('outs/finds_' + filename)

# P-Value Thresholding for Feature Selection


def p_value_thresholding(dataset, feature_subset, verbose=False):

    # TODO: buranin dogrulugundan su an emin degilim ama duzeltmesi cok zor olmamali
    p_values = []

    selected_features = utils.feature_selection(dataset=dataset, feature_subset=feature_subset) # select every feature
    selected_labels = dataset['label']

    X_p = selected_features
    y_p = selected_labels

    #y_p = selected_labels.flatten()
    #y_p = pd.Series(y_p)

    # y_p = pd.Series(y['0'])
    sorted_dict = {}
    for feature in X_p.columns:
        t_stat, p_value = stats.ttest_ind(X_p[feature][y_p == 0], X_p[feature][y_p == 1])
        p_values.append(p_value)
        sorted_dict[feature] = p_value

    alpha = 0.05

    # Select features with p-values below the significance level
    selected_features = [X_p.columns[i]
                         for i, p in enumerate(p_values) if p < alpha]

    # Alternatively, you can rank features by p-value
    sorted_features = [x for _, x in sorted(zip(p_values, X_p.columns))]
    from collections import OrderedDict

    ordered = OrderedDict(
        sorted(sorted_dict.items(), key=lambda item: np.max(item[1])))
    if verbose:
        for key, value in ordered.items():
            print(key, value)

    return sorted_features, sorted_dict


def p_value_slicing(p_values, stop_feature):
    '''
    Return the highest ranking features of the p_value list until the stop_feature.
    '''

    stop_index = 0
    for i in range(len(p_values)):
        if p_values[i] == stop_feature:
            stop_index = i

    return p_values[:stop_index]


def variance_thresholding(df, threshold):
    variance_df = pd.DataFrame()
    variance_dict = {}
    df = (df - df.min()) / (df.max()-df.min())
    for col in df.columns:
        variance_dict[col] = np.var(df[col])
        if np.var(df[col]) >= threshold:
            variance_df[col] = df[col]
    return variance_df, variance_dict