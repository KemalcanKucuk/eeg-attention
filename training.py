import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn as sk
import xgboost as xgb
from scipy import stats
from scipy.stats import ttest_ind
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

import utils
import selection

warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def data_preparation(dataset, feature_subset, split_size=0.2, pca=False, seed=447):
    '''
    TODO: 
    Select the desired channels from the total feature dataset

    Arguments:
    dataset_df -- Output of the data_loader()
    channel_list -- List of channels to be extracted from the dataset

    return -- The reduced dataset with selected channels
    '''
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = utils.feature_selection(dataset=dataset,
                                feature_subset=feature_subset)

    y = dataset.iloc[:, -1:]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size, random_state=seed)

    # apply normalization after splitting to avoid leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if pca:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        pca = PCA(n_components = 0.999)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    return [X_train, X_test, y_train, y_test]

'''
def model_switch(model_family):
    

    Select the desired channels from the total feature dataset

    Arguments:
    dataset_df -- Output of the data_loader()
    channel_list -- List of channels to be extracted from the dataset

    return -- The reduced dataset with selected channels
    model = 
    if model_family == 'K-NN':
        model = KNeighborsClassifier(leaf_size= 10, n_neighbors= int(np.sqrt(np.prod(data[0].shape))), p= 1)
    elif model_family == 'DTC':
        model = DecisionTreeClassifier(max_depth=7)
    elif model_family == 'RFC':
        model = RandomForestClassifier(n_estimators=100)
    elif model_family == 'LR':
        model = LogisticRegression(max_iter=5000)
    elif model_family == 'SVM':
        model = SVC(C=10.0, kernel='rbf', gamma=0.1, random_state=1)
        #model = SVC(C=1.0, kernel='rbf', degree=10, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=1)
    elif model_family == 'NN':
        model = MLPClassifier(activation='relu', solver='adam', alpha=1e-2, learning_rate='adaptive',
                              max_iter=1000000, hidden_layer_sizes=(60, 2), random_state=1)
    elif model_family == 'XGB':
        model = GradientBoostingClassifier(
            loss='log_loss', n_estimators=300, learning_rate=0.1, max_depth=10, random_state=1)
    return model
'''

def model_training(data, model_family, stats=False, cm=False):
    '''
    TODO: 
    Select the desired channels from the total feature dataset

    Arguments:
    dataset_df -- Output of the data_loader()
    channel_list -- List of channels to be extracted from the dataset

    return -- The reduced dataset with selected channels
    '''
    X_train, X_test, y_train, y_test = data
    # display_labels = ['drowsy' if label == 1 else 'alert' for label in labels['label'].unique()]
    display_labels = ['drowsy', 'alert']
   #model = model_switch(model_family)
    if model_family == 'K-NN':
        model = KNeighborsClassifier(leaf_size= 10, n_neighbors= int(np.sqrt(np.prod(data[0].shape))), p= 1)
    elif model_family == 'DTC':
        model = DecisionTreeClassifier(max_depth=7)
    elif model_family == 'RFC':
        model = RandomForestClassifier(n_estimators=100)
    elif model_family == 'LR':
        model = LogisticRegression(max_iter=5000)
    elif model_family == 'SVM':
        model = SVC(C=10.0, kernel='rbf', gamma=0.1, random_state=1)
    elif model_family == 'SVM-def':
        model = SVC()
        #model = SVC(C=1.0, kernel='rbf', degree=10, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=1)
    elif model_family == 'NN':
        model = MLPClassifier(activation='relu', solver='adam', alpha=1e-2, learning_rate='adaptive',
                              max_iter=1000000, hidden_layer_sizes=(60, 2), random_state=1)
    elif model_family == 'NN-def':
        model = MLPClassifier()
    elif model_family == 'XGB':
        model = GradientBoostingClassifier(
            loss='log_loss', n_estimators=300, learning_rate=0.1, max_depth=10, random_state=1)
    model.fit(X_train, y_train)
    stats_dict = {}

    stats_dict['training_acc'] = model.score(X_train, y_train)
    stats_dict['test_acc'] = model.score(X_test, y_test)

    stats_dict['sensitivity'] = recall_score(y_test, model.predict(X_test))
    stats_dict['precision'] = precision_score(y_test, model.predict(X_test))
    stats_dict['f1'] = f1_score(y_test, model.predict(X_test))

    fpr, tpr, thresholds = roc_curve(y_test, model.predict(X_test))
    stats_dict['auc'] = roc_auc_score(y_test, model.predict(X_test))
    stats_dict['logloss'] = log_loss(y_test, model.predict(X_test))
   #stats_dict['predictions'] = model.predict(X_test)
    t_stat, p_value = ttest_ind(model.predict(X_test), y_test)
    stats_dict['t_score'] = t_stat
    stats_dict['p_value'] = p_value



    if stats:
        print()
        print("==== Stats_dict for the {} model ====".format(model_family))
        print('Training Accuracy: ', stats_dict['training_acc'])
        print('Test Accuracy: ', stats_dict['test_acc'])
        print("Sensitivity (Recall):", stats_dict['sensitivity'])
        print("Precision:", stats_dict['precision'])
        print("F1_score:", stats_dict['f1'])
        print("AUC:", stats_dict['auc'])    
        print("Logloss:", stats_dict['logloss'])
        print()

    if cm:
        model_cm = confusion_matrix(y_test, model.predict(X_test))
        model_disp = ConfusionMatrixDisplay(
            confusion_matrix=model_cm, display_labels=display_labels)
        model_disp.plot()
    

    return stats_dict

