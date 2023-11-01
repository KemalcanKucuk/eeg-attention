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

warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def data_loader(path, channel_list, ds=False, ds_rate=1):
    '''
    Load the data and apply the necessary transformations, downsample if necessary.
    '''
    # TODO: remove the downsampling part

    # load the feature dataset as a dataframe
    df = pd.read_csv(path, float_precision='round_trip')

    if ds:
        df = downsampling(df, sr=ds_rate)

    # we might delete this condition but i'll leave it be
    # in case we try it on multiple datasets
    # if csv_path == 'eeg_features.csv':

    df = df.drop('Unnamed: 0', axis=1)
    # split the dataset to features and labels
    features = df.drop('label', axis=1)
    labels = df.iloc[:, -1:]

    selected_channels, selected_labels = channel_selection(
        features=features, labels=labels, channel_list=channel_list)
    # TODO: might add a test argument to directly return (features,labels)

    return (selected_channels, selected_labels)


def channel_selection(features, labels, channel_list):
    '''
    Select the desired channels from the total feature dataset
    '''
    selected_channels = []
    for channel in channel_list:
        selected_channels.append(features.loc[features['channels'] == channel])
    # return the corresponding labels for the selected channels
    selected_labels = labels[0:2022*len(channel_list)].to_numpy()
    return ((pd.concat(selected_channels).drop('channels', axis=1)), selected_labels)


def feature_selection(selected_channels, feature_subset):
    ''' 
    Select the desired subset of features to prepare training data on.
    '''
    selected_features = pd.DataFrame()
    for feature in feature_subset:
        selected_features[feature] = selected_channels[feature]
    return selected_features


def data_preparation(selected_channels, selected_labels, feature_subset, split_size=0.2, seed=1):

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = feature_selection(selected_channels=selected_channels,
                          feature_subset=feature_subset)
    y = selected_labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size, random_state=seed)

    # apply normalization after splitting to avoid leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return [X_train, X_test, y_train, y_test]


def model_training(data, model_family, verbose=True, stats=False, cm=False):

    X_train, X_test, y_train, y_test = data
    # TODO: burada bi seyleri karistirdim bu kisim bi kontrol edilsin
    # display_labels = ['drowsy' if label == 1 else 'alert' for label in labels['label'].unique()]
    display_labels = ['drowsy', 'alert']
    if model_family == 'K-NN':
        model = KNeighborsClassifier()
    elif model_family == 'DTC':
        model = DecisionTreeClassifier()
    elif model_family == 'RFC':
        model = RandomForestClassifier(n_estimators=100)
    elif model_family == 'Logistic Regression':
        model = LogisticRegression(max_iter=5000)
    elif model_family == 'SVM':
        model = SVC(C=1.0, kernel='rbf', degree=10, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001,
                    cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=1)
    elif model_family == 'NN':
        model = MLPClassifier(activation='relu', solver='adam', alpha=1e-2, learning_rate='adaptive',
                              max_iter=1000000, hidden_layer_sizes=(60, 2), random_state=1)
    elif model_family == 'GBC':
        model = GradientBoostingClassifier(
            loss='log_loss', n_estimators=300, learning_rate=0.1, max_depth=10, random_state=1)

    model.fit(X_train, y_train)
    training_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    if verbose:
        print('Accuracy of {} classifier on training set: {:.8f}'
              .format(model_family, training_acc))
        print('Accuracy of {} classifier on test set: {:.8f}'
              .format(model_family, test_acc))

    if stats:
        print()
        print("==== Stats for the {} model ====".format(model_family))
        sensitivity = recall_score(y_test, model.predict(X_test))
        print("Sensitivity (Recall):", sensitivity)

        precision = precision_score(y_test, model.predict(X_test))
        print("Precision:", precision)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        print("Accuracy (Recall):", accuracy)

        f1 = f1_score(y_test, model.predict(X_test))
        print("F1_score:", f1)

        fpr, tpr, thresholds = roc_curve(y_test, model.predict(X_test))
        auc = roc_auc_score(y_test, model.predict(X_test))
        print("AUC:", auc)

        logloss = log_loss(y_test, model.predict(X_test))
        print("Logloss:", logloss)
        print()

    if cm:
        model_cm = confusion_matrix(y_test, model.predict(X_test))
        model_disp = ConfusionMatrixDisplay(
            confusion_matrix=model_cm, display_labels=display_labels)
        model_disp.plot()

    return [training_acc, test_acc]


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


def feature_combination(feature_subset, selected_channels, selected_labels, models=['K-NN', 'SVM', 'DTC'], training=False, min_n=1, max_n=5):
    '''
    Go through a feature subset and calculate the combinations of different features on the subsets.
    '''
    result_df = pd.DataFrame(
        columns=['model', 'train_acc', 'test_acc', 'combination'])
    for i in range(min_n, max_n):
        for comb in list(itertools.combinations(feature_subset, i)):
            if training:
                data = data_preparation(
                    selected_channels=selected_channels, selected_labels=selected_labels, feature_subset=comb)
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


def p_value_thresholding(selected_features, selected_labels, verbose=False):

    # TODO: fix the formatting of this
    p_values = []

    X_p = selected_features

    y_p = selected_labels.flatten()
    y_p = pd.Series(y_p)

    # y_p = pd.Series(y['0'])
    sorted_dict = {}
    for feature in X_p.columns:
        t_stat, p_value = stats.ttest_ind(
            X_p[feature][y_p == 0], X_p[feature][y_p == 1])
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

    '''
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

#y = y.reset_index(drop=True)

pca = PCA(n_components = 0.999)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
#X = dataPCA
variance = pd.DataFrame(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))
'''
