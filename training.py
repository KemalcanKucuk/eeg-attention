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

import utils
import selection

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def data_preparation(dataset, feature_subset, split_size=0.2, seed=447):

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = utils.feature_selection(dataset=dataset,
                          feature_subset=feature_subset)

    labels = dataset.iloc[:, -1:]
    y = labels

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
