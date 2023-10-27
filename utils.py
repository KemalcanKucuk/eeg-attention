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
    # temporary sanity check, will delete
    # print(selected_features.head())
    return selected_features

def data_preparation(selected_channels, selected_labels, feature_subset, split_size = 0.2, seed = 1):

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = feature_selection(selected_channels=selected_channels, feature_subset = feature_subset) # select every feature
    y = selected_labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_size, random_state = seed)

    # apply normalization after splitting to avoid leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return [X_train, X_test, y_train, y_test]

def model_training(data, model_family, verbose = True, stats=False, cm=False):

  X_train, X_test, y_train, y_test = data
  # TODO: burada bi seyleri karistirdim bu kisim bi kontrol edilsin
  #display_labels = ['drowsy' if label == 1 else 'alert' for label in labels['label'].unique()]
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
    model = SVC(C=1.0, kernel='rbf', degree=10, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=1)
  elif model_family == 'NN':
    model = MLPClassifier(activation='relu',solver='adam', alpha=1e-2, learning_rate='adaptive', max_iter=1000000, hidden_layer_sizes=(60,2), random_state=1)
  elif model_family == 'GBC':
    model = GradientBoostingClassifier(loss='log_loss',n_estimators=300, learning_rate=0.1, max_depth=10, random_state=1)

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
    model_disp = ConfusionMatrixDisplay(confusion_matrix=model_cm,display_labels=display_labels)
    model_disp.plot()
  
  return [training_acc, test_acc]

def downsampling(df, sr=0.5):
    '''
    Returns the downsampled version of a dataframe, conserving the class ratios.
    '''
    ds_df = pd.DataFrame()
    row_count = len(df.index)
    ds_sample_n = row_count * sr
    s0 = df.label[df.label.eq(0)].sample(int(ds_sample_n/2)).index
    s1 = df.label[df.label.eq(1)].sample(int(ds_sample_n/2)).index 
    ds_df = df.loc[s0.union(s1)]
    return ds_df

def feature_combination(feature_subset, selected_channels, selected_labels, stop_feature, min_n = 1, max_n = 5, training=False, pvalue=False):
    '''
    Go through a feature subset and calculate the combinations of different features on the subsets.
    '''
    filename = 'until' + stop_feature + '_' + str(min_n) + 'to' + str(max_n) + '.txt'
    file = open(filename, 'w')
    # selected channels and labels are global now, might fix it if we decide settling on this method.
    for i in range(min_n, max_n):
        for comb in list(itertools.combinations(feature_subset, i)):
            if training:
                data = data_preparation(selected_channels=selected_channels, selected_labels=selected_labels, feature_subset=comb)
                # parametrize the models
                for model in ['K-NN', 'SVM', 'DTC']:
                    train_acc, test_acc = model_training(data, model, verbose=False, stats=False, cm=False)
                    # TODO: improve the readability of the file by either modifying the string or changing the filetype entirely
                    file.writelines(f"{comb}: {model} train acc: {train_acc:.2f} test acc: {test_acc:.2f}\n")
            # we can also add a thresholding section to see if the pvalue or variance value changes
            if pvalue:
                pass
    file.close()

# P-Value Thresholding for Feature Selection
def p_value_thresholding(selected_features, selected_labels):

    p_values = []

    X_p = selected_features
    
    y_p = selected_labels.flatten()
    y_p = pd.Series(y_p)


    #y_p = pd.Series(y['0'])
    sorted_dict = {}
    for feature in X_p.columns:
        t_stat, p_value = stats.ttest_ind(X_p[feature][y_p == 0], X_p[feature][y_p == 1])
        p_values.append(p_value)
        sorted_dict[feature] = p_value


    alpha = 0.05

    # Select features with p-values below the significance level
    selected_features = [X_p.columns[i] for i, p in enumerate(p_values) if p < alpha]

    # Alternatively, you can rank features by p-value
    sorted_features = [x for _, x in sorted(zip(p_values, X_p.columns))]
    from collections import OrderedDict

    ordered = OrderedDict(sorted(sorted_dict.items(), key=lambda item:np.max(item[1])))
    for key, value in ordered.items():
        print(key, value)
    print(ordered)
        
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