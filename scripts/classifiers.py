import pandas as pd
import numpy as np
import sklearn

from numpy import mean
from numpy import std

from sklearn.ensemble import  RandomForestClassifier
from sklearn import svm


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb


import shap
import matplotlib.pyplot as plt


from sklearn.metrics import plot_roc_curve
import re




def read_in_data(cols_to_drop=[]):
    df = pd.read_csv('../data/JJ_JK_8_26_21_LC_MECFS.csv')
    prots = df['Protein']
    df = df.drop(['Protein'], axis=1)
    df = df.T
    df.columns = prots
    df = df.drop(columns=cols_to_drop)
    return df

def label_data(df):
    patients =  df.index.tolist()
    status_list = []
    for patient in patients:
        patient_status = ''
        if 'HCW' in patient or 'Con' in patient:
            patient_status = 'H'
        if 'LC.MS' in patient:
            patient_status = 'C'
        if 'MECFS' in patient:
            patient_status = 'M'
        status_list.append(patient_status)

    mlb = MultiLabelBinarizer()
    one_hot = mlb.fit_transform(status_list)

    one_hot_df = pd.DataFrame(one_hot, columns=mlb.classes_, index=df.index)

    return status_list, one_hot_df


def datasplit(df, labels):
   return train_test_split(df, labels, test_size=0.20, random_state=42)

def add_pseudo_count(x):
    min_x = min(x[x > 0])
    x = [y + min_x/2 if y == 0 else y for y in x]
    return(x)


def raw(df):
    return df

def log_transform(df):
    tokeep = df.columns[df.any()]
    df = df[tokeep]
    colnames = df.columns
    df = df.apply(add_pseudo_count)
    df = df.apply(np.log10)

    df = pd.DataFrame(df, columns = colnames)
    return df


def sqrt_transform(df):
    colnames = df.columns
    df = df.pow(1./2)
    df = pd.DataFrame(df, columns = colnames)
    return df

def zscore(df):
    scaler = StandardScaler()

    colnames = df.columns
    df = scaler.fit_transform(df)

    df = pd.DataFrame(df, columns = colnames)
    return df

def lib_size_norm(df):
    tokeep = df.columns[df.any()]
    df = df[tokeep]
    colnames = df.columns
    df = df.div(df.sum(axis=0), axis = 1)
    df = pd.DataFrame(df, columns = colnames)
    return df

def zscore_log(df):
    df = log_transform(df)
    df = zscore(df)
    return df

def lib_size_norm_log(df):
    df = lib_size_norm(df)
    df = log_transform(df)
    return df


def normalize_data(df, norm_funcs):
    norms = {'log_transform':log_transform, 'zscore': zscore,
             'lib_size_norm': lib_size_norm, 'zscore_log': zscore_log,
             'lib_size_norm_log': lib_size_norm_log, 'sqrt': sqrt_transform,
             'raw': raw}

    normed_df_dict = {}
    for func in norm_funcs:
        norm_func = norms[func]
        normed_df = norm_func(df)
        normed_df_dict[func] = normed_df
    return normed_df_dict


def rand_for(X, clss, cv):
    clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=42)
    return clf

def support_vm(X, y, cv):
    clf = svm.SVC(class_weight='balanced',probability=True)
    return clf

def xg_boost(X, y, cv):
    clf = xgb.XGBClassifier()
    return clf

def get_best_model(model, X, y, model_type='rf'):
    rf_params = {'class_weight':['balanced'], 'n_estimators':[5, 10, 25, 50, 100],
                 'min_impurity_decrease': [0, .1, .5], 'criterion': ['gini', 'entropy']}

    svm_params = {'C':[0.25, 0.5, 1, 2], 'kernel':['linear', 'rbf', 'poly', 'sigmoid'],
                  'degree':[2, 3, 4], 'class_weight':['balanced'],
                  'probability':[1]}

    xgb_params = {'n_estimators':[1, 2, 4, 8], 'booster':['gbtree', 'gblinear', 'dart'],
                  'reg_alpha':[0.001, 0.01, 0.1, .5, 1], 'reg_lambda':[0.001, 0.01, 0.1, .5 ,1]
    }

    model_case = {'rf': RandomizedSearchCV(model, param_distributions=rf_params, scoring='accuracy', cv=10,
                                            n_iter=10, refit=True, n_jobs=-1),
                  'svm': RandomizedSearchCV(model, param_distributions=svm_params, scoring='accuracy', cv=10,
                                            n_iter=10, refit=True, n_jobs=-1),
                  'xgb': RandomizedSearchCV(model, param_distributions=xgb_params, scoring='accuracy', cv=10,
                                            n_iter=10, refit=True, n_jobs=-1)
                   }

    search = model_case[model_type]
    search.fit(X, y)
    model = search.best_estimator_
    return(model)


def shapley_analysis(X, model, model_type, norm):
    if model_type == 'rf':
        explainer = shap.TreeExplainer(model, feature_names=X.columns.tolist())
    else:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 20))

    shap_values = explainer.shap_values(X, approximate=False, check_additivity=False)

    # this should be 0 for Long Covid
    # should be 1 for non Long Covid
    shap_df = pd.DataFrame(shap_values[1])
    shap_df.columns = X.columns
    shap_df.index = X.index
    shap_df.to_csv("../results/shap_res_long_covid_%s_normed.csv" % norm)

    mean_importance = np.absolute(shap_values[0]).mean(axis=0)
    mean_importance = np.absolute(mean_importance)
    feat_import = pd.DataFrame({'feature':X.columns, 'mean_importance':mean_importance}).sort_values('mean_importance', ascending=False)
    feat_import.to_csv('../results/shap_feat_import_list_%s_normed.csv' % norm)

def eval_model(model, X, y, cv, model_type='rf', norm=''):
    #print(y.sum())
    #print(len(y))
    y_pred = cross_val_predict(model, X, y, cv=cv)
    model = get_best_model(model, X, y, model_type)

    #y_test = np.vectorize(factor_dict.get)(y)
    #y_pred = np.vectorize(factor_dict.get)(y_pred)
    
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    scores_sens = cross_val_score(model, X, y, scoring='recall', cv=cv, n_jobs=-1)
    scores_spec = cross_val_score(model, X, y, scoring='precision', cv=cv, n_jobs=-1)
    acc = '%.3f.(%.3f)' % (mean(scores), std(scores))
    sens = '%.3f.(%.3f)' % (mean(scores_sens), std(scores_sens))
    spec = '%.3f.(%.3f)' % (mean(scores_spec), std(scores_spec))
    print('Accuracy: %.3f (%.3f), Sensitivity: %.3f (%.3f), Specificity: %.3f (%.3f)' %
          (
              mean(scores), std(scores),
              mean(scores_sens), std(scores_sens),
              mean(scores_spec), std(scores_spec)
          )
          )
    #shapley_analysis(X, model, model_type, norm)

def run_models(normed_dfs, labs, classes, model_type='rf'):
    cv = KFold(n_splits=10, random_state=43, shuffle=True)

    for norm in normed_dfs:
        print('Normalized with: ' + norm)
        X = normed_dfs[norm]
        for clss in classes:
            rf = rand_for(X, labs[clss], cv)
            svm = support_vm(X, labs[clss], cv)
            eval_model(svm, X, labs[clss], cv, model_type='svm', norm=norm)
    

def filter_classes(df, labels_onehot, classes_to_drop=[]):
    for clss in classes_to_drop:
        filt = labels_onehot[clss] != 1
        df = df[filt]
        labels_onehot = labels_onehot[filt]
    return df, labels_onehot



def main():
    df = read_in_data(cols_to_drop=['COV2-RBD', 'SARS-RBD'])
    labels, labels_onehot = label_data(df)
    df, labels_onehot = filter_classes(df, labels_onehot, 'M')
    #normed_dfs = normalize_data(df, ['log_transform', 'zscore', 'lib_size_norm', 'lib_size_norm_log', 'sqrt', 'zscore_log'])
    #normed_dfs = normalize_data(df, ['log_transform', 'zscore', 'lib_size_norm', 'sqrt', 'raw'])
    # These normalization schemes work well for SVM models
    normed_dfs = normalize_data(df, ['sqrt'])
    run_models(normed_dfs, labels_onehot, 'C', model_type='xgb')



main()
