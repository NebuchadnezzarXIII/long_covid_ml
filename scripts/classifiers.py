import pandas as pd
import numpy as np
import sklearn
import warnings
#warnings.filterwarnings('ignore')


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

from sklearn.metrics import plot_roc_curve
from sklearn.metrics import make_scorer

from sklearn import preprocessing

import datetime


import shap
import matplotlib.pyplot as plt


from sklearn.metrics import plot_roc_curve
import re
import copy




def read_in_data(cols_to_drop=[]):
    df = pd.read_csv('../data/JJ_JK_8_26_21_LC_MECFS.csv')
    prots = df['Protein']
    df = df.drop(['Protein'], axis=1)
    df = df.T
    df.columns = prots
    df = df.drop(columns=cols_to_drop)
    return df

def remove_cols_w_keywords(keywords, df):
    col_list = []
    df_cols = df.columns
    for k in keywords:
        cols = [x for x in df_cols if k in x]
        col_list += cols
    col_list = set(col_list)
    col_list = list(col_list)
    for col in col_list:
        df.drop(col, axis=1, inplace=True)
    return df

    

def read_in_meta():
    meta = pd.read_excel('../data/LC_MS_master.xlsx')

    meta_cleaned = remove_cols_w_keywords(
        ['REAP', 'ELISA', 'Cytokines',
         'nAb', 'Date'], meta)
    meta.set_index(['x0_LC_ID'], inplace=True)
    return meta



def label_data_disease_status(df):
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
    if y['lab'].nunique() > 2:
        clf = xgb.XGBClassifier(objective='multi:softmax', use_label_encoder=False)
    else:
        clf = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
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
                                            n_iter=11, refit=True, n_jobs=-1)
                   }
    try:
        search = model_case[model_type]
        search.fit(X, y)
        model = search.best_estimator_
        best_params = search.best_params_
    except:
        search = model_case['rf']
        search.fit(X, y)
        model = search.best_estimator_
    return(model, best_params)



def shapley_analysis(X, model, model_type, norm, feat):
    if model_type == 'rf' or model_type == 'XGB':
        explainer = shap.TreeExplainer(model, feature_names=X.columns.tolist())
    else:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 20))

    shap_values = explainer.shap_values(X, approximate=False, check_additivity=False)

    # this should be 0 for Long Covid
    # should be 1 for non Long Covid
    for index in range(len(shap_values)):
        shap_df = pd.DataFrame(shap_values[index])
        shap_df.columns = X.columns
        shap_df.index = X.index
        shap_df.to_csv("../results/shap_res_long_covid_%s_class_index_%s_%s_normed.csv" % (feat, index, norm))

        mean_importance = np.absolute(shap_values[index]).mean(axis=0)
        mean_importance = np.absolute(mean_importance)
        feat_import = pd.DataFrame({'feature':X.columns, 'mean_importance':mean_importance}).sort_values('mean_importance', ascending=False)
        feat_import.to_csv('../results/shap_feat_import_list_%s_clas_index_%s_%s_normed.csv' % (feat, index, norm))

def datasplit(df, labels):
   return train_test_split(df, labels, test_size=0.20, random_state=42)

def eval_model(model, X, y, cv, model_obj, model_type='rf', norm=''):
    #print(y.sum())
    #print(len(y))


    #X_train, X_test, y_train, y_test = datasplit(X, y)


    model, best_params = get_best_model(model, X, y, model_type)

    model_obj.best_model_params = best_params

    y_pred_all = model.predict(X)
    y_pred_10x_cv = model.predict(X)

    #ax = plt.gca()

    #rand =  np.random.randn(*X_test.shape)

    #plot_roc_curve(model, rand, y_test, ax=ax, alpha = 0.8)
    #plt.savefig('../figures/%s_%s_ROC_curve.pdf' % (model_type, norm))
    #y_test = np.vectorize(factor_dict.get)(y)
    #y_pred = np.vectorize(factor_dict.get)(y_pred)
    '''
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    scores_sens = cross_val_score(model, X, y, scoring='recall', cv=cv, n_jobs=-1)
    scores_spec = cross_val_score(model, X, y, scoring='precision', cv=cv, n_jobs=-1)

    cf_acc = (mean(scores), std(scores))
    cf_sens = (mean(scores_sens), std(scores_sens))
    cf_spec = (mean(scores_spec), std(scores_spec))
    '''

    model_obj.mean_acc_10x_cv = metrics.accuracy_score(y, y_pred_10x_cv)
    model_obj.mean_recall_10x_cv = metrics.precision_score(y, y_pred_10x_cv, average='macro')
    model_obj.mean_precision_10x_cv = metrics.recall_score(y, y_pred_10x_cv, average='macro')




    model_obj.full_acc = metrics.accuracy_score(y, y_pred_all)
    model_obj.full_prec =  metrics.precision_score(y, y_pred_all, average='macro')
    model_obj.full_recall = metrics.recall_score(y, y_pred_all, average='macro')

    shapley_analysis(X, model, model_type, norm, model_obj.meta_category)
    model_obj.write_self_to_file()
    return model_obj


def run_models(normed_dfs, labs, classes, model_obj,  model_type='rf', out_fp='../results/model_output.txt'):
    cv = KFold(n_splits=10, random_state=43, shuffle=True)
    res_str = 'Tested %s model\n' % model_type
    norm_mod_obj = copy.deepcopy(model_obj)
    for norm in normed_dfs:
        res_str += 'Normalized with: ' + norm + '\n'
        X = normed_dfs[norm]

        norm_mod_obj.norm_type = norm
        norm_mod_obj.model_type = model_type

        xgb_mod = xg_boost(X, labs, cv)
        try:
            metric_str = eval_model(xgb_mod, X, labs, cv, norm_mod_obj, model_type=model_type, norm=norm)
        except:
            print('some bad thing happened at some point')
        print('mutliclass worked')
        '''
        for clss in classes:
            rf = rand_for(X, labs[clss], cv)
            svm = support_vm(X, labs[clss], cv)
            xgb_mod = xg_boost(X, labs[clss], cv)
            class_str = 'classification_metrics_for_class_%s:\n' % clss
            res_str += class_str
            metric_str = eval_model(xgb_mod, X, labs[clss], cv, model_type=model_type, norm=norm)
            res_str += metric_str
        '''

def filter_classes(df, labels_onehot, classes_to_drop=[]):
    for clss in classes_to_drop:
        filt = labels_onehot[clss] != 1
        df = df[filt]
        labels_onehot = labels_onehot[filt]
    return df, labels_onehot


def label_data(df, label_col):



    df_no_nas = df[df[label_col].notna()]
    n_obs = len(df_no_nas)



    labs = df_no_nas[label_col]

    le = preprocessing.LabelEncoder()
    le.fit(labs)
    labs_encoded = le.transform(labs)
    labs_encoded = pd.DataFrame(labs_encoded, index=labs.index, columns=['lab'])

    one_hot_df = pd.DataFrame(pd.get_dummies(labs), index=df_no_nas.index)

    n_cats = len(labs.unique())
    is_continuous = False

    if n_cats * 2 > n_obs:
        is_continuous = True
        labs = df_no_nas[label_col]

    return labs, one_hot_df, df_no_nas, is_continuous, labs_encoded, le

class ModelResult:
    def __init__(self, fp):
        self.model_type=''
        self.best_model_params={}
        self.norm_type=''
        self.mean_acc_10x_cv=-1
        self.mean_recall_10x_cv=-1
        self.mean_precision_10x_cv=-1
        self.full_acc=-1
        self.full_prec=-1
        self.full_recall=-1
        self.meta_category=''
        self.naive_guessing_acc=-1
        self.fp=fp
    def write_header(self):

        header_str='''model_type\tnorm_type\tmeta_category\t\
        naive_guessing_accuracy\tmean_accuracy_10x_cv\t\
        mean_recall_10x_cv\tmean_precision_10x_cv\t\
        full_dataset_accuracy\tfull_dataset_precision\t\
        full_dataset_recall\tbest_model_params\n'''
        with open(self.fp, 'w') as outF:
            outF.write(header_str)

    def write_self_to_file(self, append=True):
        if append:
            with open(self.fp, 'a') as outF:
                outF.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %
                           (self.model_type, self.norm_type, self.meta_category,
                            self.naive_guessing_acc, self.mean_acc_10x_cv,
                            self.mean_recall_10x_cv, self.mean_precision_10x_cv,
                            self.full_acc, self.full_prec,
                            self.full_recall, self.best_model_params))




def main():
    #df = read_in_data(cols_to_drop=['COV2-RBD', 'SARS-RBD'])
    out_fp = '../results/symptom_model_results_obj.tsv'
    mod_1 = ModelResult(out_fp)
    mod_1.write_header()
    mod_1.write_self_to_file()
    print('finished')

    df = read_in_data()
    meta = read_in_meta()


    df_samps = df.index.tolist()
    meta_samps = meta.index.tolist()


    meta_cols = meta.columns.tolist()
    df_cols = df.columns.tolist()
    df_meta = df.join(meta, how='inner')

    baseline_mod = ModelResult(out_fp)

    baseline_mod.write_header()
    #meta_cols = ["x0_LC_Symptom_dysaut_total"]
    meta_cols = []
    with open ('../data/symptom_colss.txt', 'r') as cols_file:
        meta_cols += cols_file.readline().split('\t')

    
    all_models = []
    for cat in meta_cols:
        cat_mod = copy.deepcopy(baseline_mod)
        cat_mod.meta_category = cat
        labels, labels_onehot, df_na_filt, is_continuous, labs_encoded, le = label_data(df_meta, cat)
        df_filt = df_na_filt[df_cols]

        if is_continuous:
            pass
        else:
            print('categorical breakdown:')
            cat_breakdown = labels_onehot.sum()
            max_cat = cat_breakdown.max()
            tot_cat = cat_breakdown.sum()
            magic_number = max_cat/tot_cat
            cat_mod.naive_guessing_acc = magic_number

            df_normed = normalize_data(df_filt, ['sqrt'])

            classes = labels_onehot.columns.tolist()
            run_models(df_normed, labs_encoded, classes, cat_mod, model_type='xgb', out_fp=out_fp)

    #labels, labels_onehot = label_data_disease_status(df_meta)


    #df, labels_onehot = filter_classes(df, labels_onehot, ['C'])
    print(labels_onehot.sum())
    #normed_dfs = normalize_data(df, ['log_transform', 'zscore', 'lib_size_norm', 'lib_size_norm_log', 'sqrt', 'zscore_log'])
    #normed_dfs = normalize_data(df, ['log_transform', 'zscore', 'lib_size_norm', 'sqrt', 'raw'])
    # These normalization schemes work well for SVM models
    normed_dfs = normalize_data(df, ['sqrt'])
    run_models(normed_dfs, labels_onehot, 'M', model_type='svm')



main()
