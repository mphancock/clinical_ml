from monitor_gbt import Monitor
import pandas as pd
import numpy as np
import os
import argparse
import time

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from prepare_data import create_matched_kfold_splits
from prepare_data import get_cohort
from prepare_data import std_scale_train_test

from xgboost import XGBClassifier


def get_args():
    parser = argparse.ArgumentParser(description='Run XGBoost benchmark on EHR data')
    parser.add_argument('--demo', dest='demo', action='store_true', default=False)
    parser.add_argument('--no_cw', dest='cw', action='store_false')

    parser.add_argument('--no_std', dest='std', action='store_false')
    parser.add_argument('--no_phemed_bin', dest='phemed_bin', action='store_false')

    # parser.add_argument('--norm', action='store_true', default=True)
    # parser.add_argument('--phemed_bin', action='store_true', default=True)
    parser.add_argument('--impute_0', action='store_true', default=False)
    parser.add_argument('--reg', action='store_true', default=False)
    parser.add_argument('--csv_name', action='store', dest='csv_name', type=str)
    parser.add_argument('--fold', action='store', dest='fold_name', type=str)
    parser.add_argument('--model', action='store', dest='model', type=str)

    parser.add_argument('--early-stop', action='store_true', default=False)

    cmd_arg = parser.parse_args()

    print(cmd_arg)

    return cmd_arg


def get_pred(clf, df_test, fold, model):
    X_test = df_test.drop(['GRID', 'CLASS'], axis=1)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    df_pred = pd.DataFrame({'GRID': list(df_test['GRID']),
                            'PRED': list(y_pred)})
    if model == 'xgb':
        prob_lst = y_prob
        df_pred['PROB(0)'] = [prob[0] for prob in prob_lst]
        df_pred['PROB(1)'] = [prob[1] for prob in prob_lst]
    else:
        df_pred['PROB(0)'] = y_prob[:, 0]
        df_pred['PROB(1)'] = y_prob[:, 1]

    df_pred['CLASS'] = list(df_test['CLASS'])

    if model == 'lr' or model == 'gbt':
        df_pred['SCORE'] = clf.decision_function(X_test)
    else:
        df_pred['SCORE'] = df_pred['PROB(1)']

    df_pred['FOLD'] = [fold] * len(df_pred)

    return df_pred


def run_xgboost(df_train, df_test, params, fold):
    clf = XGBClassifier(**params)

    X_train, y_train = df_train.drop(['GRID', 'CLASS'], axis=1), df_train['CLASS']
    X_test, y_test = df_test.drop(['GRID', 'CLASS'], axis=1), df_test['CLASS']

    print(clf.get_params())

    # print(X_train.head())

    clf.fit(X=X_train,
            y=y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric='logloss',
            early_stopping_rounds=8,
            verbose=True)

    eval_results = clf.evals_result()
    df_eval = pd.DataFrame({**(list(eval_results.values())[0]), **(list(eval_results.values())[1])})

    df_pred = get_pred(clf=clf,
                       df_test=df_test,
                       fold=fold,
                       model='xgb')

    df_feat = pd.DataFrame({'FEATURE': list(X_train.columns), 'WEIGHT': list(clf.feature_importances_)})

    return {'pred': df_pred, 'feat': df_feat, 'eval': df_eval}


def run_logistic_regression(df_test, df_train, params, fold):
    print('Run logistic regression')

    X_train, y_train = df_train.drop(['GRID', 'CLASS'], axis=1), df_train['CLASS']
    X_test, y_test = df_test.drop(['GRID', 'CLASS'], axis=1), df_test['CLASS']

    # print(X_train.head())

    clf = LogisticRegression(**params)

    print(clf.get_params())

    ts1 = time.time()
    clf.fit(X=X_train,
            y=y_train)
    ts2 = time.time()
    print('computation time:\t{}'.format(ts2-ts1))

    df_pred = get_pred(clf=clf,
                       df_test=df_test,
                       fold=fold,
                       model='lr')

    df_feat = pd.DataFrame({'FEATURE': list(X_train.columns), 'WEIGHT': clf.coef_.T.flatten()})

    return {'pred': df_pred, 'feat': df_feat}


def run_random_forest(df_train, df_test, params, fold):
    print('run random forest')
    X_train, y_train = df_train.drop(['GRID', 'CLASS'], axis=1), df_train['CLASS']
    X_test, y_test = df_test.drop(['GRID', 'CLASS'], axis=1), df_test['CLASS']

    # print(X_train.head())

    clf = RandomForestClassifier(**params)

    print(clf.get_params)

    ts1 = time.time()
    clf.fit(X=X_train,
            y=y_train)
    ts2 = time.time()
    print('computation time:\t{}'.format(ts2-ts1))

    df_pred = get_pred(clf=clf,
                       df_test=df_test,
                       fold=fold,
                       model='rf')

    # print(clf.feature_importances_.shape)
    df_feat = pd.DataFrame({'FEATURE': list(X_train.columns), 'WEIGHT': clf.feature_importances_})

    return {'pred': df_pred, 'feat': df_feat}


def run_gbt(df_train, df_test, params, fold):
    print('run gbt')
    X_train, y_train = df_train.drop(['GRID', 'CLASS'], axis=1), df_train['CLASS']
    X_test, y_test = df_test.drop(['GRID', 'CLASS'], axis=1), df_test['CLASS']

    clf = GradientBoostingClassifier(**params)

    ts1 = time.time()
    clf.fit(X=X_train,
            y=y_train,
            monitor=Monitor(X_valid=X_test.values,
                            y_valid=y_test.values,
                            max_consecutive_decreases=5),
            sample_weight=compute_sample_weight('balanced', y_train, indices=None))

    ts2 = time.time()
    print('computation time:\t{}'.format(ts2-ts1))

    df_pred = get_pred(clf=clf,
                       df_test=df_test,
                       fold=fold,
                       model='gbt')

    df_feat = pd.DataFrame({'FEATURE': list(X_train.columns), 'WEIGHT': list(clf.feature_importances_)})

    return {'pred': df_pred, 'feat': df_feat}


def run_model_cv(df_cohort, df_map, params, model, early_stop, std, n_folds):
    skf = StratifiedKFold(n_splits=n_folds)
    skf.get_n_splits(X=df_cohort,
                     y=df_cohort['CLASS'])

    feat_lst = list(df_cohort.columns)
    feat_lst.remove('CLASS')
    feat_lst.remove('GRID')

    df_pred_cv = pd.DataFrame()
    df_feat_cv = pd.DataFrame({'FEATURE': feat_lst})

    print('# of subjects in cohort:\t{}'.format(len(df_cohort)))
    print('# of features in cohort:\t{}'.format(len(df_cohort.columns)))

    # fold = 0
    # for train_index, test_index in skf.split(X=df_cohort,
    #                                          y=df_cohort['CLASS']):
    kfold_dict = create_matched_kfold_splits(df_map=df_map,
                                             df_cohort=df_cohort,
                                             num_splits=n_folds)

    for fold in range(n_folds):
        train_index = (kfold_dict[fold])['train']
        test_index = (kfold_dict[fold])['test']

        print('FOLD: {}'.format(fold))

        # df_train, df_test = df_cohort.iloc[train_index], df_cohort.iloc[test_index]
        df_train, df_test = df_cohort.loc[df_cohort['GRID'].isin(train_index)], df_cohort.loc[df_cohort['GRID'].isin(test_index)]

        print('\t# of train subject: {}'.format(len(df_train)))
        print('\t# of test subject: {}'.format(len(df_test)))

        if std:
            df_train, df_test = std_scale_train_test(df_train=df_train,
                                                     df_test=df_test)

        if model == 'lr':
            clf_dict = run_logistic_regression(df_train=df_train,
                                               df_test=df_test,
                                               params=params['lr'],
                                               fold=fold)
        elif model == 'rf':
            clf_dict = run_random_forest(df_train=df_train,
                                         df_test=df_test,
                                         params=params['rf'],
                                         fold=fold)
        elif model == 'xgb':
            clf_dict = run_xgboost(df_train=df_train,
                                   df_test=df_test,
                                   fold=fold,
                                   params=params['xgb'])
        elif model == 'gbt':
            clf_dict = run_gbt(df_train=df_train,
                               df_test=df_test,
                               fold=fold,
                               params=params['gbt'])
        else:
            raise Exception('Unknown model')

        # if model == 'rf':
        #     print(clf_dict['pred'])

        df_pred_cv = pd.concat([df_pred_cv, clf_dict['pred']])
        df_feat_rank = clf_dict['feat']
        df_feat_rank = df_feat_rank.rename(columns={'WEIGHT': 'WEIGHT{}'.format(fold)})
        # needs to be inner so that feature reduction will allow # of features in df_feat_cv to be reduced
        df_feat_cv = df_feat_cv.merge(df_feat_rank, on='FEATURE', how='inner')

        # print(df_feat_rank.sort_values(by='WEIGHT{}'.format(fold), ascending=False).head())
        if early_stop:
            break

        fold += 1

    return df_pred_cv, df_feat_cv


def get_params(cmd_arg):
    params_lr = {'class_weight': 'balanced',
                 'solver': 'sag',
                 'verbose': 1, 'random_state': 1,
                 'n_jobs': 4,
                 'C': 1}

    params_rf = {'n_estimators': 200,
                 'max_features': .1,
                 'max_depth': 8,
                 'bootstrap': True,
                 'verbose': 1,
                 'class_weight': 'balanced',
                 'min_samples_leaf': 1,
                 'random_state': 1,
                 'n_jobs': 4}

    params_xgb = {'max_depth': 8, 'learning_rate': 0.1,
                  'n_estimators': 200, 'silent': False,
                  'objective': 'binary:logistic',
                  'booster': 'gbtree', 'n_jobs': 4,
                  'subsample': 1.0, 'scale_pos_weight': 10, 'random_state': 1}

    params_gbt = {'loss': 'deviance', 'learning_rate': 0.01,
                  'n_estimators': 100, 'subsample':1.0,
                  'min_samples_split': 2, 'min_samples_leaf': 1,
                  'max_depth': 3, 'verbose': 1,
                  'random_state': 1, 'presort': 'auto'}

    if not cmd_arg.cw:
        params_lr.pop('class_weight', 0)
        params_rf.pop('class_weight', 0)
        params_xgb.pop('scale_pos_weight', 0)

    model_params = {'lr': params_lr, 'rf': params_rf, 'xgb': params_xgb, 'gbt': params_gbt}

    return model_params


if __name__ == '__main__':
    cmd_arg = get_args()

    data_path = '/legodata/zhaoj/cvd_risk_time2/data_1_8/'
    csv_path = '/legodata/zhaoj/cvd_risk_time2/src/ML/csv/'

    suffix = ''

    if not cmd_arg.cw: suffix += '_no_cw'
    if not cmd_arg.phemed_bin: suffix += '_no_pm_bin'
    if cmd_arg.reg: suffix += '_reg'
    if cmd_arg.demo: suffix += '_demo'
    if not cmd_arg.std: suffix += '_no_std'
    if cmd_arg.csv_name: suffix += '_{}'.format(cmd_arg.csv_name)

    fold_name = cmd_arg.fold_name if cmd_arg.fold_name else ''

    trial_dir_path = os.path.join(csv_path, fold_name)
    if not os.path.isdir(trial_dir_path):
        os.makedirs(trial_dir_path)

    model_params = get_params(cmd_arg)

    for cohort in ['true_fram', 'full', 'fram']:
        print('Start {}'.format(cohort))

        df_cohort = get_cohort(data_path=os.path.join(data_path, 'processed'),
                               cohort=cohort,
                               cmd_arg=cmd_arg)

        df_pred_cv, df_feat_cv = run_model_cv(df_cohort=df_cohort,
                                              df_map=pd.read_csv(os.path.join(data_path, 'raw', 'control_case_mappings.csv')),
                                              params=model_params,
                                              model=cmd_arg.model,
                                              std=cmd_arg.std,
                                              early_stop=cmd_arg.early_stop,
                                              n_folds=10)

        df_pred_cv = df_pred_cv.merge(df_cohort[['GRID']],
                                      how='inner',
                                      on='GRID')

        df_pred_cv.to_csv(os.path.join(csv_path, trial_dir_path, '{}_pred_{}{}.csv'.format(cmd_arg.model, cohort, suffix)), index=False)
        df_feat_cv.to_csv(os.path.join(csv_path, trial_dir_path, '{}_feat_{}{}.csv'.format(cmd_arg.model, cohort, suffix)), index=False)





