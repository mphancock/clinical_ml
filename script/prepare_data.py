import pandas as pd
import os

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier


def xgb_regularization(df_cohort):
    df_features = df_cohort.drop(['GRID', 'CLASS'], axis=1)

    clf_feat = SelectFromModel(estimator=XGBClassifier().fit(X=df_features,
                                                             y=df_cohort['CLASS']),
                               prefit=True)

    clf_feat.transform(df_features)

    feat_mask = list(clf_feat.get_support())
    lst_select_feat = list()
    for it in range(len(feat_mask)):
        if feat_mask[it]:
            lst_select_feat.append(list(df_features.columns)[it])

    df_cohort_select_feat = df_cohort[['GRID', 'CLASS'] + lst_select_feat]

    print('NUM FEATURES SELECTED: {}'.format(len(df_cohort_select_feat.columns)))

    return df_cohort_select_feat


def l1_regularization(df, alpha_inv):
    print('L1 REGULARIZATION')
    l1_clf = LogisticRegression(penalty='l1',
                                C=alpha_inv)

    l1_clf.fit(X=df.drop(['GRID', 'CLASS'], axis=1),
               y=df['CLASS'])

    feature_lst = list(df.columns)
    feature_lst.remove('GRID')
    feature_lst.remove('CLASS')

    df_feat = pd.DataFrame({'FEATURE': feature_lst, 'COEF': l1_clf.coef_.tolist()[0]})

    reg_feat_lst = list(df_feat.loc[df_feat['COEF'] != 0]['FEATURE'])

    df_reg = pd.concat([df['GRID'], df[reg_feat_lst], df['CLASS']], axis=1)

    return df_reg


def get_chi2_features(df_cohort):
    df_features = df_cohort.drop(['GRID'], axis=1)
    df_features[df_features < 0] = 0

    results = chi2(df_cohort.drop(['GRID', 'AGE_AT_BASELINE', 'DURATION_BF_BASELINE', 'RACE_B', 'RACE_W', 'RACE_U', 'CLASS'],
                       axis=1), df_cohort['CLASS'])


# def norm_feat(df):
#     print('NORMALZING FEATURES')
#     features_norm, norm = normalize(X=df.drop(['GRID', 'CLASS'], axis=1),
#                                     norm='max',
#                                     axis=0,
#                                     return_norm=True)
#
#     feat_lst = list(df.columns)
#     feat_lst.remove('GRID')
#     feat_lst.remove('CLASS')
#
#     df_feat_norm = pd.DataFrame(features_norm)
#     df_feat_norm.columns = feat_lst
#
#     df = pd.concat([df['GRID'], df_feat_norm, df['CLASS']], axis=1)
#
#     return df

def std_scale_train_test(df_train, df_test):
    print('BEGIN STD SCALE')

    X_train = df_train.drop(['CLASS', 'GRID'], axis=1)
    X_test = df_test.drop(['CLASS', 'GRID'], axis=1)

    std_scaler = StandardScaler()
    std_scaler.fit(X_train)

    X_train_std = std_scaler.transform(X_train)
    X_test_std = std_scaler.transform(X_test)

    X_train_std = pd.DataFrame(X_train_std)
    X_train_std.columns = list(X_train.columns)

    X_test_std = pd.DataFrame(X_test_std)
    X_test_std.columns = list(X_test.columns)

    X_test_std['CLASS'] = list(df_test['CLASS'])
    X_test_std['GRID'] = list(df_test['GRID'])

    X_train_std['CLASS'] = list(df_train['CLASS'])
    X_train_std['GRID'] = list(df_train['GRID'])

    return X_train_std, X_test_std


def split_data(df, pct_test):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['CLASS'], axis=1), df['CLASS'],
                                                        stratify=df['CLASS'],
                                                        test_size=pct_test,
                                                        shuffle=True,
                                                        random_state=8925)

    subj_id = X_test.pop('GRID')
    X_train.pop('GRID')

    print('Cohort stats:')
    print('\t{}\ttrain subjects'.format(len(X_train)))
    print('\t{}\ttrain features'.format(len(list(X_train.columns))))
    print('\t{}\ttest subjects'.format(len(X_test)))

    return X_train, X_test, y_train, y_test, subj_id


def create_matched_kfold_splits(df_map, df_cohort, num_splits):
    # df_case = df_cohort.sample(frac=1)
    df_case = df_cohort.loc[df_cohort['CLASS'] == 1]

    num_case_per_split = len(df_case) // num_splits

    kfold_dict = {}

    for i in range(num_splits):
        fold_dict = {}

        if i == (num_splits - 1):
            df_case_split = df_case.iloc[i * num_case_per_split:]
        else:
            df_case_split = df_case.iloc[i * num_case_per_split:(i+1) * num_case_per_split]

        print('split: {}\t # of cases: {}'.format(i, len(df_case_split)))

        df_matched_controls = df_case_split.merge(df_map,
                                                  how='inner',
                                                  on='GRID')

        print('\t\t # of matched controls: {}'.format(len(df_matched_controls)))

        test_case_control_lst = list(df_case_split['GRID'])

        test_case_control_lst.extend(list(df_matched_controls['GRID_CONTROL']))

        fold_dict['test'] = test_case_control_lst

        print('\t\t # of unique cases and matched controls:\t{}'.format(len(set(test_case_control_lst))))

        train_case_control_lst = list(set(df_cohort['GRID']) - set(test_case_control_lst))

        fold_dict['train'] = train_case_control_lst

        kfold_dict[i] = fold_dict

    return kfold_dict


def convert_phemed_to_bin(df_cohort):
    print('CONVERTING PHEMED TO BIN')
    col_lst = list(df_cohort.columns)

    med_col_lst = [col for col in col_lst if 'med_' in col]
    phe_col_lst = [col for col in col_lst if 'phe_' in col]

    df_med = df_cohort[med_col_lst]
    df_phe = df_cohort[phe_col_lst]

    for col in med_col_lst:
        df_med[col] = df_med[col].map(lambda x: 'yes' if x != 0 else 'no')
    df_cohort[med_col_lst] = pd.get_dummies(df_med, drop_first=True)

    for col in phe_col_lst:
        df_phe[col] = df_phe[col].map(lambda x: 'yes' if x != 0 else 'no')
    df_cohort[phe_col_lst] = pd.get_dummies(df_phe, drop_first=True)

    return df_cohort


def get_cohort(data_path, cohort, cmd_arg):
    # if cohort == 'fram':
    #     if cmd_arg.impute_0:
    #         df_cohort = pd.read_csv(os.path.join(data_path, 'df_fram_cohort_0.csv'))
    #     else:
    #         df_cohort = pd.read_csv(os.path.join(data_path, 'df_fram_cohort.csv'))
    # elif cohort == 'full':
    #     if cmd_arg.impute_0:
    #         df_cohort = pd.read_csv(os.path.join(data_path, 'df_full_cohort_0.csv'))
    #         df_cohort = df_cohort.drop(['predict'], axis=1)
    #     else:
    #         df_cohort = pd.read_csv(os.path.join(data_path, 'df_full_cohort.csv'))
    #         df_cohort = df_cohort.drop(['predict'], axis=1)
    # else:
    #     raise Exception('unknown cohort')

    file = 'df_{}_cohort.csv'.format(cohort)

    df_cohort = pd.read_csv(os.path.join(data_path, file))
    df_cohort = df_cohort.drop(['predict'], axis=1)
    if cmd_arg.phemed_bin:
        df_cohort = convert_phemed_to_bin(df_cohort)

    if not cmd_arg.demo:
        print('drop_demo')
        df_cohort = df_cohort.drop(['AGE_AT_BASELINE', 'RACE_W', 'RACE_B', 'RACE_U', 'GENDER'], axis=1)

    # if cmd_arg.norm:
    #     df_cohort = norm_feat(df_cohort)

    if cmd_arg.reg:
        # df_cohort = l1_regularization(df=df_cohort,
        #                               alpha_inv=100)
        df_cohort = xgb_regularization(df_cohort=df_cohort)

    return df_cohort
