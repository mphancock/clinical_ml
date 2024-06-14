from genetic import run_genetic_cv
from genetic import encode
import pandas as pd
import os
import argparse

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    data_path = '/legodata/zhaoj/cvd_risk_time2/genetic_data'
    dnn_path = '/legodata/zhaoj/cvd_risk_time2/dnn/'

    parser = argparse.ArgumentParser(description='Run DNN on genetic data')
    parser.add_argument('--thresh', action='store', dest='thresh', type=str)

    cmd_arg = parser.parse_args()

    df_genetic = pd.read_csv(os.path.join(data_path, 'snps_{}_with_demo.csv'.format(cmd_arg.thresh)))

    if True:
        snps_col_lst = list(df_genetic.drop(['GENDER', 'AGE', 'RACE_A', 'RACE_B', 'RACE_W', 'RACE_U', 'GRID', 'CLASS'], axis=1).columns)
        df_genetic = pd.get_dummies(df_genetic.astype('str'), columns=snps_col_lst)
        df_genetic['CLASS'] = df_genetic['CLASS'].astype('int64')

    if True:
        df_genetic = df_genetic.drop(['AGE', 'GENDER', 'RACE_A', 'RACE_U', 'RACE_W', 'RACE_B'], axis=1)

    df_cohort = pd.read_csv(os.path.join(data_path, 'for_framingham/processed', 'merged_fr_results.csv'))
    df_genetic_merged = df_genetic.merge(df_cohort[['GRID', 'predict']], how='inner', on='GRID')

    df_fram_genetic = df_genetic_merged.loc[((df_genetic_merged['predict'] == 0) & (df_genetic_merged['CLASS'] == 1)) | (df_genetic_merged['CLASS'] == 0)]
    df_full_genetic = df_genetic_merged

    df_fram_encoder_metric = pd.DataFrame()
    df_full_encoder_metric = pd.DataFrame()

    for n_comp in [None, 25, 50, 100, 250, 500, 1000, 2000]:
        df_genetic_cohort = {'full': df_full_genetic, 'fram': df_fram_genetic}

        for cohort in ['full', 'fram']:
            df_pred, df_train = run_genetic_cv(df_cohort=df_genetic_cohort[cohort],
                                               name=None,
                                               n_folds=10,
                                               n_comp=n_comp,
                                               early_stop=True)

            score_report = classification_report(df_pred['CLASS'], df_pred['PRED'])
            auroc = roc_auc_score(df_pred['CLASS'], df_pred['SCORE'])
            mse = mean_squared_error(df_pred['CLASS'], df_pred['PRED'])

            df_n_comp_metric = pd.DataFrame({'N_COMP': n_comp, 'AUROC': auroc, 'PREC': score_report.split()[10],
                                             'RECALL': score_report.split()[11], 'MSE': mse})

            if cohort == 'full':
                pd.concat([df_full_encoder_metric, df_n_comp_metric])
            else:
                pd.concat([df_fram_encoder_metric, df_n_comp_metric])

    df_fram_encoder_metric.to_csv(os.path.join(dnn_path, 'results', 'fram_encoder_results.csv'))
    df_full_encoder_metric.to_csv(os.path.join(dnn_path, 'results', 'full_encoder_results.csv'))





