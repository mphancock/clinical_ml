import pandas as pd
import numpy as np
import os
import argparse

from sklearn.model_selection import StratifiedKFold
from sklearn.random_projection import GaussianRandomProjection

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
# from keras.layers.noise import AlphaDropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.optimizers import Adam
from keras.optimizers import SGD


def get_args():
    parser = argparse.ArgumentParser(description='Run DNN on genetic data')
    parser.add_argument('--no_demo', dest='demo', action='store_false')
    parser.add_argument('--no_bin_snp', dest='bin_snp', action='store_false')
    parser.add_argument('--csv_name', action='store', dest='csv_name', type=str)
    parser.add_argument('--fold', action='store', dest='fold_name', type=str)

    cmd_arg = parser.parse_args()

    print(cmd_arg)

    return cmd_arg


def get_network(input_shape,units1=128,units2=64,units3=8,
                       dropout_rate1=0.2,dropout_rate2=0.1,dropout_rate3=0.1, lr=0.00004):
    model = Sequential()
    model.add(Dense(units1, activation='relu', input_shape=(input_shape,)))
    # Add one hidden layer
    model.add(Dropout(dropout_rate1))
    model.add(Dense(units2, activation='relu'))
    model.add(Dropout(dropout_rate2))
    model.add(Dense(units3, activation='relu'))
    model.add(Dropout(dropout_rate3))
    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(lr=lr)  # lr=0.00004
    # opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def run_network(df_cohort, results_path, n_folds, csv_name):
    skf = StratifiedKFold(n_splits=n_folds)
    skf.get_n_splits(X=df_cohort,
                     y=df_cohort['CLASS'])
    #
    # feat_lst = list(df_cohort.columns)
    # feat_lst.remove('CLASS')
    # feat_lst.remove('GRID')
    #
    # df_pred_cv = pd.DataFrame()
    # df_feat_cv = pd.DataFrame({'FEATURE': feat_lst})

    df_cv = pd.DataFrame()
    df_cv_train_history = pd.DataFrame()

    fold = 0

    for train_index, test_index in skf.split(X=df_cohort,
                                             y=df_cohort['CLASS']):
        print('FOLD: {}'.format(fold))

        df_train, df_test = df_cohort.iloc[train_index], df_cohort.iloc[test_index]

        X_train, y_train = df_train.drop(['CLASS', 'GRID'], axis=1), df_train['CLASS']
        X_test, y_test = df_test.drop(['CLASS', 'GRID'], axis=1), df_test['CLASS']

        rand_proj = GaussianRandomProjection(n_components='auto')
        X_train = rand_proj.fit_transform(X_train)

        model = get_network(input_shape=X_train.shape[1])

        print('# of subjects in cohort:\t{}'.format(len(df_cohort)))
        print('# of features in cohort:\t{}'.format(len(df_cohort.columns)))

        hdf5_file_path = os.path.join(results_path, 'dnnmodel_{}.hdf5'.format(csv_name))
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
        checkpointer = ModelCheckpoint(filepath=hdf5_file_path, verbose=1,
                                       save_best_only=True)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        num_epochs = 100

        history_model = model.fit(X_train,
                                  y_train,
                                  batch_size=128,
                                  epochs=num_epochs,
                                  verbose=1,
                                  validation_split=0.11, callbacks=[reduce_lr, checkpointer])

        model.load_weights(hdf5_file_path)

        X_test = rand_proj.transform(X_test)
        y_score = model.predict(X_test)

        auc = roc_auc_score(y_test, y_score)  # main_input
        # cvs_aucs.append(auc)
        ap = average_precision_score(y_test, y_score)

        df_cv = pd.concat([df_cv, pd.DataFrame({'FOLD': [fold], 'AUC': [auc], 'AP': [ap]})])

        dict_train_history = {'epoch': list(range(num_epochs)),
                              'val_loss': history_model.history['val_loss'],
                              'loss': history_model.history['loss'],
                              'val_acc': history_model.history['val_acc'],
                              'acc': history_model.history['acc'],
                              'fold': [fold] * num_epochs}

        df_cv_train_history = pd.concat([df_cv_train_history, pd.DataFrame(dict_train_history)])

        fold += 1

        break

    return df_cv, df_cv_train_history
    # fold = 0
    # for train_index, test_index in skf.split(X=df_cohort,
    #                                          y=df_cohort['CLASS']):


if __name__ =='__main__':
    data_path = '/legodata/zhaoj/cvd_risk_time2/genetic_data'
    results_path = '/legodata/zhaoj/cvd_risk_time2/src/dnn/'
    df_genetic = pd.read_csv(os.path.join(data_path, 'snps_5e6_with_demo.csv'))

    cmd_arg = get_args()

    suffix = ''

    if cmd_arg.bin_snp:
        snps_col_lst = list(df_genetic.drop(['GENDER', 'AGE', 'RACE_A', 'RACE_B', 'RACE_W', 'RACE_U', 'GRID', 'CLASS'], axis=1).columns)
        df_genetic = pd.get_dummies(df_genetic.astype('str'), columns=snps_col_lst)
        df_genetic['CLASS'] = df_genetic['CLASS'].astype('int64')

    if not cmd_arg.demo:
        df_genetic = df_genetic.drop(['AGE', 'GENDER', 'RACE_A', 'RACE_U', 'RACE_W', 'RACE_B'], axis=1)
        suffix += '_drop_demo'

    if cmd_arg.csv_name:
        suffix += '_{}'.format(cmd_arg.csv_name)

    df_cv, df_cv_train_history = run_network(df_cohort=df_genetic,
                                             results_path=results_path,
                                             csv_name=cmd_arg.csv_name,
                                             n_folds=10)

    df_cv.to_csv(os.path.join(results_path, 'df_cv{}.csv'.format(suffix)), index=False)
    df_cv_train_history.to_csv(os.path.join(results_path, 'df_cv_train_history{}.csv'.format(suffix)), index=False)






