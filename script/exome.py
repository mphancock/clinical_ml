import pandas as pd
import numpy as np
import os
import argparse

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.random_projection import GaussianRandomProjection

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import BatchNormalization
# from keras.layers.noise import AlphaDropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.optimizers import Adam
from keras.optimizers import SGD

import datetime


def get_args():
    parser = argparse.ArgumentParser(description='Run DNN on genetic data')
    parser.add_argument('--no_demo', dest='demo', action='store_false')
    parser.add_argument('--no_bin_snp', dest='bin_snp', action='store_false')

    parser.add_argument('--early-stop', action='store_true', default=False)

    parser.add_argument('--name', action='store', dest='name', type=str)
    parser.add_argument('--fold', action='store', dest='fold', type=str)

    cmd_arg = parser.parse_args()

    print(cmd_arg)

    return cmd_arg


def get_network(input_shape, units1=128, units2=64, units3=8,
                dropout_rate1=0.2, dropout_rate2=0.1, dropout_rate3=0.1, lr=0.00004):
    model = Sequential()
    model.add(Dense(units1, activation='relu', input_shape=(input_shape,)))
    # Add one hidden layer
    model.add(Dropout(dropout_rate1))
    model.add(Dense(units2, activation='relu'))
    model.add(Dropout(dropout_rate2))
    model.add(Dense(units3, activation='relu'))
    model.add(Dropout(dropout_rate3))
    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(lr=lr, epsilon=.001)  # lr=0.00004
    #     opt = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def get_network_2_bn(input_shape, units, dropout, epsilon, lr=0.00004):
    units1, units2 = units
    dropout_rate1, dropout_rate2 = dropout

    model = Sequential()

    model.add(Dense(units1, input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate1))

    model.add(Dense(units2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate2))

    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(lr=lr, epsilon=epsilon)  # lr=0.00004
    #     opt = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def get_network_3_bn(input_shape, units, dropout, epsilon, lr=0.00004):
    units1, units2, units3 = units
    dropout_rate1, dropout_rate2, dropout_rate3 = dropout

    model = Sequential()

    model.add(Dense(units1, input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate1))

    model.add(Dense(units2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate2))

    model.add(Dense(units3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate3))

    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(lr=lr, epsilon=epsilon)  # lr=0.00004
    #     opt = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def get_network_4_bn(input_shape, units, dropout, epsilon, lr=0.00004):
    model = Sequential()

    units1, units2, units3, units4 = units
    dropout_rate1, dropout_rate2, dropout_rate3, dropout_rate4 = dropout

    model.add(Dense(units1, input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate1))

    model.add(Dense(units2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate2))

    model.add(Dense(units3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate3))

    model.add(Dense(units4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate4))

    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(lr=lr, epsilon=epsilon)  # lr=0.00004
    #     opt = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


from keras.layers import Input
from keras.models import Model


def encode(X_train, X_test, n_comp):
    input_layer = Input(shape=(X_train.shape[1],))
    encoded = Dense(n_comp, activation='relu')(input_layer)
    decoded = Dense(X_train.shape[1], activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(X_train.values, X_train.values,
                    epochs=20,
                    batch_size=128,
                    shuffle=False, validation_data=(X_train.values, X_train.values))

    encoder = Model(input_layer, encoded)

    X_train_encoded = encoder.predict(X_train.values)
    X_test_encoded = encoder.predict(X_test.values)

    return X_train_encoded, X_test_encoded


def run_genetic(df_train, df_test, name, fold, encode, n_comp):
    X_train, y_train = df_train.drop(['CLASS', 'GRID', 'predict'], axis=1), df_train['CLASS']
    X_test, y_test = df_test.drop(['CLASS', 'GRID', 'predict'], axis=1), df_test['CLASS']

    #     rand_proj = GaussianRandomProjection(n_components=200)
    #     rand_proj.fit(X_train)
    #     X_train = rand_proj.transform(X_train)

    model = get_network_2_bn(input_shape=X_train.shape[1], units=(256, 8), dropout=(.75, .5), epsilon=.001)

    print('# of subjects in cohort:\t{}'.format(len(df_fram_genetic)))
    #     print('# of features in cohort:\t{}'.format(len(df_fram_genetic.columns)))

    print(X_test.shape[1])

    hdf5_file_path = os.path.join(results_path, 'model', '{}.hdf5'.format(str(datetime.datetime.now().time())))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
    checkpointer = ModelCheckpoint(filepath=hdf5_file_path, verbose=1,
                                   save_best_only=True)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    if False:
        X_train, X_test = encode(X_train, X_test, n_comp=n_comp)

    num_epochs = 150

    class_weight_vec = compute_class_weight(class_weight='balanced',
                                            classes=[0,1],
                                            y=y_train)

    print(class_weight_vec)

    history_model = model.fit(X_train,
                              y_train,
                              batch_size=256,
                              class_weight={0: class_weight_vec[0], 1: class_weight_vec[1]},
                              epochs=num_epochs,
                              verbose=1,
                              validation_split=0.11, callbacks=[reduce_lr, checkpointer])

    model.load_weights(hdf5_file_path)

    # X_test = rand_proj.transform(X_test)
    y_score = model.predict(X_test)

    dict_train_history = {'EPOCH': list(range(num_epochs)),
                          'VAL_LOSS': history_model.history['val_loss'],
                          'LOSS': history_model.history['loss'],
                          'VAL_ACC': history_model.history['val_acc'],
                          'ACC': history_model.history['acc'],
                          'FOLD': [fold] * num_epochs}

    y_score = y_score.flatten()
    y_pred = [0 if score < .5 else 1 for score in list(y_score)]

    return pd.DataFrame({'CLASS': list(y_test), 'PRED': y_pred, 'SCORE': list(y_score), 'FOLD': [fold] * len(y_test)}), pd.DataFrame(dict_train_history)


def run_genetic_cv(df_cohort, name, n_folds, early_stop, encode, n_comp):
    skf = StratifiedKFold(n_splits=n_folds)
    skf.get_n_splits(X=df_cohort,
                     y=df_cohort['CLASS'])
    #
    # feat_lst = list(df_fram_genetic.columns)
    # feat_lst.remove('CLASS')
    # feat_lst.remove('GRID')
    #
    # df_pred_cv = pd.DataFrame()
    # df_feat_cv = pd.DataFrame({'FEATURE': feat_lst})

    df_pred_cv = pd.DataFrame()
    df_train_cv = pd.DataFrame()

    fold = 0

    for train_index, test_index in skf.split(X=df_cohort,
                                             y=df_cohort['CLASS']):
        print('FOLD: {}'.format(fold))

        df_train, df_test = df_cohort.iloc[train_index], df_cohort.iloc[test_index]

        df_pred, df_train = run_genetic(df_train,
                                        df_test,
                                        name=name,
                                        fold=fold,
                                        encode=encode,
                                        n_comp=n_comp)

        df_pred_cv = pd.concat([df_pred_cv, df_pred])
        df_train_cv = pd.concat([df_train_cv, df_train])

        fold += 1
        if early_stop:
            break

    return df_pred_cv, df_train_cv


if __name__ =='__main__':
    cmd_arg = get_args()

    data_path = '/legodata/exome/'

    # TW_Liver.csv  TW_Whole_Blood.csv

    df_exome = pd.read_csv(os.path.join(data_path, 'cvd_merged', 'TW_Whole_Blood.csv'))

    results_path = '/legodata/zhaoj/cvd_risk_time2/src/exome/'

    df_fram = pd.read_csv(os.path.join('/legodata/zhaoj/cvd_risk_time2/genetic_data/for_framingham/processed/merged_fr_results.csv'))

    # df_fram_2 = pd.read_csv('/legodata/zhaoj/shared_with_mat/exom/merged_fr_results.csv')
    #
    # df_fram = pd.concat([df_fram_1, df_fram_2])

    df_exome_merged = df_exome.merge(df_fram[['GRID', 'predict']], how='inner', left_on='grid', right_on='GRID')
    df_exome_merged = df_exome_merged.rename(columns={'grid': 'GRID'})

    df_fram_genetic = df_exome_merged.loc[((df_exome_merged['predict'] == 0) & (df_exome_merged['CLASS'] == 1)) | (df_exome_merged['CLASS'] == 0)]
    df_full_genetic = df_exome_merged

    df_exome_cohort = {'full': df_full_genetic, 'fram': df_fram_genetic}

    for cohort in ['full', 'fram']:
        print(len(df_exome_cohort[cohort]))

        df_pred, df_train = run_genetic_cv(df_cohort=df_exome_cohort[cohort],
                                           name=cmd_arg.name,
                                           n_folds=10,
                                           early_stop=cmd_arg.early_stop,
                                           encode=True,
                                           n_comp=100)

        df_pred.to_csv(os.path.join(results_path, 'results', '{}_{}_pred.csv'.format(cmd_arg.name, cohort)), index=False)
        df_train.to_csv(os.path.join(results_path, 'results', '{}_{}_train.csv'.format(cmd_arg.name, cohort)), index=False)






