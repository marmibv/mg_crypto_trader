import sys
# sys.path.append('../')

from src.utils import *
from src.calcEMA import calc_RSI
import src.myenv as myenv
import src.send_message as sm

from binance.client import Client
from pycaret.classification import ClassificationExperiment
from sklearn.model_selection import train_test_split

import pandas as pd
import datetime

import time
import traceback


def start_train_engine(symbol,
                       estimator,
                       train_size=myenv.train_size,
                       start_train_date='2010-01-01',
                       start_test_date=None,
                       numeric_features=myenv.numeric_features,
                       regression_times=myenv.regression_times,
                       regression_profit_and_loss=myenv.regression_profit_and_loss,
                       calc_rsi=True,
                       compare_models=False,
                       n_jobs=-1,
                       use_gpu=False,
                       verbose=False,
                       normalize=True,
                       fold=3):

    use_cols = date_features + numeric_features
    print('start_train_engine: use cols: ', use_cols)
    print(f'start_train_engine: reading data - start date: {start_train_date}...')
    # all_data = read_data(f'{datadir}/{symbol}', all_cols=None, use_cols=use_cols)
    all_data = get_data(symbol, save_database=False, interval='1h', tail=-1, adjust_index=True, columns=all_cols)
    all_data = all_data[(all_data['open_time'] >= start_train_date)]  # .copy()
    print('start_train_engine: info after reading data: ')
    all_data.info()

    if calc_rsi:
        print('start_train_engine: calculating RSI...')
        all_data = calc_RSI(all_data)
        numeric_features.append('rsi')
        all_data.dropna(inplace=True)
    print('start_train_engine: info after calculating RSI: ')
    all_data.info()

    print('start_train_engine: calculating regress_until_diff...')
    all_data = regress_until_diff(all_data, stop_loss, regression_profit_and_loss)
    print('start_train_engine: info after calculating regress_until_diff: ')
    all_data.info()

    print('start_train_engine: calculating regresstion_times...')
    all_data, _ = regresstion_times(all_data, numeric_features, regression_times, last_one=False)
    print('start_train_engine: info after calculating regresstion_times: ')
    all_data.info()

    print(f'start_train_engine: Filtering Data: start_train_date: {start_train_date} - start_test_date: {start_test_date}')
    train_data = all_data[(all_data['open_time'] >= start_train_date) & (all_data['open_time'] < start_test_date)]
    print('start_train_engine: info after filtering data: ')
    all_data.info()

    print('start_train_engine: setup model...:')
    ca = ClassificationExperiment()
    setup = ca.setup(train_data,
                     train_size=train_size,
                     target=label,
                     numeric_features=numeric_features,
                     date_features=['open_time'],
                     create_date_columns=["hour", "day", "month"],
                     fold_strategy='timeseries',
                     fold=fold,
                     session_id=123,
                     normalize=normalize,
                     use_gpu=use_gpu,
                     verbose=verbose,
                     n_jobs=n_jobs)

    # Accuracy	AUC	Recall	Prec.	F1	Kappa	MCC
    if compare_models:
        print('start_train_engine: comparing models...')
        best = setup.compare_models()
        estimator = setup.pull().index[0]
    else:
        print('start_train_engine: creating model...')
        best = setup.create_model(estimator)

    # predict on test set
    holdout_pred = setup.predict_model(best)

    print('start_train_engine: filtering test data...')
    test_data = all_data[all_data['open_time'] >= start_test_date]
    print('start_train_engine: info after filtering test data: ')
    test_data.info()

    print('start_train_engine: predicting model...')
    predict = setup.predict_model(best, data=test_data.drop(columns=[label]))
    predict[label] = test_data[label]
    predict['_score'] = predict['prediction_label'] == predict[label]

    print('start_train_engine: fianalizing model...')
    final_predict = setup.finalize_model(best)

    print('start_train_engine: predicting final model...')
    _predict = setup.predict_model(final_predict, data=test_data.sort_values(date_features).drop(columns=[label]))
    _predict[label] = test_data[label]
    _predict['_score'] = _predict['prediction_label'] == _predict[label]
    print('Score Mean:', _predict['_score'].mean())
    print('Score Group:', _predict[[label, '_score']].groupby(label).mean())

    print('start_train_engine: saving model...')
    save_model(symbol, final_predict, setup, estimator)


def main(args):
    try:
        estimator = myenv.estimator
        symbol = 'BTCUSDT'
        start_train_date = '2010-01-01'
        start_test_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        numeric_features = myenv.data_numeric_fields
        regression_times = myenv.regression_times
        regression_profit_and_loss = myenv.regression_profit_and_loss
        calc_rsi = False
        compare_models = False
        n_jobs = myenv.n_jobs
        train_size = myenv.train_size
        use_gpu = False
        verbose = False
        normalize = False
        fold = 3

        for arg in args:
            if (arg.startswith('-download_data')):
                sm.send_to_telegram('Iniciando MG Crypto Trader...')
                sm.send_to_telegram('Atualizando base de dados')
                download_data()
                sm.send_to_telegram('Base atualizada')

        for arg in args:
            if (arg.startswith('-symbol=')):
                symbol = arg.split('=')[1]

            if (arg.startswith('-estimator=')):
                estimator = arg.split('=')[1]

            if (arg.startswith('-start-train-date=')):
                start_train_date = arg.split('=')[1]

            if (arg.startswith('-start-test-date=')):
                start_test_date = arg.split('=')[1]

            if (arg.startswith('-numeric-features=')):
                aux = arg.split('=')[1]
                numeric_features = aux.split(',')

            if (arg.startswith('-regression-times=')):
                regression_times = int(arg.split('=')[1])

            if (arg.startswith('-n-jobs=')):
                n_jobs = int(arg.split('=')[1])

            if (arg.startswith('-regression-profit-and-loss=')):
                regression_profit_and_loss = float(arg.split('=')[1])

            if (arg.startswith('-train-size=')):
                train_size = float(arg.split('=')[1])

            if (arg.startswith('-fold=')):
                fold = int(arg.split('=')[1])

            if (arg.startswith('-calc-rsi')):
                calc_rsi = True

            if (arg.startswith('-all-cols')):
                numeric_features = myenv.all_cols

            if (arg.startswith('-copare-models')):
                compare_models = True

            if (arg.startswith('-use-gpu')):
                use_gpu = True

            if (arg.startswith('-normalize')):
                normalize = True

            if (arg.startswith('-verbose')):
                verbose = True

        sm.send_to_telegram(f'Iniciando Modelo Preditor para Symbol: {symbol}...')
        start_train_engine(symbol, estimator, train_size, start_train_date, start_test_date, numeric_features,
                           regression_times, regression_profit_and_loss, calc_rsi, compare_models, n_jobs, use_gpu, verbose, normalize, fold)
    except Exception as e:
        sm.send_to_telegram('ERRO: ' + str(e))
        traceback.print_exc()


if __name__ == '__main__':
    main(sys.argv[1:])
