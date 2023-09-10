import sys

from src.utils import *
from src.calcEMA import calc_RSI
import src.myenv as myenv
import src.send_message as sm

from pycaret.classification import ClassificationExperiment

import pandas as pd
import datetime

import traceback


def start_train_engine(symbol,
                       estimator,
                       train_size=myenv.train_size,
                       start_train_date='2010-01-01',
                       start_test_date=None,
                       numeric_features=myenv.numeric_features,
                       stop_loss=myenv.stop_loss,
                       regression_times=myenv.regression_times,
                       times_regression_profit_and_loss=myenv.times_regression_profit_and_loss,
                       calc_rsi=True,
                       compare_models=False,
                       n_jobs=-1,
                       use_gpu=False,
                       verbose=False,
                       normalize=True,
                       fold=3,
                       parametros=None):

    all_data, features_added = prepare_all_data(symbol,
                                                start_train_date,
                                                calc_rsi,
                                                numeric_features,
                                                True,
                                                times_regression_profit_and_loss,
                                                regression_times)

    train_data = prepare_train_data(all_data, start_train_date, start_test_date)

    print('start_train_engine: setup model...:')
    ca = ClassificationExperiment()
    setup = ca.setup(train_data,
                     train_size=train_size,
                     target=label,
                     numeric_features=numeric_features + features_added,
                     date_features=['open_time'],
                     create_date_columns=["hour", "day", "month"],
                     fold_strategy='timeseries',
                     fold=fold,
                     session_id=123,
                     normalize=normalize,
                     use_gpu=use_gpu,
                     verbose=verbose,
                     n_jobs=n_jobs,)

    # Accuracy	AUC	Recall	Prec.	F1	Kappa	MCC
    if compare_models:
        print('start_train_engine: comparing models...')
        best_model = setup.compare_models()
        estimator = setup.pull().index[0]
    else:
        print('start_train_engine: creating model...')
        best_model = setup.create_model(estimator)

    test_data, ajusted_test_data = prepare_test_data(all_data, start_test_date)

    df_final_predict, final_model = predict_model(setup, best_model, test_data, ajusted_test_data)

    print('start_train_engine: saving model...')
    model_name = save_model(symbol, final_model, setup, estimator, stop_loss, regression_times, times_regression_profit_and_loss)

    print('start_train_engine: simule trading...')
    start_test_date = df_final_predict['open_time'].min()
    end_test_date = df_final_predict['open_time'].max()

    print('Min Data: ', start_test_date)
    print('Max Data: ', start_test_date)
    saldo_inicial = 100.0
    saldo_final = simule_trading_crypto(df_final_predict, start_test_date, end_test_date, saldo_inicial, stop_loss)

    save_results(model_name, symbol, estimator, train_size, start_train_date, start_test_date, regression_times,
                 times_regression_profit_and_loss, stop_loss, fold, saldo_inicial, saldo_final, parametros)


def save_results(model_name,
                 symbol,
                 estimator,
                 train_size,
                 start_train_date,
                 start_test_date,
                 regression_times,
                 regression_profit_and_loss,
                 stop_loss,
                 fold,
                 saldo_inicial,
                 saldo_final,
                 parametros):

    df_resultado_simulacao = pd.DataFrame()
    if (os.path.exists('resultado_simulacao.csv')):
        df_resultado_simulacao = pd.read_csv('resultado_simulacao.csv', sep=';')

    result_simulado = {}
    result_simulado['model_name'] = model_name
    result_simulado['data'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_simulado['symbol'] = symbol
    result_simulado['estimator'] = estimator
    result_simulado['train_size'] = train_size
    result_simulado['start_train_date'] = start_train_date
    result_simulado['start_test_date'] = start_test_date
    result_simulado['regression_times'] = regression_times
    result_simulado['regression_profit_and_loss'] = regression_profit_and_loss
    result_simulado['stop_loss'] = stop_loss
    result_simulado['fold'] = fold
    result_simulado['saldo_inicial'] = saldo_inicial
    result_simulado['saldo_final'] = saldo_final
    result_simulado['parametros'] = parametros

    df_resultado_simulacao = pd.concat([df_resultado_simulacao, pd.DataFrame([result_simulado])], ignore_index=True)
    df_resultado_simulacao.sort_values('saldo_final', inplace=True)

    df_resultado_simulacao.to_csv('resultado_simulacao.csv', sep=';', index=False)


def prepare_all_data(symbol,
                     start_train_date,
                     calc_rsi=True,
                     numeric_features=myenv.numeric_features,
                     prepare_profit_and_loss=True,
                     times_regression_profit_and_loss=myenv.times_regression_profit_and_loss,
                     regression_times=myenv.regression_times):

    use_cols = date_features + numeric_features
    print('start_train_engine: use cols: ', use_cols)
    print(f'start_train_engine: reading data - start date: {start_train_date}...')
    all_data = get_data(symbol, save_database=False, interval='1h', tail=-1, adjust_index=True, columns=use_cols)
    print('start_train_engine: get_data  all_data duplicated: ', all_data.index.duplicated().sum())

    if calc_rsi:
        print('start_train_engine: calculating RSI...')
        all_data = calc_RSI(all_data)
        numeric_features.append('rsi')
        all_data.dropna(inplace=True)
    print('start_train_engine: info after calculating RSI: ')
    all_data.info()
    print('start_train_engine: all_data duplicated: ', all_data.index.duplicated().sum())

    features_added = []
    if regression_times > 0:
        print('start_train_engine: calculating regresstion_times...')
        all_data, features_added = regresstion_times(all_data, numeric_features, regression_times, last_one=False)
        print('start_train_engine: info after calculating regresstion_times: ')
        all_data.info()
        print('start_train_engine: all_data duplicated: ', all_data.index.duplicated().sum())

    all_data = all_data[(all_data['open_time'] >= start_train_date)]  # .copy()
    print('start_train_engine: info after reading data: ')
    all_data.info()
    print('start_train_engine: filter start_train_date all_data duplicated: ', all_data.index.duplicated().sum())

    if prepare_profit_and_loss:
        print('start_train_engine: calculating regression_profit_and_loss...')
        all_data = regression_profit_and_loss(all_data, label, stop_loss, times_regression_profit_and_loss)
        print('start_train_engine: info after calculating regression_profit_and_loss: ')
        all_data.info()
        print('start_train_engine: all_data duplicated: ', all_data.index.duplicated().sum())

    return all_data, features_added


def prepare_train_data(all_data, start_train_date, start_test_date):
    print(f'start_train_engine: Filtering train_data: start_train_date: {start_train_date} - start_test_date: {start_test_date}')
    train_data = all_data[(all_data['open_time'] >= start_train_date) & (all_data['open_time'] < start_test_date)]
    print('start_train_engine: info after filtering train_data: ')
    train_data.info()
    print('start_train_engine: train_data duplicated: ', train_data.index.duplicated().sum())

    return train_data


def prepare_test_data(all_data, start_test_date):
    print(f'start_train_engine: Filtering test_data: start_test_date: {start_test_date}')
    test_data = all_data[all_data['open_time'] >= start_test_date]
    print('start_train_engine: info after filtering test_data: ')
    test_data.info()
    print('start_train_engine: test_data duplicated: ', test_data.index.duplicated().sum())

    print('start_train_engine: predicting model...')
    ajusted_test_data = test_data.drop(columns=[label])
    print('start_train_engine: _test_data drop label duplicated: ', ajusted_test_data.index.duplicated().sum())

    return test_data, ajusted_test_data


def predict_model(setup, model, test_data, ajusted_test_data):
    print('predict_model: predicting model...')
    df_predict = setup.predict_model(model, data=ajusted_test_data)
    df_final_predict = df_predict
    final_model = model

    if test_data is not None:
        df_predict[label] = test_data[label]
        df_predict['_score'] = df_predict['prediction_label'] == df_predict[label]

        print('start_train_engine: finalizing model...')
        final_model = setup.finalize_model(model)

        print('start_train_engine: predicting final model...')
        df_final_predict = setup.predict_model(final_model, data=ajusted_test_data)

        df_final_predict[label] = test_data[label]
        df_final_predict['_score'] = df_final_predict['prediction_label'] == df_final_predict[label]

        print('Score Mean:', df_final_predict['_score'].mean())
        print('Score Group:', df_final_predict[[label, '_score']].groupby(label).mean())

    return df_final_predict, final_model


def exec_simule_trading(symbol,
                        calc_rsi=True,
                        numeric_features=myenv.numeric_features,
                        start_test_date=None,
                        start_value=100.0,
                        stop_loss=myenv.stop_loss,
                        estimator=myenv.estimator):

    all_data, _ = prepare_all_data(symbol,
                                   start_test_date,
                                   calc_rsi,
                                   numeric_features,
                                   False,
                                   regression_profit_and_loss,
                                   regression_times)

    experiment, model = load_model(symbol, estimator)

    df_final_predict, _ = predict_model(experiment, model, None, all_data)

    start_test_date = df_final_predict['open_time'].min()
    end_test_date = df_final_predict['open_time'].max()

    print('Min Data: ', start_test_date)
    print('Max Data: ', start_test_date)
    simule_trading_crypto(df_final_predict, start_test_date, end_test_date, start_value, stop_loss)


def main(args):
    try:
        estimator = myenv.estimator
        symbol = 'BTCUSDT'
        start_train_date = '2010-01-01'
        start_test_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        numeric_features = myenv.data_numeric_fields
        stop_loss = myenv.stop_loss
        regression_times = myenv.regression_times
        regression_profit_and_loss = myenv.times_regression_profit_and_loss
        calc_rsi = False
        compare_models = False
        n_jobs = myenv.n_jobs
        train_size = myenv.train_size
        use_gpu = False
        verbose = False
        normalize = False
        fold = 3
        simule_trading = False

        for arg in args:
            if (arg.startswith('-download-data')):
                sm.send_status_to_telegram('Iniciando MG Crypto Trader...')
                sm.send_status_to_telegram('Atualizando base de dados')
                download_data()
                sm.send_status_to_telegram('Base atualizada')
                sys.exit()

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

            if (arg.startswith('-stop-loss=')):
                stop_loss = float(arg.split('=')[1])

            if (arg.startswith('-regression-times=')):
                regression_times = int(arg.split('=')[1])

            if (arg.startswith('-regression-profit-and-loss=')):
                regression_profit_and_loss = float(arg.split('=')[1])

            if (arg.startswith('-n-jobs=')):
                n_jobs = int(arg.split('=')[1])

            if (arg.startswith('-train-size=')):
                train_size = float(arg.split('=')[1])

            if (arg.startswith('-fold=')):
                fold = int(arg.split('=')[1])

            if (arg.startswith('-calc-rsi')):
                calc_rsi = True

            if (arg.startswith('-all-cols')):
                aux = float_kline_cols + integer_kline_cols  # + ['close_time']
                numeric_features = aux

            if (arg.startswith('-copare-models')):
                compare_models = True

            if (arg.startswith('-use-gpu')):
                use_gpu = True

            if (arg.startswith('-normalize')):
                normalize = True

            if (arg.startswith('-verbose')):
                verbose = True

            if (arg.startswith('-simule-trading')):
                simule_trading = True

        if simule_trading:
            print(f'Iniciando simulação de trading Symbol: {symbol} - start_train_date: {start_train_date} - start_test_date: {start_test_date}')
            exec_simule_trading(symbol, calc_rsi, numeric_features, start_test_date, 100.0, stop_loss, estimator)
        else:
            sm.send_status_to_telegram(f'Iniciando Modelo Preditor para Symbol: {symbol}...')
            start_train_engine(symbol, estimator, train_size, start_train_date, start_test_date, numeric_features, stop_loss,
                               regression_times, regression_profit_and_loss, calc_rsi, compare_models, n_jobs, use_gpu, verbose, normalize, fold, args)
    except Exception as e:
        traceback.print_exc()
        sm.send_status_to_telegram('ERRO: ' + str(e))
        return False
    return True


if __name__ == '__main__':
    main(sys.argv[1:])
