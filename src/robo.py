import sys
# sys.path.append('../')

from src.utils import *
from src.calcEMA import calc_RSI
from src.myenv import *
import src.send_message as sm

from binance.client import Client
from pycaret.classification import ClassificationExperiment
from sklearn.model_selection import train_test_split

import pandas as pd
import datetime

import time
import traceback


def get_model_name(symbol, estimator='xgboost'):
    '''
    return: Last model file stored in MODELS_DIR or None if not exists. Max 999 models per symbol
    '''
    model_name = None
    for i in range(999, 0, -1):
        model_name = f'{symbol}_{estimator}_SL_{stop_loss}_RT_{regression_times}_RPL_{regression_profit_and_loss}_{i}'
        if os.path.exists(f'{model_name}.pkl'):
            break
    return model_name


def save_model(symbol, model, experiment, estimator='xgboost'):
    for i in range(1, 999):
        model_name = f'{symbol}_{estimator}_SL_{stop_loss}_RT_{regression_times}_RPL_{regression_profit_and_loss}_{i}'
        if os.path.exists(f'{model_name}.pkl'):
            continue
        else:
            print('Model file name: ', model_name)
            experiment.save_model(model, model_name)
            break


def load_model(symbol, estimator='xgboost'):
    ca = ClassificationExperiment()
    model_name = get_model_name(symbol, estimator)
    print('Loading model:', model_name)
    model = ca.load_model(model_name)
    # model = ca.load_model('xgboost_SL_2.0_RT_720_RPL_24_1')
    return ca, model


def regresstion_times(df_database, numeric_features=['close'], regression_times=24 * 30):
    new_numeric_features = []
    new_numeric_features.append(numeric_features)
    for nf in numeric_features:
        for i in range(1, regression_times + 1):
            col = nf + "_" + str(i)
            df_database[col] = df_database[nf].shift(i)
            new_numeric_features.append(col)

    df_database.dropna(inplace=True)
    df_database = df_database.round(2)
    return df_database, new_numeric_features


def get_max_date(df_database):
    max_date = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d')
    if df_database is not None:
        max_date = pd.to_datetime(df_database['open_time'].max(), unit='ms')
    return max_date


def get_database(symbol, tail=-1):
    database_name = get_database_name(symbol)

    df_database = pd.DataFrame()
    if os.path.exists(database_name):
        df_database = pd.read_csv(database_name, sep=';', parse_dates=date_features, date_parser=date_parser, decimal='.')
    if tail > 0:
        df_database = df_database.tail(tail).copy()
    return df_database


def get_database_name(symbol):
    return datadir + '/' + symbol + '/' + symbol + '.csv'


def download_data(save_database=True):
    symbols = pd.read_csv(datadir + '/symbol_list.csv')
    for symbol in symbols['symbol']:
        get_data(symbol + currency, save_database)


def get_klines(symbol, interval='1h', max_date='2010-01-01', limit=1000):
    client = Client()
    klines = client.get_historical_klines(symbol=symbol, interval=interval, start_str=max_date, limit=limit)
    df_klines = pd.DataFrame(data=klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                   'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    return df_klines


def get_data(symbol, save_database=True, interval='1h', tail=-1):
    database_name = get_database_name(symbol)

    # print(f'Database name: {database_name}')
    df_database = get_database(symbol, tail)
    # print(f'Database df:\n {df_database}')
    max_date = get_max_date(df_database)
    # print(f'Max date: {max_date}')

    print('Downloading data for symbol: ' + symbol)
    # Interval for Kline data (e.g., '1h' for 1-hour candles)
    while (max_date < datetime.datetime.now()):
        # Fetch Kline data
        print('Max date: ', max_date)
        df_klines = get_klines(symbol, interval=interval, max_date=max_date.strftime('%Y-%m-%d'))
        df_database = pd.concat([df_klines, df_database])
        df_database = df_database.sort_values('open_time')
        df_database['symbol'] = symbol

        max_date = get_max_date(df_database)

        if save_database:
            if not os.path.exists(database_name):
                os.makedirs(database_name)
            df_database.to_csv(database_name, sep=';', index=False)
            print('Database updated at ', database_name)
    return df_database


def send_message(df_predict):
    message = f'Operação: {df_predict["prediction_label"].values[0]} - Valor Atual: {df_predict["close"].values[0]}'
    sm.send_to_telegram(message)
    print(message)


def start_predict_engine(symbol, tail=-1, numeric_features=['close', 'rsi'], regression_times=24 * 30):
    experiment, model = load_model(symbol)  # cassification_experiment
    df = get_data(symbol, save_database=False, interval='1h', tail=tail)
    df.index = df['open_time']
    while True:
        # df = get_data(symbol, save_database=False, interval='1h', tail=tail)
        _df = get_klines(symbol, interval='1h', max_date=None, limit=1)
        _df.index = _df['open_time']

        df = calc_RSI(df)
        df, _ = regresstion_times(df, numeric_features, regression_times)
        df_predict = experiment.predict_model(model, df.tail(1))
        operacao = df_predict['prediction_label'].values[0]
        if (operacao.startswith('SOBE') or operacao.startswith('CAI')):
            send_message(df_predict)
        time.sleep(2)


def main(args):
    # while True:
    try:
        symbol = 'BTCUSDT'
        if (len(args) > 1 and args[1:] == '-download_data'):
            sm.send_to_telegram('Iniciando MG Crypto Trader...')
            sm.send_to_telegram('Atualizando base de dados')
            download_data()
            sm.send_to_telegram('Base atualizada')
            sm.send_to_telegram(f'Iniciando Modelo Preditor para Symbol: {symbol}...')
        start_predict_engine(symbol)
    except Exception as e:
        sm.send_to_telegram('ERRO: ' + str(e))
        traceback.print_exc()
    # finally:
    #    time.sleep(60)
    #    continue


if __name__ == '__main__':
    main(sys.argv[1:])
