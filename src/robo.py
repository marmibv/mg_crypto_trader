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

use_cols = ['open_time', 'close']


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
            print('save_model: Model file name: ', model_name)
            experiment.save_model(model, model_name)
            break


def load_model(symbol, estimator='xgboost'):
    ca = ClassificationExperiment()
    model_name = get_model_name(symbol, estimator)
    print('load_model: Loading model:', model_name)
    model = ca.load_model(model_name, verbose=False)
    # model = ca.load_model('xgboost_SL_2.0_RT_720_RPL_24_1')
    return ca, model


def regresstion_times(df_database, numeric_features=['close'], regression_times=24 * 30, last_one=False):
    count = df_database.shape[0]
    new_numeric_features = []
    if last_one:
        col_ant = ''
        col_atual = ''
        for nf in numeric_features:
            for i in range(1, regression_times + 1):
                if i == 1:
                    col_ant = nf
                    col_atual = nf + "_" + str(i)
                elif i == regression_times:
                    continue
                else:
                    col_ant = nf + "_" + str(i)
                    col_atual = nf + "_" + str(i + 1)
                df_database.iloc[count:count + 1][col_atual] = df_database.iloc[count - 1:count][col_ant]
    else:
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


def get_database(symbol, tail=-1, adjust_index=False, columns=['open_time', 'close']):
    database_name = get_database_name(symbol)
    print('get_database: name: ', database_name)

    df_database = pd.DataFrame()
    if os.path.exists(database_name):
        df_database = pd.read_csv(database_name, sep=';', parse_dates=date_features, date_parser=date_parser, decimal='.')
        if adjust_index:
            df_database.index = df_database['open_time']
            df_database.index.name = 'ix_open_time'

        df_database = df_database[use_cols]
    if tail > 0:
        df_database = df_database.tail(tail).copy()
    print(f'get_database: count_rows: {df_database.shape[0]} - symbol: {symbol} - tail: {tail} - adjust_index: {adjust_index}')

    print(f'get_database: duplicated: {df_database.duplicated().sum()}')
    return df_database


def get_database_name(symbol):
    return datadir + '/' + symbol + '/' + symbol + '.csv'


def download_data(save_database=True):
    symbols = pd.read_csv(datadir + '/symbol_list.csv')
    for symbol in symbols['symbol']:
        get_data(symbol + currency, save_database)


def get_klines(symbol, interval='1h', max_date='2010-01-01', limit=1000, adjust_index=False,
               columns=['open_time', 'close']):
    _all_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    client = Client()
    klines = client.get_historical_klines(symbol=symbol, interval=interval, start_str=max_date, limit=limit)
    df_klines = pd.DataFrame(data=klines, columns=_all_cols)[columns]
    df_klines['symbol'] = symbol
    df_klines['open_time'] = pd.to_datetime(df_klines['open_time'], unit='ms')

    for col in ['open', 'high', 'low', 'close', 'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
        if col in df_klines.columns:
            df_klines[col] = df_klines[col].astype(float)

    if adjust_index:
        df_klines.index = df_klines['open_time']
        df_klines.index.name = 'ix_open_time'

    print(f'get_klines: count: {df_klines.shape[0]} - max_date: {max_date} - limit: {limit} - adjust_index: {adjust_index}')
    return df_klines


def get_data(symbol, save_database=True, interval='1h', tail=-1, adjust_index=False):
    database_name = get_database_name(symbol)
    df_database = get_database(symbol, tail, adjust_index=adjust_index)
    max_date = get_max_date(df_database)

    print('get_data: Downloading data for symbol: ' + symbol)
    while (max_date < datetime.datetime.now()):
        print('get_data: Max date: ', max_date)
        df_klines = get_klines(symbol, interval=interval, max_date=max_date.strftime('%Y-%m-%d'), adjust_index=adjust_index)

        df_database = pd.concat([df_database, df_klines]).drop_duplicates(keep='last')
        df_database['symbol'] = symbol
        max_date = get_max_date(df_database)
        if save_database:
            if not os.path.exists(database_name):
                os.makedirs(database_name)
            df_database.to_csv(database_name, sep=';', index=False)
            print('get_data: Database updated at ', database_name)
    return df_database


def send_message(df_predict):
    message = f'Ticker: {df_predict["symbol"].values[0]} - Operação: {df_predict["prediction_label"].values[0]} - Valor Atual: {df_predict["close"].values[0]}'
    sm.send_to_telegram(message)
    print('send_message:', message)


def start_predict_engine(symbol, tail=-1, numeric_features=['close', 'rsi'], regression_times=24 * 30, trace=False):
    experiment, model = load_model(symbol)  # cassification_experiment
    df = get_data(symbol, save_database=False, interval='1h', tail=tail, adjust_index=True)
    df.to_csv('log_after_1st_get_data.csv', sep=';', index=False) if trace else None
    df = calc_RSI(df)
    df.dropna(inplace=True)
    df.to_csv('log_after_1st_calc_RSI.csv', sep=';', index=False) if trace else None
    df, _ = regresstion_times(df, numeric_features, regression_times)
    df.to_csv('log_after_1st_regresstion_times.csv', sep=';', index=False) if trace else None
    print('start_predict_engine: Info after regresstion_times: ', df.info())

    cont = 0
    cont_aviso = 0
    while True:
        # df = get_data(symbol, save_database=False, interval='1h', tail=tail)
        _df = get_klines(symbol, interval='1h', max_date=None, limit=1, adjust_index=True)
        _df.to_csv(f'log_after_{cont}_get_klines.csv', sep=';', index=False) if trace else None

        if _df.index.isin(df.index):
            df.update(_df)
        else:
            df = pd.concat([df, _df])
        df.to_csv(f'log_after_{cont}_update.csv', sep=';', index=False) if trace else None

        df = calc_RSI(df, last_one=True)
        df.to_csv(f'log_after_{cont}_calc_RSI.csv', sep=';', index=False) if trace else None

        df, _ = regresstion_times(df, numeric_features, regression_times, last_one=True)
        df.to_csv(f'log_after_{cont}_regresstion_times.csv', sep=';', index=False) if trace else None

        df[['open_time'] + numeric_features].to_csv('log_experiment_data.log', sep=';', index=False) if trace else None

        df_predict = experiment.predict_model(model, df.tail(1), verbose=False)
        df_predict.to_csv(f'log_after_{cont}_df_predict.csv', sep=';', index=False) if trace else None

        operacao = df_predict['prediction_label'].values[0]

        msg = f'Last Data: open_time: {df_predict.tail(1)["open_time"].values[0]} - close: {df_predict.tail(1)["close"].values[0]} - rsi: {df_predict.tail(1)["rsi"].values[0]} - prediction_label: {operacao}'
        print(msg)

        if (operacao.startswith('SOBE') or operacao.startswith('CAI')):
            send_message(df_predict)
        time.sleep(sleep_refresh)
        cont += 1
        cont_aviso += 1
        if cont_aviso > 100:
            sm.send_to_telegram('Ainda trabalhando...')
            cont_aviso = 0


def main(args):
    while True:
        try:
            symbol = 'BTCUSDT'
            if (len(args) > 1 and args[1:] == '-download_data'):
                sm.send_to_telegram('Iniciando MG Crypto Trader...')
                sm.send_to_telegram('Atualizando base de dados')
                download_data()
                sm.send_to_telegram('Base atualizada')
            sm.send_to_telegram(f'Iniciando Modelo Preditor para Symbol: {symbol}...')
            start_predict_engine(symbol, tail=regression_times + 14, regression_times=regression_times)
        except Exception as e:
            sm.send_to_telegram('ERRO: ' + str(e))
            traceback.print_exc()
        finally:
            gc.collect()
            time.sleep(60)
            continue


if __name__ == '__main__':
    main(sys.argv[1:])
