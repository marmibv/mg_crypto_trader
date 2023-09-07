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


def start_predict_engine(symbol, tail=-1, numeric_features=['close', 'rsi'], regression_times=24 * 30, trace=False):
    experiment, model = load_model(symbol)  # cassification_experiment
    df = get_data(symbol, save_database=False, interval='1h', tail=tail, adjust_index=True)
    df.to_csv('log_after_1st_get_data.csv', sep=';', index=False) if trace else None

    df = calc_RSI(df)
    df.dropna(inplace=True)

    df.to_csv('log_after_1st_calc_RSI.csv', sep=';', index=False) if trace else None
    df, _ = regresstion_times(df, numeric_features + ['rsi'], regression_times)
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
        if cont_aviso > 1000:
            sm.send_to_telegram('Ainda trabalhando...')
            cont_aviso = 0


def main(args):
    while True:
        try:
            symbol = 'BTCUSDT'
            for arg in args:
                if (arg.startswith('-download_data')):
                    sm.send_to_telegram('Iniciando MG Crypto Trader...')
                    sm.send_to_telegram('Atualizando base de dados')
                    download_data()
                    sm.send_to_telegram('Base atualizada')
            for arg in args:
                if (arg.startswith('-symbol=')):
                    symbol = arg.split('=')[1]

            sm.send_to_telegram(f'Iniciando Modelo Preditor para Symbol: {symbol}...')
            start_predict_engine(symbol, numeric_features=data_numeric_fields, tail=regression_times + 50, regression_times=regression_times)
        except Exception as e:
            sm.send_to_telegram('ERRO: ' + str(e))
            traceback.print_exc()
        finally:
            gc.collect()
            time.sleep(60)
            continue


if __name__ == '__main__':
    main(sys.argv[1:])
