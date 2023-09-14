from src.myenv import *

import src.send_message as sm
import src.myenv as myenv

from pycaret.regression.oop import RegressionExperiment
from pycaret.classification.oop import ClassificationExperiment
from binance.client import Client

import datetime
import os
import pandas as pd
import plotly.express as px
import gc
import traceback


def increment_time(interval='1h'):
    match(interval):
        case '1min':
            return pd.Timedelta(minutes=1)
        case '3min':
            return pd.Timedelta(minutes=3)
        case '5min':
            return pd.Timedelta(minutes=5)
        case '15min':
            return pd.Timedelta(minutes=15)
        case '30min':
            return pd.Timedelta(minutes=30)
        case '1h':
            return pd.Timedelta(hours=1)
        case '2h':
            return pd.Timedelta(hours=2)
        case '4h':
            return pd.Timedelta(hours=4)
        case '6h':
            return pd.Timedelta(hours=6)
        case '8h':
            return pd.Timedelta(hours=8)
        case '12h':
            return pd.Timedelta(hours=12)
        case '1d':
            return pd.Timedelta(days=1)
        case '3d':
            return pd.Timedelta(days=3)
        case '1w':
            return pd.Timedelta(weeks=1)
        case '1M':
            return pd.Timedelta(days=30)


def date_parser(x):
    return pd.to_datetime(x, unit='ms')


def read_data(dir, sep=';', all_cols=None, use_cols=use_cols) -> pd.DataFrame:
    filenames = []

    for file in os.listdir(dir):
        if file.endswith(".csv"):
            filenames.append(os.path.join(dir, file))

    parse_dates = ['open_time']
    dataframes = []

    for filename in filenames:
        print('read_data: Start reading file: ', filename)
        df = pd.read_csv(filename, names=all_cols, parse_dates=parse_dates,
                         date_parser=date_parser, sep=sep, decimal='.', usecols=use_cols)
        dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.sort_values(['open_time'], inplace=True)
    combined_df.reset_index(inplace=True, drop=True)
    return combined_df


def rotate_label(df, rows_to_rotate=-1, label='label_shifted', dropna=False):
    new_label = label + '_' + str(rows_to_rotate)
    df[new_label] = df[label].shift(rows_to_rotate)
    if dropna:
        df.dropna(inplace=True)

    return new_label, df


def setup_regression_model(
        data: pd.DataFrame,
        label: str,
        train_size=0.7,
        numeric_features=['open', 'high', 'low', 'volume', 'close', 'rsi'],
        date_features=['open_time'],
        use_gpu=False,
        regressor_estimator='lr',
        apply_best_analisys=False,
        fold=3,
        sort='MAE',
        verbose=False) -> [RegressionExperiment, any]:

    re = RegressionExperiment()

    setup = re.setup(data,
                     train_size=train_size,
                     target=label,
                     numeric_features=numeric_features,
                     date_features=date_features,
                     create_date_columns=["hour", "day", "month"],
                     fold_strategy='timeseries',
                     fold=fold,
                     session_id=123,
                     normalize=True,
                     use_gpu=use_gpu,
                     verbose=verbose,
                     )
    best = regressor_estimator
    if apply_best_analisys:
        print('setup_model: Applying best analisys...') if verbose else None
        best = setup.compare_models(sort=sort, verbose=True, exclude=['lightgbm'])

    print(f'setup_model: Creating model Best: [{best}]') if verbose else None
    model = setup.create_model(best, verbose=False)
    model_name_file = str(model)[0:str(model).find('(')] + '_' + label
    print(f'setup_model: Saving model {model_name_file}') if verbose else None
    setup.save_model(model, model_name_file)

    return setup, model


def predict(setup: RegressionExperiment,
            model: any,
            predict_data: pd.DataFrame = None,
            numeric_features=['open', 'high', 'low', 'volume', 'close', 'rsi'],
            date_features=['open_time'],
            verbose=False) -> RegressionExperiment:

    print('predict: predict.setup: \n', setup) if verbose else None
    print('predict: predict.model: \n', model) if verbose else None
    print('predict: predict.predict_data: \n', predict_data) if verbose else None
    print('predict: predict.numeric_features: \n', numeric_features) if verbose else None
    print('predict: predict.date_features: \n', date_features) if verbose else None

    predict = None
    if predict_data is None:
        predict = setup.predict_model(model, verbose=verbose)
    else:
        predict = setup.predict_model(model, data=predict_data[date_features + numeric_features], verbose=verbose)

    return predict


def forecast(data: pd.DataFrame,
             fh: int = 1,
             train_size=0.7,
             interval='1h',
             numeric_features=['open', 'high', 'low', 'volume', 'close', 'rsi'],
             date_features=['open_time'],
             regressor_estimator='lr',
             apply_best_analisys=False,
             use_gpu=False,
             fold=3,
             ):
    list_models = {}

    _data = data.copy()
    test_data = data.tail(1).copy().reset_index(drop=True)
    print('forecast: numeric_features: ', numeric_features)

    open_time = test_data['open_time']
    df_result = pd.DataFrame()
    for i in range(1, fh + 1):
        df_predict = pd.DataFrame()
        open_time = open_time + increment_time(interval)
        df_predict['open_time'] = open_time
        print(f'forecast: Applying predict No: {i} for open_time: {df_predict["open_time"].values}')

        for label in numeric_features:
            if label not in list_models:
                print('forecast: Training model for label:', label)
                target, train_data = rotate_label(_data, -1, label, True)
                setup, model = setup_regression_model(train_data, target, train_size=train_size, fold=fold,
                                                      regressor_estimator=regressor_estimator, use_gpu=use_gpu, apply_best_analisys=apply_best_analisys)
                train_data.drop(columns=target, inplace=True)
                list_models[label] = {'setup': setup, 'model': model}
                print('forecast: Training model Done!')

            _setup = list_models[label]['setup']
            _model = list_models[label]['model']

            df = predict(_setup,
                         _model,
                         test_data if i == 1 else df_result.tail(1).copy(),
                         numeric_features,
                         date_features)

            print('forecast: Label:', label, 'Predict Label:', df['prediction_label'].values[0])
            df_predict[label] = df['prediction_label']
            gc.collect()

        df_result = pd.concat([df_result, df_predict], axis=0)
        gc.collect()

    return df_result.sort_values('open_time').reset_index(drop=True)


def shift_test_data(predict_data: pd.DataFrame, label='close', columns=[], verbose=False):
    print('forecast: Shifting: \n', predict_data.tail(1)[columns]) if verbose else None
    _test_data = predict_data[columns].tail(1).copy().shift(1, axis='columns')
    _test_data.drop(columns=label, inplace=True)
    _test_data['open_time'] = predict_data['open_time']
    print('forecast: Shifted: \n', _test_data.tail(1)) if verbose else None
    return _test_data


def forecast2(data: pd.DataFrame,
              label: str = 'close',
              fh: int = 1,
              train_size=0.7,
              interval='1h',
              numeric_features=['open', 'high', 'low', 'volume', 'close', 'rsi'],
              date_features=['open_time'],
              regressor_estimator='lr',
              apply_best_analisys=False,
              use_gpu=False,
              fold=3,
              regression_times=1,
              sort='MAE',
              verbose=False,
              ):

    _data = data.copy()
    for i in range(1, regression_times + 1):
        _label, _data = rotate_label(_data, i, label)
        numeric_features.append(_label)
    _data.dropna(inplace=True)

    print('forecast: numeric_features: ', numeric_features) if verbose else None

    open_time = data.tail(1)['open_time']
    df_result = pd.DataFrame()
    setup = None
    model = None
    for i in range(1, fh + 1):
        if model is None:
            print('forecast: Training model for label:', label) if verbose else None
            setup, model = setup_regression_model(_data, label, train_size, numeric_features, date_features,
                                                  use_gpu, regressor_estimator, apply_best_analisys, fold, sort, verbose)
            print('forecast: Training model Done!') if verbose else None

        open_time = open_time + increment_time(interval)
        print(f'forecast: Applying predict No: {i} for open_time: {open_time}') if verbose else None
        predict_data = shift_test_data(_data.tail(1).copy() if i == 1 else df_result.tail(1).copy(), label=label, columns=[label] + numeric_features)
        predict_data['open_time'] = open_time

        df_predict = predict(setup, model, predict_data, numeric_features, date_features, verbose)
        df_predict['close'] = df_predict['prediction_label']

        gc.collect()

        df_result = pd.concat([df_result, df_predict], axis=0)
        gc.collect()

    return df_result.sort_values('open_time').reset_index(drop=True), model, setup


def calc_diff(predict_data, validation_data, regressor):
    start_date = predict_data["open_time"].min()  # strftime("%Y-%m-%d")
    end_date = predict_data["open_time"].max()  # .strftime("%Y-%m-%d")
    # now = datetime.now().strftime("%Y-%m-%d")

    predict_data.index = predict_data['open_time']
    validation_data.index = validation_data['open_time']

    filtered_data = validation_data.loc[(validation_data['open_time'] >= start_date) & (validation_data['open_time'] <= end_date)].copy()
    filtered_data['prediction_label'] = predict_data['prediction_label']
    filtered_data['diff'] = ((filtered_data['close'] - filtered_data['prediction_label']) / filtered_data['close']) * 100
    filtered_data.drop(columns=['open_time'], inplace=True)
    filtered_data.round(2)
    return filtered_data


def plot_predic_model(predict_data, validation_data, regressor):
    start_date = predict_data["open_time"].min()  # strftime("%Y-%m-%d")
    end_date = predict_data["open_time"].max()  # .strftime("%Y-%m-%d")

    filtered_data = calc_diff(predict_data, validation_data, regressor)

    fig1 = px.line(filtered_data, x=filtered_data.index, y=['close', 'prediction_label'], template='plotly_dark', range_x=[start_date, end_date])
    fig2 = px.line(filtered_data, x=filtered_data.index, y=['diff'], template='plotly_dark', range_x=[start_date, end_date])
    fig1.show()
    fig2.show()
    return filtered_data


def get_model_name(symbol, estimator=myenv.estimator, stop_loss=myenv.stop_loss, regression_times=myenv.regression_times, times_regression_profit_and_loss=myenv.times_regression_profit_and_loss):
    '''
    return: Last model file stored in MODELS_DIR or None if not exists. Max 999 models per symbol
    '''
    model_name = None
    for i in range(999, 0, -1):
        model_name = f'{symbol}_{estimator}_SL_{stop_loss}_RT_{regression_times}_RPL_{times_regression_profit_and_loss}_{i}'
        if os.path.exists(f'{model_name}.pkl'):
            break
    return model_name


def save_model(symbol, model, experiment, estimator='xgboost', stop_loss=myenv.stop_loss, regression_times=myenv.regression_times, times_regression_profit_and_loss=myenv.times_regression_profit_and_loss):
    model_name = ''
    for i in range(1, 999):
        model_name = f'{symbol}_{estimator}_SL_{stop_loss}_RT_{regression_times}_RPL_{times_regression_profit_and_loss}_{i}'
        if os.path.exists(f'{model_name}.pkl'):
            continue
        else:
            print('save_model: Model file name: ', model_name)
            experiment.save_model(model, model_name)
            break
    return model_name


def load_model(symbol, estimator=myenv.estimator, stop_loss=myenv.stop_loss, regression_times=myenv.regression_times, times_regression_profit_and_loss=myenv.times_regression_profit_and_loss):
    ca = ClassificationExperiment()
    model_name = get_model_name(symbol, estimator, stop_loss, regression_times, times_regression_profit_and_loss)
    print('load_model: Loading model:', model_name)
    model = ca.load_model(model_name, verbose=False)
    print('load_model: Model obj:', model)

    return ca, model


def regresstion_times(df_database, regression_features=['close'], regression_times=24 * 30, last_one=False):
    print('regresstion_times: regression_features: ', regression_features)
    count = df_database.shape[0]
    features_added = []
    if last_one:
        col_ant = ''
        col_atual = ''
        for nf in regression_features:
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
        # features_added.append(regression_features)
        for nf in regression_features:
            for i in range(1, regression_times + 1):
                col = nf + "_" + str(i)
                df_database[col] = df_database[nf].shift(i)
                features_added.append(col)

        df_database.dropna(inplace=True)
        df_database = df_database.round(2)
    return df_database, features_added


def get_max_date(df_database):
    max_date = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d')
    if df_database is not None and df_database.shape[0] > 0:
        max_date = pd.to_datetime(df_database['open_time'].max(), unit='ms')
    return max_date


def get_database(symbol, tail=-1, columns=['open_time', 'close'], parse_data=True):
    database_name = get_database_name(symbol)
    print('get_database: name: ', database_name)

    df_database = pd.DataFrame()
    print('get_database: columns: ', columns)
    if os.path.exists(database_name):
        if parse_data:
            df_database = pd.read_csv(database_name, sep=';', parse_dates=date_features, date_parser=date_parser, decimal='.', usecols=columns)
            df_database = parse_type_fields(df_database)
        else:
            df_database = pd.read_csv(database_name, sep=';', decimal='.', usecols=columns)
        df_database = adjust_index(df_database)
        df_database = df_database[columns]
    if tail > 0:
        df_database = df_database.tail(tail).copy()
    print(f'get_database: count_rows: {df_database.shape[0]} - symbol: {symbol} - tail: {tail}')
    print(f'get_database: duplicated: {df_database.index.duplicated().sum()}')
    return df_database


def get_database_name(symbol):
    return datadir + '/' + symbol + '/' + symbol + '.csv'


def download_data(save_database=True, parse_data=False):
    symbols = pd.read_csv(datadir + '/symbol_list.csv')
    for symbol in symbols['symbol']:
        get_data(symbol + currency, save_database=save_database, columns=myenv.all_klines_cols, parse_data=parse_data)


def adjust_index(df):
    df.drop_duplicates(keep='last', subset=['open_time'], inplace=True)
    df.index = df['open_time']
    df.index.name = 'ix_open_time'
    df.sort_index(inplace=True)
    return df


def get_klines(symbol, interval='1h', max_date='2010-01-01', limit=1000, columns=['open_time', 'close'], parse_data=True):
    start_time = datetime.datetime.now()
    client = Client()
    klines = client.get_historical_klines(symbol=symbol, interval=interval, start_str=max_date, limit=limit)
    if 'symbol' in columns:
        columns.remove('symbol')
    # print('get_klines: columns: ', columns)
    df_klines = pd.DataFrame(data=klines, columns=all_klines_cols)[columns]
    if parse_data:
        df_klines = parse_type_fields(df_klines, parse_dates=True)
    df_klines = adjust_index(df_klines)
    delta = datetime.datetime.now() - start_time
    # Print the delta time in days, hours, minutes, and seconds
    print(f"get_klines: shape: {df_klines.shape} - Delta time: {delta.seconds % 60} seconds")
    return df_klines


def parse_type_fields(df, parse_dates=False):
    try:
        if 'symbol' in df.columns:
            df['symbol'] = pd.Categorical(df['symbol'])

        for col in float_kline_cols:
            if col in df.columns:
                df[col] = df[col].astype('float32')

        for col in integer_kline_cols:
            if col in df.columns:
                df[col] = df[col].astype('int16')

        if parse_dates:
            for col in date_features:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], unit='ms')

    except Exception as e:
        print(e)
        print(df)
        traceback.print_exc()
    return df


def get_data(symbol, save_database=False, interval='1h', tail=-1, columns=['open_time', 'close'], parse_data=True):
    database_name = get_database_name(symbol)
    df_database = get_database(symbol, tail, columns=columns, parse_data=parse_data)

    max_date = get_max_date(df_database)
    max_date_aux = ''
    print(f'get_data: Downloading data for symbol: {symbol} - max_date: {max_date}')
    while (max_date != max_date_aux):
        print(f'get_data: max_date: {max_date} - max_date_aux: {max_date_aux}')
        max_date_aux = get_max_date(df_database)
        print('get_data: Max date database: ', max_date_aux)

        df_klines = get_klines(symbol, interval=interval, max_date=max_date_aux.strftime('%Y-%m-%d'), columns=columns, parse_data=parse_data)
        df_database = pd.concat([df_database, df_klines])
        df_database.drop_duplicates(keep='last', subset=['open_time'], inplace=True)
        df_database.sort_index(inplace=True)
        df_database['symbol'] = symbol
        max_date = get_max_date(df_database)
        if save_database:
            if not os.path.exists(database_name.removesuffix(f'{symbol}.csv')):
                os.makedirs(database_name.removesuffix(f'{symbol}.csv'))
            df_database.to_csv(database_name, sep=';', index=False)
            print('get_data: Database updated at ', database_name)
    return df_database


def send_message(df_predict):
    message = f'Ticker: {df_predict["symbol"].values[0]} - Operação: {df_predict["prediction_label"].values[0]} - Valor Atual: {df_predict["close"].values[0]}'
    sm.send_to_telegram(message)
    print('send_message:', message)


def set_status_PL(row, stop_loss, max_regression_profit_and_loss, prefix_col_diff, strategy='SOBE_CAI'):
    for s in range(1, max_regression_profit_and_loss + 1):
        if (strategy == 'SOBE') or (strategy == 'SOBE_CAI'):
            if row[f'{prefix_col_diff}{s}'] >= stop_loss:
                return f'SOBE_{stop_loss}'
        if (strategy == 'CAI') or (strategy == 'SOBE_CAI'):
            if row[f'{prefix_col_diff}{s}'] <= -stop_loss:
                return f'CAI_{stop_loss}'
    return 'ESTAVEL'


def regression_PnL(data: pd.DataFrame, label: str, diff_percent: float, max_regression_profit_and_loss=6, drop_na=True, drop_calc_cols=True, strategy=None):
    col = 'c_'
    diff_col = 'd_'
    cols = []
    diff_cols = []
    for i in range(1, max_regression_profit_and_loss + 1):
        data[col + str(i)] = data['close'].shift(-i)
        data[diff_col + str(i)] = 100 * ((data[col + str(i)] - data['close']) / data['close'])
        cols.append(col + str(i))
        diff_cols.append(diff_col + str(i))

    if strategy == 'SOBE_CAI':
        data[label + '_sobe'] = data.apply(set_status_PL, axis=1, args=[diff_percent, max_regression_profit_and_loss, diff_col, 'SOBE'])
        data[label + '_cai'] = data.apply(set_status_PL, axis=1, args=[diff_percent, max_regression_profit_and_loss, diff_col, 'CAI'])
        data[label + '_sobe'] = pd.Categorical(data[label + '_sobe'])
        data[label + '_sobe'] = pd.Categorical(data[label + '_sobe'])
    else:
        data[label] = data.apply(set_status_PL, axis=1, args=[diff_percent, max_regression_profit_and_loss, diff_col, strategy])
        data[label] = pd.Categorical(data[label])

    if drop_na:
        data.dropna(inplace=True)
    if drop_calc_cols:
        data.drop(columns=cols + diff_cols, inplace=True)

    return data


def regress_until_diff(data: pd.DataFrame, diff_percent: float, max_regression_profit_and_loss=6, label: str = None):
    data['close_shift_x'] = 0.0
    data['diff_shift_x'] = 0.0
    data['shift_x'] = 0
    data[label] = 'ESTAVEL'
    for row_nu in range(1, data.shape[0]):
        diff = 0
        i = 1

        while (abs(diff) <= diff_percent):
            if (i > max_regression_profit_and_loss) or ((row_nu + i) >= data.shape[0]):
                break

            close = data.iloc[row_nu:row_nu + 1]['close'].values[0]
            close_px = data.iloc[row_nu + i:row_nu + i + 1]['close'].values[0]
            diff = -100 * (close - close_px) / close
            # print(f'ROW_NU: {row_nu} - regresssion_times: {i} - diff: {diff}')
            i += 1

        data['close_shift_x'].iloc[row_nu:row_nu + 1] = close_px
        data['diff_shift_x'].iloc[row_nu:row_nu + 1] = diff
        data['shift_x'].iloc[row_nu:row_nu + 1] = i - 1 if i == max_regression_profit_and_loss + 1 else i

        if diff >= diff_percent:
            data[label].iloc[row_nu:row_nu + 1] = 'SOBE_' + str(diff_percent)

        elif diff <= -diff_percent:
            data[label].iloc[row_nu:row_nu + 1] = 'CAI_' + str(diff_percent)

        # end for

    data.drop(columns=['close_shift_x', 'diff_shift_x', 'shift_x'], inplace=True)
    data[label] = pd.Categorical(data[label])

    return data


def simule_trading_crypto(df_predicted: pd.DataFrame, start_date, end_date, value: float, stop_loss=3.0, revert=False):
    _data = df_predicted.copy()
    _data.index = _data['open_time']
    _data = _data[(_data.index >= start_date) & (_data.index <= end_date)]
    saldo = value
    operacao = ''
    comprado = False
    valor_compra = 0
    valor_venda = 0
    diff = 0.0

    operacao_compra = ''
    for row_nu in range(1, _data.shape[0]):
        open_time = pd.to_datetime(_data.iloc[row_nu:row_nu + 1]['open_time'].values[0]).strftime("%Y-%m-%d %Hh")
        operacao = _data.iloc[row_nu:row_nu + 1]['prediction_label'].values[0]

        if (operacao.startswith('SOBE') or operacao.startswith('CAI')) and not comprado:
            operacao_compra = operacao
            valor_compra = round(_data.iloc[row_nu:row_nu + 1]['close'].values[0], 2)
            print(f'[{row_nu}][{operacao_compra}][{open_time}] => Compra: {valor_compra:.4f}')
            comprado = True

        if comprado:
            diff = 100 * (_data.iloc[row_nu:row_nu + 1]['close'].values[0] - valor_compra) / valor_compra
            # print(f'[{row_nu}][{operacao_compra}][{open_time}] Diff ==> {round(diff,2)}% - Comprado: {comprado}')

        if (abs(diff) >= stop_loss) and comprado:
            valor_venda = round(_data.iloc[row_nu:row_nu + 1]['close'].values[0], 2)
            if revert:
                if operacao_compra.startswith('SOBE'):
                    saldo -= round(saldo * (diff / 100), 2)
                else:
                    saldo -= round(saldo * (-diff / 100), 2)
            else:
                if operacao_compra.startswith('SOBE'):
                    saldo += round(saldo * (diff / 100), 2)
                else:
                    saldo += round(saldo * (-diff / 100), 2)

            print(f'[{row_nu}][{operacao_compra}][{open_time}] => Venda: {valor_venda:.4f} => Diff: {diff:.2f}% ==> PnL: $ {saldo:.2f}')
            comprado = False

    print(f'Saldo: {saldo}')
    return saldo
