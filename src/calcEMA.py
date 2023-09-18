import datetime
import pandas as pd
import numpy as np
import traceback
import gc

valid_times = ['1mo', '2mo', '3mo', '6mo', '1y', '2y', '3y', '4y', '5y']


def run_calc_emas(df: pd.DataFrame, column='close', times=False, periods=True) -> pd.DataFrame:
    _df = df.copy()
    # Calc EMA for valid_times
    if times:
        for _time in valid_times:
            _df = calc_ema_days(_df, _time, close_price=column)

    if periods:
        _df = calc_ema_periods(_df, [200], close_price=column, diff_price=True)
    # Calc RSI
    _df = calc_RSI(_df, column)
    gc.collect()
    return _df

    # df.to_csv(s['symbol'] + '.csv', sep=';', decimal=',')
    # print(df)


def calc_ema_days(df: pd.DataFrame, period_of_time: str, close_price='close') -> pd.DataFrame:
    days = 0
    match period_of_time:
        case '1mo':
            days = 30
        case '2mo':
            days = 60
        case '3mo':
            days = 90
        case '6mo':
            days = 180
        case '1y':
            days = 365
        case '2y':
            days = 365 * 2
        case '3y':
            days = 365 * 3
        case '4y':
            days = 365 * 4
        case '5y':
            days = 365 * 5
        case defaul:
            raise Exception('Infome um período válido> ' + valid_times)

    # print('Days to calc> ', days)

    start_date = df.index.max() - datetime.timedelta(days=days)
    end_date = df.index.max()
    min_date = df.index.min()
    # print('min_data> {}; start_date> {}; end_date> {}'.format(min_date, start_date, end_date))
    mme_price = "ema_" + period_of_time
    diff_price = "ema_" + period_of_time + "_diff"
    if min_date <= start_date:
        mask = (df.index > start_date) & (df.index <= end_date)
        # print('Mask> ', mask)
        count_occours = df.loc[mask].shape[0]
        # print('Tamanho DF> ', count_occours)
        if count_occours == 0:
            print(
                'calc_ema_days: Não foi encontrado registros no período informado> ' + period_of_time)
            df[mme_price] = None
        else:
            df[mme_price] = df[close_price].ewm(
                span=count_occours, min_periods=count_occours).mean()
            df[diff_price] = round(
                ((df[close_price] - df[mme_price]) / df[mme_price]) * 100, 2)
    else:
        df[mme_price] = None
        df[diff_price] = None

    return df


def calc_ema_periods(df: pd.DataFrame, periods_of_time: any, close_price='close', diff_price=True) -> pd.DataFrame:
    count_occours = df.shape[0]
   # print('Tamanho DF> ', count_occours)

    for periods in periods_of_time:
        mme_price = "ema_" + str(periods) + 'p'
        s_diff_price = mme_price + "_diff"
        if periods > count_occours:
            print('calc_ema_periods: Não foi encontrado registros no período informado> ' + str(periods))
            df[mme_price] = None
            if diff_price:
                df[s_diff_price] = None
        else:
            df[mme_price] = df[close_price].ewm(
                span=periods, min_periods=periods).mean()
            if diff_price:
                df[s_diff_price] = round(
                    ((df[close_price] - df[mme_price]) / df[mme_price]) * 100, 2)
    return df


def calc_RSI(df: pd.DataFrame, close_price='close', window=14, fillna=False, last_one=False) -> pd.DataFrame:
    count = df.shape[0]
    if count > 0:
        if last_one:
            try:
                _df = df[[close_price]].tail(window + 100).copy()
                # _df = df[[close_price]].copy()
                _count = _df.shape[0]
                _df['change'] = _df[close_price].diff()
                _df['gain'] = _df.change.mask(_df.change < 0, 0.0)
                _df['loss'] = -_df.change.mask(_df.change > 0, -0.0)
                _df['avg_gain'] = rma(_df.gain.to_numpy(), window)
                _df['avg_loss'] = rma(_df.loss.to_numpy(), window)

                _df['rs'] = _df.avg_gain / _df.avg_loss
                _df['rsi'] = 100 - (100 / (1 + _df.rs))
                df.update(_df.iloc[_count - 1:_count]['rsi'].round(2))
            except Exception as error:
                print('calc_RSI: last_one=TRUE: Erro no calculo do RSI!', error)
                print(df)
                print(df.info())
                #traceback.print_exc()
        else:
            try:
                df['change'] = df[close_price].diff()
                df['gain'] = df.change.mask(df.change < 0, 0.0)
                df['loss'] = -df.change.mask(df.change > 0, -0.0)
                df['avg_gain'] = rma(df.gain.to_numpy(), window)
                df['avg_loss'] = rma(df.loss.to_numpy(), window)

                df['rs'] = df.avg_gain / df.avg_loss
                df['rsi'] = (100 - (100 / (1 + df.rs))).round(2)

                if fillna:
                    df['rsi'].fillna(0.0, inplace=True)
            except Exception as error:
                print('calc_RSI: Erro no calculo do RSI!', error)
                print(df)
                print(df.info())
                #traceback.print_exc()
            finally:
                df.drop(columns=['change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], inplace=True, errors='ignore')

    return df


def rma(x, n):
    """Running moving average"""
    a = np.full_like(x, np.nan)
    a[n] = x[1:n + 1].mean()
    for i in range(n + 1, len(x)):
        a[i] = (a[i - 1] * (n - 1) + x[i]) / n
    return a
