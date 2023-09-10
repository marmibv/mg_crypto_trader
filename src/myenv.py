import sys

float_kline_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']

integer_kline_cols = ['number_of_trades', 'ignore']

date_kline_cols = ['open_time', 'close_time']

# must be that order
all_klines_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                   'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

all_cols = all_klines_cols + ['symbol']

data_numeric_fields = ['open', 'high', 'low', 'volume', 'close']
date_features = ['open_time']
use_cols = date_features + data_numeric_fields
numeric_features = data_numeric_fields + ['rsi']
datadir = sys.path[0] + '/src/data'
modeldir = sys.path[0] + '/src/models'
label = 'status'
stop_loss = 2.0
regression_times = 24 * 30 * 2  # horas
times_regression_profit_and_loss = 24
currency = 'USDT'
n_jobs = -1
train_size = 0.7
estimator = 'xgboost'
symbol = 'BTCUSDT'

sleep_refresh = 2  # seconds
