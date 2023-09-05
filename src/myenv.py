all_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore', 'symbol']

data_numeric_fields = ['open', 'high', 'low', 'volume', 'close']
date_features = ['open_time']
use_cols = date_features + data_numeric_fields
numeric_features = data_numeric_fields + ['rsi']
datadir = './src/data'
modeldir = './src/models'
label = 'status'
stop_loss = 2.0
regression_times = 24 * 30  # horas
regression_profit_and_loss = 24
currency = 'USDT'
