import src.utils as utils
import src.myenv as myenv
import logging
import src.send_message as sm
import pandas as pd
import time
import src.calcEMA as calc_utils
import os


class RoboTrader():
  def __init__(self, params: dict):
    # Single arguments
    self._all_data = params['all_data'].copy()
    self._symbol = params['symbol']
    self._interval = params['interval']
    self._estimator = params['estimator']

    # List arguments
    self._start_date = params['start_date']
    self._numeric_features = params['numeric_features'].split(',')
    self._stop_loss = float(params['stop_loss'])
    self._regression_times = int(params['regression_times'])
    if self._regression_times > 0:
      self._regression_features = params['regression_features'].split(',')
    else:
      self._regression_features = []
    self._times_regression_profit_and_loss = int(params['times_regression_profit_and_loss'])
    # Boolean arguments
    self._calc_rsi = params['calc_rsi']
    self._verbose = params['verbose']
    self._arguments = params['arguments']

    # Internal atributes
    self._features_added = []
    self._experiement = None
    self._setup = None
    self._model = None
    self._model_name_init = ''

    # Prepare columns to kline
    self._kline_features = myenv.date_features + self._numeric_features
    self._all_features = []

    # Initialize logging
    self.log = self._configure_log(params['log_level'])
    sm.send_status_to_telegram(f'Starting Robo Trader')

  def _configure_log(self, log_level):

    log_file_path = os.path.join(myenv.logdir, f'robo_trader_{self._symbol}_{self._interval}_{self._estimator}.log')
    logger = logging.getLogger(f'robo_trader_{self._symbol}_{self._interval}_{self._estimator}')
    logger.propagate = False

    logger.setLevel(log_level)

    fh = logging.FileHandler(log_file_path, mode='a', delay=True)
    fh.setFormatter(logging.Formatter(f'[%(asctime)s] - %(levelname)s - RoboTrader: {self._symbol}-{self._interval}-{self._estimator} - %(message)s', '%Y-%m-%d %H:%M:%S'))
    fh.setLevel(log_level)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(f'[%(asctime)s] - %(levelname)s - RoboTrader: {self._symbol}-{self._interval}-{self._estimator} - %(message)s', '%Y-%m-%d %H:%M:%S'))
    sh.setLevel(log_level)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

  def _data_preprocessing(self):
    return

  def _feature_engineering(self):
    self._all_features += self._kline_features
    self._all_features += ['rsi'] if self._calc_rsi and 'rsi' not in self._all_features else []

    if int(self._regression_times) > 0:
      self.log.info(f'calculating regresstion_times: {self._regression_times} - regression_features: {self._regression_features}')
      self._all_data, self._features_added = utils.regresstion_times(
          self._all_data,
          self._regression_features,
          self._regression_times,
          last_one=False)
      self.log.info(f'info after calculating regresstion_times: ')

      self._all_features += self._features_added

    self.log.info(f'All Features: {self._all_features}')
    self.log.debug('Reshape to All Features')
    self._all_data = self._all_data[self._kline_features]
    self._all_data.info()

  def _load_model(self):
    self._model_name_init = utils.get_model_name_to_load(
        self._symbol,
        self._interval,
        self._estimator,
        self._stop_loss,
        self._regression_times,
        self._times_regression_profit_and_loss)

    self._experiment, self._model = utils.load_model(
        self._symbol,
        self._interval,
        self._estimator,
        self._stop_loss,
        self._regression_times,
        self._times_regression_profit_and_loss)

    self.log.info(f'model {self._model_name_init} loaded.')

  def update_data_from_web(self):
    self.log.debug(f'Getting data from web. all_data shape: {self._all_data.shape}')
    df_klines = utils.get_klines(symbol=self._symbol, interval=self._interval, max_date=None, limit=3, columns=self._kline_features, parse_data=True)
    # df_klines['symbol'] = self._symbol
    self.log.debug(f'df_klines shape: {df_klines.shape}')
    df_klines.info() if self._verbose else None

    self._all_data = pd.concat([self._all_data, df_klines])
    self._all_data.drop_duplicates(keep='last', subset=['open_time'], inplace=True)
    self._all_data.sort_index(inplace=True)
    self.log.info(f'Updated - all_data.shape: {self._all_data.shape}')

    now_price = df_klines.tail(1)['close'].values[0]
    self.log.info(f'{self._symbol} Price: {now_price:.6f}')

    latest_closed_candle_open_time = df_klines.iloc[df_klines.shape[0] - 2:df_klines.shape[0] - 1]['open_time'].values[0]
    return now_price, latest_closed_candle_open_time

  def feature_engineering_on_loop(self):
    if self._calc_rsi:
      self.log.info(f'Start Calculating RSI...')
      self._all_data = calc_utils.calc_RSI(self._all_data)
      rsi = self._all_data.tail(1)['rsi'].values[0]
      self.log.debug(f'After Calculating RSI. all_data.shape: {self._all_data.shape}')

    self.log.info(f'regression_times {self._regression_times}...')
    if (self._regression_times is not None) and (self._regression_times > 0):
      self._all_data, _ = utils.regresstion_times(self._all_data, self._numeric_features, self._regression_times, last_one=True)

    return rsi

  def calc_sl_pnl(self, operation, actual_value, margin):
    take_profit_value = 0.0
    stop_loss_value = 0.0
    if operation.startswith('CAI'):  # Short
      take_profit_value = actual_value * (1 - margin / 100)
      stop_loss_value = actual_value * (1 + (margin * myenv.stop_loss_multiplier) / 100)
    elif operation.startswith('SOBE'):  # Long
      take_profit_value = actual_value * (1 + margin / 100)
      stop_loss_value = actual_value * (1 - (margin * myenv.stop_loss_multiplier) / 100)
    return take_profit_value, stop_loss_value

  def validate_short_or_long(self, operation):
    return operation.startswith('SOBE') or operation.startswith('CAI')

  def is_long(self, operation):
    return operation.startswith('SOBE')

  def is_short(self, operation):
    return operation.startswith('CAI')

  def get_amount_to_invest(self, register=True):
    account_balance = utils.get_account_balance()
    balance = account_balance['balance']
    amount_invested = 0.0

    if balance >= myenv.default_amount_invested:
      amount_invested = myenv.default_amount_invested
    elif balance > 0 and balance < myenv.default_amount_invested:
      amount_invested = balance
    balance -= amount_invested
    if register:
      utils.register_account_balance(balance)

    return amount_invested, balance

  def get_account_balance(self):
    account_balance = utils.get_account_balance()
    return account_balance['balance']

  def log_info(self, purchased, open_time, operation, purchase_price, actual_price, margin, amount_invested, profit_and_loss, balance, take_profit_price,
               stop_loss_price, margin_operation):
    _open_time = pd.to_datetime(open_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
    if purchased:
      msg = f'*PURCHASED*: Symbol: {self._symbol}_{self._interval} - Open Time: {_open_time} - Operation: {operation} - Target Margin: {margin_operation:.2f}% '
      msg += f'- Purchased Price: $ {purchase_price:.6f} - Actual Price: $ {actual_price:.6f} - Margin: {100*margin:.2f}% - Amount invested: $ {amount_invested:.2f} '
      msg += f'- PnL: $ {profit_and_loss:.2f} - Take Profit: $ {take_profit_price:.6f} - Stop Loss: $ {stop_loss_price:.6f} - Balance: $ {balance:.2f}'
    else:
      msg = f'*NOT PURCHASED*: Symbol: {self._symbol}_{self._interval} - Open Time: {_open_time} - Actual Price: $ {actual_price:.6f} - Balance: $ {balance:.2f}'
    self.log.info(f'{msg}')
    sm.send_status_to_telegram(msg)

  def log_buy(self, open_time, operation, purchase_price, amount_invested, balance, margin_operation, take_profit_price, stop_loss_price):
    _open_time = pd.to_datetime(open_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
    msg = f'*BUYING*: Symbol: {self._symbol}_{self._interval} - Open Time: {_open_time} - Operation: {operation} - Target Margin: {margin_operation:.2f}% - '
    msg += f'Purchased Price: $ {purchase_price:.6f} - Amount invested: $ {amount_invested:.2f} - Take Profit: $ {take_profit_price:.6f} - '
    msg += f'Stop Loss: $ {stop_loss_price:.6f} - Balance: $ {balance:.2f}'
    self.log.info(f'{msg}')
    sm.send_to_telegram(msg)

  def log_selling(self, open_time, operation, purchase_price, actual_price, margin, amount_invested, profit_and_loss, balance, take_profit_price,
                  stop_loss_price, margin_operation):
    _open_time = pd.to_datetime(open_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
    msg = f'*SELLING*: Symbol: {self._symbol}_{self._interval} - Open Time: {_open_time} - Operation: {operation} - Target Margin: {margin_operation:.2f}% '
    msg += f'- Purchased Price: $ {purchase_price:.6f} - Actual Price: $ {actual_price:.6f} - Margin: {100*margin:.2f}% - Amount invested: $ {amount_invested:.2f} '
    msg += f'- PnL: $ {profit_and_loss:.2f} - Take Profit: $ {take_profit_price:.6f} - Stop Loss: $ {stop_loss_price:.6f} - Balance: $ {balance:.2f}'
    self.log.info(f'{msg}')
    sm.send_to_telegram(msg)

  def predict_operation(self):
    operation, margin = '', 0.0
    self.log.info(f'Start Predicting operation.')

    df_to_predict = self._all_data.tail(1)
    self.log.debug(f'Data input to Predict:\n{df_to_predict.to_dict(orient="records")}')

    df_predict = self._experiment.predict_model(self._model, df_to_predict, verbose=self._verbose)

    prediction_label = df_predict['prediction_label'].values[0]
    self.log.info(f'Operation Predicted: {prediction_label}')

    if self.validate_short_or_long(prediction_label):
      operation = prediction_label.split('_')[0]
      margin = float(prediction_label.split('_')[1])

    return operation, margin

  def run(self):
    self.log.info(f'Columns: {self._kline_features}')

    self.log.info(f'Start _data_preprocessing...')
    self._data_preprocessing()
    self.log.info(f'Start _feature_engineering...')
    self._feature_engineering()
    self.log.info(f'Start _load_model...')
    self._load_model()

    cont = 0
    cont_aviso = 101

    purchased = False
    purchase_price = 0.0
    actual_price = 0.0
    amount_invested = 0.0
    # balance = 0.0
    take_profit_price = 0.0
    stop_loss_price = 0.0
    profit_and_loss = 0.0
    operation = ''
    margin = 0.0
    margin_operation = 0.0
    rsi = 0.0
    latest_closed_candle_open_time_aux = None

    balance = self.get_account_balance()

    params_operation = utils.get_latest_operation(self._symbol, self._interval)
    if len(params_operation) > 0 and params_operation['operation'] == 'BUY':
      purchased = True
      purchase_price = float(params_operation['purchase_price'])
      take_profit_price = float(params_operation['take_profit'])
      stop_loss_price = float(params_operation['stop_loss'])
      operation = params_operation['status']
      latest_closed_candle_open_time_aux = pd.to_datetime(params_operation["operation_date"], unit='ms').strftime('%Y-%m-%d %H:%M:%S')
      amount_invested = float(params_operation['amount_invested'])
      rsi = float(params_operation['rsi'])
      self.log_buy(latest_closed_candle_open_time_aux, operation, purchase_price, amount_invested, balance)

    error = False
    while True:
      try:
        error = False
        # Update data
        balance = self.get_account_balance()
        actual_price, latest_closed_candle_open_time = self.update_data_from_web()

        # Apply predict only on time per interval
        if (not purchased) and (latest_closed_candle_open_time_aux != latest_closed_candle_open_time):
          latest_closed_candle_open_time_aux = latest_closed_candle_open_time
          rsi = self.feature_engineering_on_loop()
          operation, margin_operation = self.predict_operation()

          if self.validate_short_or_long(operation):  # If true, BUY
            amount_invested, balance = self.get_amount_to_invest(register=True)
            purchased = True
            purchase_price = actual_price
            take_profit_price, stop_loss_price = self.calc_sl_pnl(operation, purchase_price, margin_operation)
            params_operation = utils.get_params_operation(self._symbol, self._interval, 'BUY', amount_invested, balance, take_profit_price, stop_loss_price,
                                                          purchase_price, 0.0, 0.0, rsi, operation)
            utils.register_operation(params_operation)
            self.log_buy(latest_closed_candle_open_time, operation, purchase_price, amount_invested, balance, margin_operation, take_profit_price, stop_loss_price)
            self.log.info(f'\nOperation: {operation} - Perform BUY: {self.validate_short_or_long(operation)}' +
                          f'\nActual Price: $ {actual_price:.6f}\nPurchased Price: $ {purchase_price:.6f}\nAmount Invested: $ {amount_invested:.2f}' +
                          f'\nTake Profit: $ {take_profit_price:.6f}\nStop Loss: $ {stop_loss_price:.6f}\nPnL: $ {profit_and_loss:.2f}' +
                          f'\nMargin Operation: {margin_operation:.2f}%\nBalance: $ {balance:.2f}')
            continue

        if purchased:  # and (operation.startswith('SOBE') or operation.startswith('CAI')):
          perform_sell = False
          if self.is_long(operation):
            margin = (actual_price - purchase_price) / purchase_price
            if ((actual_price >= take_profit_price) or (actual_price <= stop_loss_price)):  # Long ==> Sell - Take Profit / Stop Loss
              perform_sell = True
          elif self.is_short(operation):
            margin = (purchase_price - actual_price) / purchase_price
            if ((actual_price <= take_profit_price) or (actual_price >= stop_loss_price)):  # Short ==> Sell - Take Profit / Stop Loss
              perform_sell = True

          profit_and_loss = amount_invested * margin
          self.log.info(f'\nOperation: {operation} - Perform SELL: {perform_sell}' +
                        f'\nActual Price: $ {actual_price:.6f}\nPurchased Price: $ {purchase_price:.6f}\nAmount Invested: $ {amount_invested:.2f}' +
                        f'\nTake Profit: $ {take_profit_price:.6f}\nStop Loss: $ {stop_loss_price:.6f}\nMargin: {100*margin:.2f}\nPnL: $ {profit_and_loss:.2f}' +
                        f'\nMargin Operation: {margin_operation:.2f}%\nBalance: $ {balance:.2f}')

          if perform_sell:  # Register Sell
            balance += (amount_invested + profit_and_loss)
            utils.register_account_balance(balance)
            params_operation = utils.get_params_operation(self._symbol, self._interval, 'SELL', amount_invested, balance, take_profit_price, stop_loss_price,
                                                          purchase_price, actual_price, profit_and_loss, rsi, operation)
            utils.register_operation(params_operation)
            self.log_selling(latest_closed_candle_open_time, operation, purchase_price, actual_price, margin, amount_invested, profit_and_loss, balance,
                             take_profit_price, stop_loss_price, margin_operation)
            # Reset variables
            purchased, purchase_price, amount_invested, take_profit_price, stop_loss_price, profit_and_loss, margin, margin_operation = (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      except Exception as e:
        self.log.exception(e)
        sm.send_status_to_telegram('ERROR: ' + str(e))
        error = True
      finally:
        time.sleep(myenv.sleep_refresh)
        cont += 1
        cont_aviso += 1
        if cont_aviso > 100 and not error:
          cont_aviso = 0
          self.log_info(purchased, latest_closed_candle_open_time, operation, purchase_price, actual_price, margin, amount_invested, profit_and_loss, balance,
                        take_profit_price, stop_loss_price, margin_operation)
