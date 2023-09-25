from src.robo_trader import RoboTrader
from src.trainig import Train

import src.utils as utils
import src.calcEMA as calc_utils
import src.myenv as myenv
import logging
import pandas as pd
import threading
import os


class BatchRoboTrader:
  def __init__(self,
               verbose,
               start_date,
               log_level):

    # Boolean arguments
    self._verbose = verbose
    # Single arguments
    self._start_date = start_date
    self._log_level = log_level
    # Private arguments
    self._all_data_list = {}
    self._top_params = utils.get_best_parameters()

    # Initialize logging
    self.log = self._configure_log(log_level)
    self.log.setLevel(log_level)

  def _configure_log(self, log_level):
    log_file_path = os.path.join(myenv.logdir, myenv.batch_robo_log_filename)
    logging.basicConfig(
        level=log_level,  # Set the minimum level to be logged
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode='a', delay=True),  # Log messages to a file
            logging.StreamHandler()  # Log messages to the console
        ]
    )
    return logging.getLogger("batch_robo_logger")

  def _data_collection(self):
    self.log.info(f'Loading data to memory: Symbols: {[s["symbol"] for s in self._top_params]} - Intervals: {[s["interval"] for s in self._top_params]}')
    for param in self._top_params:
      try:
        ix_symbol = f'{param["symbol"]}_{param["interval"]}'
        self.log.info(f'Loading data for symbol: {ix_symbol}...')
        self._all_data_list[ix_symbol] = utils.get_data(
            symbol=f'{param["symbol"]}',
            save_database=False,
            interval=param['interval'],
            tail=-1,
            columns=myenv.all_cols,
            parse_data=True,
            updata_data_from_web=False)
      except Exception as e:
        self.log.error(e)
    self.log.info(f'Loaded data to memory for symbols: {[s["symbol"] for s in self._top_params]}')

  def _data_preprocessing(self):
    self.log.info('Prepare All Data...')
    for param in self._top_params:
      ix_symbol = f'{param["symbol"]}_{param["interval"]}'
      try:
        self.log.info(f'Calc RSI for symbol: {ix_symbol}')
        self._all_data_list[ix_symbol] = calc_utils.calc_RSI(self._all_data_list[ix_symbol])
        self._all_data_list[ix_symbol].dropna(inplace=True)
        self._all_data_list[ix_symbol].info() if self._verbose else None

        self.log.info(f'Filtering Data for start_date: {self._start_date}')
        self._all_data_list[ix_symbol] = self._all_data_list[ix_symbol][(self._all_data_list[ix_symbol]['open_time'] >= self._start_date)]

        self.log.info('info after filtering start_date: ') if self._verbose else None
        self._all_data_list[ix_symbol].info() if self._verbose else None

      except Exception as e:
        self.log.error(e)

  # Public methods

  def run(self):
    self.log.info(f'{self.__class__.__name__}: Start _data_collection...')
    self._data_collection()
    self.log.info(f'{self.__class__.__name__}: Start _data_preprocessing...')
    self._data_preprocessing()

    self.log.info(f'{self.__class__.__name__}: Start Running...')
    params_list = []
    for params in self._top_params:
      ix_symbol = f'{params["symbol"]}_{params["interval"]}'
      robo_trader_param = {
          'all_data': self._all_data_list[ix_symbol],
          'symbol': f'{params["symbol"]}',
          'interval': params['interval'],
          'estimator': params['estimator'],
          'start_date': self._start_date,
          'numeric_features': params['numeric_features'],
          'stop_loss': params['stop_loss'],
          'regression_times': params['regression_times'],
          'regression_features': params['regression_features'],
          'times_regression_profit_and_loss': params['times_regression_profit_and_loss'],
          'calc_rsi': '-calc-rsi' in params['arguments'],
          'verbose': self._verbose,
          'arguments': params['arguments'],
          'log_level': self._log_level}
      params_list.append(robo_trader_param)

    self.log.info(f'Total Robo Trades to start...: {len(params_list)}')

    for params in params_list:
      model_name = utils.get_model_name_to_load(
          symbol=params['symbol'],
          interval=params['interval'],
          estimator=params['estimator'],
          stop_loss=params['stop_loss'],
          regression_times=params['regression_times'],
          times_regression_profit_and_loss=params['times_regression_profit_and_loss']
      )
      if model_name is None:
        raise Exception(f'Best model not found: {model_name}')

    for params in params_list:
      # print(params['symbol'], params['estimator'], params['stop_loss'], params['regression_times'], params['times_regression_profit_and_loss'])
      model_name = utils.get_model_name_to_load(
          symbol=params['symbol'],
          interval=params['interval'],
          estimator=params['estimator'],
          stop_loss=params['stop_loss'],
          regression_times=params['regression_times'],
          times_regression_profit_and_loss=params['times_regression_profit_and_loss']
      )

      self.log.info(f'Starting Robo Trader for model: {model_name}')
      robo = RoboTrader(params)
      threading.Thread(target=robo.run).start()
