from src.robo_trader import RoboTrader

import os
import glob

import sys
import src.utils as utils
import src.calcEMA as calc_utils
import src.myenv as myenv
import logging
import threading
import pandas as pd


class BatchRoboTrader:
  def __init__(self,
               verbose,
               start_date,
               log_level):

    # Boolean arguments
    self.verbose = verbose
    # Single arguments
    self.start_date = start_date
    self.log_level = log_level
    # Private arguments
    self._all_data_list = {}
    self._top_params = None
    # Initialize logging
    self.log = logging.getLogger("batch_robo_logger")

  # Class methods session

  def _data_collection(self):
    self.log.info(f'Loading data to memory: Symbols: {[s["symbol"] for s in self._top_params]} - Intervals: {[s["interval"] for s in self._top_params]}')
    for param in self._top_params:
      try:
        ix_symbol = f'{param["symbol"]}{myenv.currency}_{param["interval"]}'
        self.log.info(f'Loading data for symbol: {ix_symbol}...')
        self._all_data_list[ix_symbol] = utils.get_data(
            symbol=f'{param["symbol"]}{myenv.currency}',
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
    self.log.info('Prepare Train Data...')
    for param in self._top_params:
      ix_symbol = f'{param["symbol"]}{myenv.currency}_{param["interval"]}'
      try:
        if self.calc_rsi:
          self._all_data_list[ix_symbol] = calc_utils.calc_RSI(self._all_data_list[ix_symbol])
          self._all_data_list[ix_symbol].dropna(inplace=True)
          self._all_data_list[ix_symbol].info() if self.verbose else None

        self.log.info(f'Filtering Data for start_date: {self.start_date}')
        self._all_data = self._all_data[(self._all_data['open_time'] >= self.start_date)]

        self.log.info('info after filtering start_date: ') if self._verbose else None
        self._all_data.info() if self._verbose else None

      except Exception as e:
        self.log.error(e)

  def _update_data_from_web(self):
    return

  def _prepare_top_params(self):
    file_list = glob.glob(os.path.join(f'{myenv.datadir}/', 'resultado_simulacao_*.csv'))
    df_top_params = pd.DataFrame()
    for file_path in file_list:
      if os.path.isfile(file_path):
        df = pd.read_csv(file_path, sep=';')
        df_top_params = pd.concat([df_top_params, df.tail(1)], ignore_index=True)

    df_top_params.to_csv(f'{myenv.datadir}/top_params.csv', sep=';', index=False)
    self._top_params = df_top_params.to_dict(orient='records')
    self.log.info(f'Top Params: \n{self._top_params}')
  # Public methods

  def run(self):
    self.log.info(f'{self.__class__.__name__}: Start _prepare_top_params...')
    self._prepare_top_params()
    self.log.info(f'{self.__class__.__name__}: Start _data_collection...')
    self._data_collection()
    self.log.info(f'{self.__class__.__name__}: Start _data_preprocessing...')
    self._data_preprocessing()

    self.log.info(f'{self.__class__.__name__}: Start Running...')
    # data;symbol;interval;estimator;stop_loss;regression_times;times_regression_profit_and_loss;profit_and_loss_value;start_value;final_value;numeric_features;regression_features;train_size;use-all-data-to-train;start_train_date;start_test_date;fold;no-tune;score;model_name;arguments
    params_list = []
    for params in self._top_params:
      ix_symbol = f'{params["symbol"]}{myenv.currency}_{params["interval"]}'
      robo_trader_param = {
          'all_data': self._all_data_list[ix_symbol],
          'symbol': params['symbol'],
          'interval': params['interval'],
          'estimator': params['estimator'],
          'start_date': self.start_date,
          'numeric_features': params['numeric_features'],
          'stop_loss': params['stop_loss'],
          'regression_times': params['regression_times'],
          'regression_features': params['regression_features'],
          'times_regression_profit_and_loss': params['times_regression_profit_and_loss'],
          'calc_rsi': '-calc-rsi' in params['arguments'],
          'verbose': self.verbose,
          'arguments': params['arguments']}
      params_list.append(robo_trader_param)

      self.log.info(f'Total Robo Trades to start...: {len(params_list)}')

    for params in params_list:
      print(params)
      # robo = RoboTrader(params)
      # threading.Thread(robo.run).start()
