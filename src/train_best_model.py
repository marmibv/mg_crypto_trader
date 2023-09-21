from src.robo_trader import RoboTrader
from src.trainig import Train

import src.utils as utils
import src.calcEMA as calc_utils
import src.myenv as myenv
import logging
import pandas as pd


class TrainBestModel:
  def __init__(self,
               verbose,
               log_level):

    # Boolean arguments
    self._verbose = verbose
    # Single arguments
    self._log_level = log_level
    # Private arguments
    self._all_data_list = {}
    self._top_params = utils.get_best_parameters()
    # Initialize logging
    self.log = logging.getLogger("training_logger")

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
    self.log.info('Prepare All Data...')
    for param in self._top_params:
      ix_symbol = f'{param["symbol"]}{myenv.currency}_{param["interval"]}'
      try:
        # if self.calc_rsi:
        self._all_data_list[ix_symbol] = calc_utils.calc_RSI(self._all_data_list[ix_symbol])
        self._all_data_list[ix_symbol].dropna(inplace=True)
        self._all_data_list[ix_symbol].info() if self._verbose else None

        self.log.info('info after filtering start_date: ') if self._verbose else None
        self._all_data_list[ix_symbol].info() if self._verbose else None

      except Exception as e:
        self.log.error(e)

  def run(self):
    self.log.info(f'{self.__class__.__name__}: Start _data_collection...')
    self._data_collection()
    self.log.info(f'{self.__class__.__name__}: Start _data_preprocessing...')
    self._data_preprocessing()

    params_list = []
    for param in self._top_params:
      n_jobs = -1
      if (param['arguments'].startswith('-n-jobs=')):
        n_jobs = int(param['arguments'].split('=')[1])

      fold = 3
      if (param['arguments'].startswith('-fold=')):
        n_jobs = int(param['arguments'].split('=')[1])

      ix_symbol = f'{param["symbol"]}{myenv.currency}_{param["interval"]}'
      train_param = {
          'all_data': self._all_data_list[ix_symbol],
          'symbol': param['symbol'],
          'interval': param['interval'],
          'estimator': param['estimator'],
          'train_size': myenv.train_size,
          'start_train_date': '2010-01-01',
          'start_test_date': None,
          'numeric_features': param['numeric_features'],
          'stop_loss': param['stop_loss'],
          'regression_times': param['regression_times'],
          'regression_features': param['regression_features'],
          'times_regression_profit_and_loss': param['times_regression_profit_and_loss'],
          'calc_rsi': '-calc-rsi' in param['arguments'],
          'compare_models': False,
          'n_jobs': n_jobs,
          'use_gpu': '-use-gpu' in param['arguments'],
          'verbose': self._verbose,
          'normalize': '-normalize' in param['arguments'],
          'fold': fold,
          'use_all_data_to_train': True,
          'arguments': param['arguments'],
          'no_tune': '-no-tune' in param['arguments'],
          'save_model': True}
      params_list.append(train_param)

    results = []
    for params in params_list:
      print(params)
      train = Train(params)
      res = train.run()
      results.append(res)

    self.log.info(f'Results of {len(params_list)} Models execution: \n{pd.DataFrame(results, columns=["status"])["status"].value_counts()}')
