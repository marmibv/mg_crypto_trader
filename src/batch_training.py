from src.trainig import Train

import sys
import src.utils as utils
import src.train as train
import src.calcEMA as calc_utils
import src.myenv as myenv
import src.send_message as sm
import logging
import pandas as pd
import datetime


class BatchTrain:
  def __init__(self,
               update_data_from_web,
               calc_rsi,
               use_gpu,
               normalize,
               verbose,
               use_all_data_to_train,
               revert,
               no_tune,
               save_model,
               start_train_date,
               start_test_date,
               fold,
               n_jobs,
               n_threads,
               log_level,
               symbol_list,
               interval_list,
               estimator_list,
               stop_loss_list,
               numeric_features_list,
               times_regression_PnL_list,
               regression_times_list,
               regression_features_list):

    # Boolean arguments
    self._update_data_from_web = update_data_from_web
    self._calc_rsi = calc_rsi
    self._use_gpu = use_gpu
    self._normalize = normalize
    self._verbose = verbose
    self._use_all_data_to_train = use_all_data_to_train
    self._revert = revert
    self._no_tune = no_tune
    self._save_model = save_model
    # Single arguments
    self._start_train_date = start_train_date
    self._start_test_date = start_test_date
    self._fold = fold
    self._n_jobs = n_jobs
    self._n_threads = n_threads
    self._log_level = log_level
    # List arguments
    self._symbol_list = symbol_list
    self._interval_list = interval_list
    self._estimator_list = estimator_list
    self._stop_loss_list = stop_loss_list
    self._numeric_features_list = numeric_features_list
    self._times_regression_PnL_list = times_regression_PnL_list
    self._regression_times_list = regression_times_list
    self._regression_features_list = regression_features_list
    # Private arguments
    self._all_data_list = {}

    # Initialize logging
    self.log = logging.getLogger("training_logger")

    # Prefix for log
    self.pl = f'BatchTrain: '

  # Class methods session
  def get_ix_symbol(self, symbol, interval, stop_loss, times_regression_PnL):
    return f'{symbol}_{interval}_SL_{stop_loss}_PnL_{times_regression_PnL}'

  def _data_collection(self):
    self.log.info(f'{self.pl}: Loading data to memory: Symbols: {self._symbol_list} - Intervals: {self._interval_list}')
    for interval in self._interval_list:
      for symbol in self._symbol_list:
        self.log.info(f'{self.pl}: Loading data for symbol: {symbol}_{interval}...')
        _aux_data = utils.get_data(
            symbol=f'{symbol}',
            save_database=False,
            interval=interval,
            tail=-1,
            columns=myenv.all_cols,
            parse_data=True,
            updata_data_from_web=self._update_data_from_web,
            start_date=self._start_train_date)

        try:
          self.log.info(f'{self.pl}: Calculating RSI for symbol: {symbol}_{interval}...')
          _aux_data = calc_utils.calc_RSI(_aux_data)
          _aux_data.info() if self._verbose else None
        except Exception as e:
          self.log.error(e)

        for stop_loss in self._stop_loss_list:
          for times_regression_PnL in self._times_regression_PnL_list:
            try:
              ix_symbol = self.get_ix_symbol(symbol, interval, stop_loss, times_regression_PnL)
              self.log.info(f'{self.pl}: Store data in memory for symbol: {ix_symbol}...')
              self._all_data_list[ix_symbol] = _aux_data.copy()
              if self._all_data_list[ix_symbol].shape[0] == 0:
                raise Exception(f'Data for symbol: {ix_symbol} is empty')
            except Exception as e:
              self.log.error(e)
    self.log.info(f'{self.pl}: Loaded data to memory for symbols: {self._symbol_list}')

  def _data_preprocessing(self):
    self.log.info('Start Data  Preprocessing...')
    for interval in self._interval_list:
      for symbol in self._symbol_list:
        for stop_loss in self._stop_loss_list:
          for times_regression_PnL in self._times_regression_PnL_list:
            try:
              ix_symbol = self.get_ix_symbol(symbol, interval, stop_loss, times_regression_PnL)
              self.log.info(f'{self.pl}: Calculating regression_profit_and_loss for key {ix_symbol}...')
              self._all_data_list[ix_symbol].info() if self._verbose else None
              self._all_data_list[ix_symbol] = utils.regression_PnL(
                  data=self._all_data_list[ix_symbol],
                  label=myenv.label,
                  diff_percent=float(stop_loss),
                  max_regression_profit_and_loss=int(times_regression_PnL),
                  drop_na=True,
                  drop_calc_cols=True,
                  strategy=None)
              self.log.info(f'{self.pl}:  info after calculating regression_profit_and_loss: ') if self._verbose else None
              self._all_data_list[ix_symbol].info() if self._verbose else None
            except Exception as e:
              self.log.error(e)

    if self._use_all_data_to_train:
      self._start_test_date = None

  def run(self):
    self.log.info(f'{self.pl}: {self.__class__.__name__}: Start _data_collection...')
    self._data_collection()
    self.log.info(f'{self.pl}: {self.__class__.__name__}: Start _data_preprocessing...')
    self._data_preprocessing()

    self.log.info(f'{self.pl}: {self.__class__.__name__}: Start Running...')
    params_list = []
    _prm_list = []
    for interval in self._interval_list:
      for symbol in self._symbol_list:
        for estimator in self._estimator_list:
          for stop_loss in self._stop_loss_list:
            for times_regression_PnL in self._times_regression_PnL_list:
              for nf_list in self._numeric_features_list:  # Numeric Features
                # nf_list += ',rsi' if self._calc_rsi else None
                for rt_list in self._regression_times_list:
                  ix_symbol = self.get_ix_symbol(symbol, interval, stop_loss, times_regression_PnL)
                  if rt_list != '0':
                    for rf_list in self._regression_features_list:
                      train_param = {
                          'all_data': self._all_data_list[ix_symbol],
                          'symbol': symbol,
                          'interval': interval,
                          'estimator': estimator,
                          'train_size': myenv.train_size,
                          'start_train_date': self._start_train_date,
                          'start_test_date': self._start_test_date,
                          'numeric_features': nf_list,
                          'stop_loss': stop_loss,
                          'regression_times': rt_list,
                          'regression_features': rf_list,
                          'times_regression_profit_and_loss': times_regression_PnL,
                          'calc_rsi': self._calc_rsi,
                          'compare_models': False,
                          'n_jobs': self._n_jobs,
                          'use_gpu': self._use_gpu,
                          'verbose': self._verbose,
                          'normalize': self._normalize,
                          'fold': self._fold,
                          'use_all_data_to_train': self._use_all_data_to_train,
                          'arguments': str(sys.argv[1:]),
                          'no_tune': self._no_tune,
                          'save_model': self._save_model}
                      params_list.append(train_param)
                      _prm_list.append(train_param.copy())
                  else:
                    train_param = {
                        'all_data': self._all_data_list[ix_symbol],
                        'symbol': symbol,
                        'interval': interval,
                        'estimator': estimator,
                        'train_size': myenv.train_size,
                        'start_train_date': self._start_train_date,
                        'start_test_date': self._start_test_date,
                        'numeric_features': nf_list,
                        'stop_loss': stop_loss,
                        'regression_times': rt_list,
                        'regression_features': None,
                        'times_regression_profit_and_loss': times_regression_PnL,
                        'calc_rsi': self._calc_rsi,
                        'compare_models': False,
                        'n_jobs': self._n_jobs,
                        'use_gpu': self._use_gpu,
                        'verbose': self._verbose,
                        'normalize': self._normalize,
                        'fold': self._fold,
                        'use_all_data_to_train': self._use_all_data_to_train,
                        'arguments': str(sys.argv[1:]),
                        'no_tune': self._no_tune,
                        'save_model': self._save_model}
                    params_list.append(train_param)
                    _prm_list.append(train_param.copy())

      self.log.info(f'{self.pl}: Total Trainning Models: {len(params_list)}')

    for _prm in _prm_list:
      del _prm['all_data']

    pd.DataFrame(_prm_list).to_csv(f'{myenv.datadir}/params_list{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv', index=False)
    results = []
    for params in params_list:
      train = Train(params)
      res = train.run()
      results.append(res)

    self.log.info(f'{self.pl}: Results of {len(params_list)} Models execution: \n{pd.DataFrame(results, columns=["status"])["status"].value_counts()}')
