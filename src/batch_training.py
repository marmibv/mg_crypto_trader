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
    """
    Parameters:
    update_data_from_web: a boolean indicating whether to update the data from web.        
    calc_rsi: a boolean indicating whether to calculate the RSI.
    use_gpu: a boolean indicating whether to use GPU for training.
    normalize: a boolean indicating whether to normalize the data.
    verbose: a boolean indicating whether to print verbose output.
    use_all_data_to_train: a boolean indicating whether to use all available data for training.
    revert: a boolean indicating whether to revert the data.
    no_tune: a boolean indicating whether to skip hyperparameter tuning.
    start_train_date: a string representing the start date for training.
    start_test_date: a string representing the start date for testing.
    fold: an integer representing the number of folds for cross-validation.
    n_jobs: an integer representing the number of jobs to run in parallel.
    symbol_list: a list of symbols to train on.
    interval_list: a list of intervals to train on.
    estimator_list: a list of regression models to use.
    stop_loss_list: a list of stop loss values for evaluation.
    numeric_features_list: a list of numeric features.
    times_regression_PnL_list: a list of times for regression profit and loss.
    regression_times_list: a list of regression times.
    regression_features_list: a list of regression features.
    """

    # Boolean arguments
    self.update_data_from_web = update_data_from_web
    self.calc_rsi = calc_rsi
    self.use_gpu = use_gpu
    self.normalize = normalize
    self.verbose = verbose
    self.use_all_data_to_train = use_all_data_to_train
    self.revert = revert
    self.no_tune = no_tune
    self.save_model = save_model
    # Single arguments
    self.start_train_date = start_train_date
    self.start_test_date = start_test_date
    self.fold = fold
    self.n_jobs = n_jobs
    self.n_threads = n_threads
    self.log_level = log_level
    # List arguments
    self.symbol_list = symbol_list
    self.interval_list = interval_list
    self.estimator_list = estimator_list
    self.stop_loss_list = stop_loss_list
    self.numeric_features_list = numeric_features_list
    self.times_regression_PnL_list = times_regression_PnL_list
    self.regression_times_list = regression_times_list
    self.regression_features_list = regression_features_list
    # Private arguments
    self._all_data_list = {}

    # Initialize logging
    self.logger = logging.getLogger("training_logger")

  # Class methods session

  def _data_collection(self):
    self.logger.info(f'Loading data to memory: Symbols: {self.symbol_list} - Intervals: {self.interval_list}')
    for interval in self.interval_list:
      for symbol in self.symbol_list:
        try:
          ix_symbol = f'{symbol}_{interval}'
          self.logger.info(f'Loading data for symbol: {ix_symbol}...')
          self._all_data_list[ix_symbol] = utils.get_data(
              symbol=f'{symbol}',
              save_database=False,
              interval=interval,
              tail=-1,
              columns=myenv.all_cols,
              parse_data=True,
              updata_data_from_web=self.update_data_from_web)
          if self._all_data_list[ix_symbol].shape[0] == 0:
            raise Exception(f'Data for symbol: {ix_symbol} is empty')
        except Exception as e:
          self.logger.error(e)
    self.logger.info(f'Loaded data to memory for symbols: {self.symbol_list}')

  def _data_preprocessing(self):
    self.logger.info('Prepare Train Data...')
    if self.calc_rsi:
      for interval in self.interval_list:
        for symbol in self.symbol_list:
          ix_symbol = f'{symbol}_{interval}'
          try:
            self._all_data_list[ix_symbol] = calc_utils.calc_RSI(self._all_data_list[ix_symbol])
            self._all_data_list[ix_symbol].dropna(inplace=True)
            self._all_data_list[ix_symbol].info() if self.verbose else None
          except Exception as e:
            self.logger.error(e)

    if self.use_all_data_to_train:
      self.start_test_date = None

  def run(self):
    self.logger.info(f'{self.__class__.__name__}: Start _data_collection...')
    self._data_collection()
    self.logger.info(f'{self.__class__.__name__}: Start _data_preprocessing...')
    self._data_preprocessing()

    self.logger.info(f'{self.__class__.__name__}: Start Running...')
    params_list = []
    _prm_list = []
    for interval in self.interval_list:
      for symbol in self.symbol_list:
        for estimator in self.estimator_list:
          for stop_loss in self.stop_loss_list:
            for times_regression_PnL in self.times_regression_PnL_list:
              for nf_list in self.numeric_features_list:  # Numeric Features
                for rt_list in self.regression_times_list:
                  ix_symbol = f'{symbol}_{interval}'
                  if rt_list != '0':
                    for rf_list in self.regression_features_list:
                      train_param = {
                          'all_data': self._all_data_list[ix_symbol],
                          'symbol': symbol,
                          'interval': interval,
                          'estimator': estimator,
                          'train_size': myenv.train_size,
                          'start_train_date': self.start_train_date,
                          'start_test_date': self.start_test_date,
                          'numeric_features': nf_list,
                          'stop_loss': stop_loss,
                          'regression_times': rt_list,
                          'regression_features': rf_list,
                          'times_regression_profit_and_loss': times_regression_PnL,
                          'calc_rsi': self.calc_rsi,
                          'compare_models': False,
                          'n_jobs': self.n_jobs,
                          'use_gpu': self.use_gpu,
                          'verbose': self.verbose,
                          'normalize': self.normalize,
                          'fold': self.fold,
                          'use_all_data_to_train': self.use_all_data_to_train,
                          'arguments': str(sys.argv[1:]),
                          'no_tune': self.no_tune,
                          'save_model': self.save_model}
                      params_list.append(train_param)
                      _prm_list.append(train_param.copy())
                  else:
                    train_param = {
                        'all_data': self._all_data_list[ix_symbol],
                        'symbol': symbol,
                        'interval': interval,
                        'estimator': estimator,
                        'train_size': myenv.train_size,
                        'start_train_date': self.start_train_date,
                        'start_test_date': self.start_test_date,
                        'numeric_features': nf_list,
                        'stop_loss': stop_loss,
                        'regression_times': rt_list,
                        'regression_features': None,
                        'times_regression_profit_and_loss': times_regression_PnL,
                        'calc_rsi': self.calc_rsi,
                        'compare_models': False,
                        'n_jobs': self.n_jobs,
                        'use_gpu': self.use_gpu,
                        'verbose': self.verbose,
                        'normalize': self.normalize,
                        'fold': self.fold,
                        'use_all_data_to_train': self.use_all_data_to_train,
                        'arguments': str(sys.argv[1:]),
                        'no_tune': self.no_tune,
                        'save_model': self.save_model}
                    params_list.append(train_param)
                    _prm_list.append(train_param.copy())

      self.logger.info(f'Total Trainning Models: {len(params_list)}')

    for _prm in _prm_list:
      del _prm['all_data']

    pd.DataFrame(_prm_list).to_csv(f'{myenv.datadir}/params_list{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv', index=False)
    results = []
    for params in params_list:
      train = Train(params)
      res = train.run()
      results.append(res)

    self.logger.info(f'Results of {len(params_list)} Models execution: \n{pd.DataFrame(results, columns=["status"])["status"].value_counts()}')
