import src.utils as utils
import datetime
import src.myenv as myenv
import logging
import src.send_message as sm
import pandas as pd
import time


class RoboTrader():
  def __init__(self, params: dict):
    # Single arguments
    self._all_data = params['all_data'].copy()
    self._symbol = params['symbol']
    self._interval = params['interval']
    self._estimator = params['estimator']

    self.log = logging.getLogger("training_logger")
    self.log.info(f'{self.__class__.__name__}: Init Params: {params.keys()}')
    # List arguments
    self._start_date = params['start_date']
    self._numeric_features = params['numeric_features']
    self._stop_loss = float(params['stop_loss'])
    self._regression_times = int(params['regression_times'])
    self._regression_features = params['regression_features']
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

    # Prefix for log
    self.pl = f'RoboTrader: {self._symbol}{myenv.currency}-{self._interval}-{self._estimator}'

  def _data_preprocessing(self):
    return

  def _feature_engineering(self):
    if int(self._regression_times) > 0:
      self.log.info(f'{self.pl}: calculating regresstion_times: {self._regression_times} - regression_features: {self._regression_features}')
      self._all_data, self._features_added = utils.regresstion_times(
          self._all_data,
          self._regression_features,
          self._regression_times,
          last_one=False)
      self.log.info(f'{self.pl}: info after calculating regresstion_times: ')
      self._all_data.info() if self._verbose else None

  def _load_model(self):
    self._model_name_init = utils.get_model_name_to_load(
        self.symbol,
        self.estimator,
        self.stop_loss,
        self.regression_times,
        self.times_regression_profit_and_loss)

    self._experiment, self._model = utils.load_model(
        self.symbol,
        self.estimator,
        self.stop_loss,
        self.regression_times,
        self.times_regression_profit_and_loss)

    self.log.info(f'start_predict_engine: model {self._model_name_init} loaded.')

  def run(self):
    self.log.info(f'{self.__class__.__name__}: Start _data_preprocessing...')
    self._data_preprocessing()
    self.log.info(f'{self.__class__.__name__}: Start _feature_engineering...')
    self._feature_engineering()
    self.log.info(f'{self.__class__.__name__}: Start _load_model...')
    self._load_model()

    cont = 0
    cont_aviso = 0
    operacao = ''
    operacao_compra = ''
    comprado = False
    valor_compra = 0
    valor_atual = 0
    diff = 0.0
    self.log.info('start_predict_engine: starting loop monitoring...')
    while True:
      self.log.info('------------------------>>')
      self.log.info(
          f'start_predict_engine: Loop  -->  Symbol: {self._symbol} - Cont: {cont} - Now: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
      try:
        model_name = utils.get_model_name_to_load(
            self.symbol,
            self.estimator,
            self.stop_loss,
            self.regression_times,
            self.times_regression_profit_and_loss)

        if model_name != self._model_name_init:
          self._experiment, self._model = utils.load_model(
              self.symbol,
              self.estimator,
              self.stop_loss,
              self.regression_times,
              self.times_regression_profit_and_loss)  # cassification_experiment

          self._model_name_init = model_name
          sm.send_status_to_telegram(
              f'start_predict_engine: reload new model. New model name: {model_name} - Old model name: {self._model_name_init}')

        max_date = utils.get_max_date(self._all_data)
        open_time = self._all_data.tail(1)["open_time"].dt.strftime('%Y-%m-%d %H:%M:%S').values[0]
        self.log.info('start_predict_engine: max_date: ', max_date)

        df_klines = utils.get_klines(
            symbol=self.symbol,
            max_date=max_date.strftime('%Y-%m-%d'),
            limit=1,
            columns=myenv.date_features + self._numeric_features)

        self._all_data = pd.concat([self._all_data, df_klines])
        self._all_data.drop_duplicates(keep='last', subset=['open_time'], inplace=True)
        self._all_data.sort_index(inplace=True)
        self._all_data['symbol'] = self.symbol
        self._all_data = utils.parse_type_fields(self._all_data)
        self.log.info('start_predict_engine: Updated data - all_data.shape: ', self._all_data.shape)

        self.log.info('start_predict_engine: calc_rsi...')
        if self.calc_rsi:
          self._all_data = utils.calc_RSI(self._all_data)  # , last_one=True)

        self.log.info(f'start_predict_engine: regression_times {self.regression_times}...')
        if self.regression_times > 0:
          self._all_data, _ = utils.regresstion_times(self._all_data, self.numeric_features, self.regression_times, last_one=True)

        # Calculo compra e venda
        valor_atual = self._all_data.tail(1)["close"].values[0]
        self.log.info(f'start_predict_engine: valor_atual: >> $ {valor_atual:.4f} <<')

        if comprado:
          diff = 100 * (valor_atual - valor_compra) / valor_compra

        if (abs(diff) >= self.stop_loss) and comprado:
          if operacao_compra.startswith('SOBE'):
            saldo += saldo * (diff / 100)
          else:
            saldo += saldo * (-diff / 100)
          msg = f'Venda: Symbol: {self.symbol} - open_time: {open_time} - Operação: {operacao_compra} - Valor Comprado: {valor_compra:.4f} - \
Valor Venda: {valor_atual:.4f} - Variação: {diff:.4f}% - PnL: $ {saldo:.2f}'
          sm.send_to_telegram(msg)
          # Reset variaveis
          comprado = False
          valor_compra = 0
          operacao_compra = ''
        # Fim calculo compra e venda

        if not comprado:
          self.log.info('start_predict_engine: start predict_model...')
          df_predict = self._experiment.predict_model(self._model, self._all_data.tail(1), verbose=self.verbose)
          # Inicio calculo compra
          operacao = df_predict['prediction_label'].values[0]
          self.log.info(f'start_predict_engine: operacao predita: {operacao}')
          if (operacao.startswith('SOBE') or operacao.startswith('CAI')):
            comprado = True
            valor_compra = df_predict.tail(1)["close"].values[0]
            operacao_compra = operacao
            rsi = df_predict.tail(1)["rsi"].values[0]

            msg = f'Compra: Symbol: {self.symbol} - open_time: {open_time} - Operação: {operacao_compra} - Valor Comprado: {valor_compra:.4f} - \
RSI: {rsi:.2f} - PnL: $ {saldo:.2f}'
            sm.send_to_telegram(msg)
          # Fim calculo compra
      except Exception as e:
        sm.send_status_to_telegram('ERROR: ' + str(e))
      finally:
        time.sleep(myenv.sleep_refresh)
        cont += 1
        cont_aviso += 1
        if cont_aviso > 100:
          cont_aviso = 0
          if comprado:
            msg = f'*COMPRADO*: Symbol: {self.symbol} - open_time: {open_time} - Operação: {operacao_compra} - Valor Comprado: {valor_compra:.4f} - \
Valor Atual: {valor_atual:.4f} - Variação: {diff:.4f}% - PnL: $ {saldo:.2f}'
            sm.send_status_to_telegram(msg)
          else:
            msg = f'*NÃO COMPRADO*: Symbol: {self.symbol} - open_time: {open_time} - Valor Atual: {valor_atual:.4f} - PnL: $ {saldo:.2f}'
            sm.send_status_to_telegram(msg)
