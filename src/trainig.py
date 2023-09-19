import src.utils as utils
from pycaret.classification import ClassificationExperiment

import src.myenv as myenv
import logging


class Train:
    def __init__(self, params: dict):
        '''
        ## Params structure:
        param = {
            all_data,
            symbol,
            interval,
            estimator,
            train_size,
            start_train_date,
            start_test_date,
            numeric_features,
            stop_loss,
            regression_times,
            regression_features,
            times_regression_profit_and_loss,
            calc_rsi,
            compare_models,
            n_jobs,
            use_gpu,
            verbose,
            normalize,
            fold,
            use_all_data_to_train,
            parametros,
            no_tune
        }
        '''

        # Single arguments
        self._all_data = params['all_data']
        self._symbol = params['symbol']
        self._interval = params['interval']
        self._estimator = params['estimator']
        self._train_size = float(params['train_size'])
        self._n_jobs = int(params['n_jobs'])
        self._fold = int(params['fold'])

        self.log = logging.getLogger("training_logger")
        self.log.info(f'{self.__class__.__name__}: Init Params: {params.keys()}')
        # List arguments
        self._start_train_date = params['start_train_date']
        self._start_test_date = params['start_test_date']
        self._numeric_features = params['numeric_features']
        self._stop_loss = float(params['stop_loss'])
        self._regression_times = int(params['regression_times'])
        self._regression_features = params['regression_features']
        self._times_regression_profit_and_loss = int(params['times_regression_profit_and_loss'])
        # Boolean arguments
        self._calc_rsi = params['calc_rsi']
        self._compare_models = params['compare_models']
        self._use_gpu = params['use_gpu']
        self._verbose = params['verbose']
        self._normalize = params['normalize']
        self._use_all_data_to_train = params['use_all_data_to_train']
        self._arguments = params['arguments']
        self._no_tune = params['no_tune']
        self._save_model = params['save_model']

        # Internal atributes
        self._features_added = []
        self._experiement = None
        self._setup = None
        self._model = None
        self._tuned_model = None
        self._final_model = None
        self._train_data = None
        self._test_data = None

        # Prefix for log
        self.pl = f'Train: {self._symbol}{myenv.currency}-{self._interval}-{self._estimator}'

    # Helper functions
    def _prepare_train_data(self):
        self.log.info(f'Filtering train_data: start_train_date: {self._start_train_date} - start_test_date: {self._start_test_date}')

        self.log.info('Prepare Train Data...')
        try:
            self._train_data = \
                self._all_data[(self._all_data['open_time'] >= self._start_train_date) &
                               (self._all_data['open_time'] < self._start_test_date)]
            self.log.info('info after filtering train_data: ') if self._verbose else None
            self._train_data.info() if self._verbose else None

            self.log.info(f'{self.pl}: Setup model - train_data.shape: {self._train_data.shape}')
            self.log.info(f'{self.pl}: Setup model - train_data: label stats: \n{self._train_data.groupby(myenv.label)[myenv.label].count()}')

        except Exception as e:
            self.log.error(e)

    def _prepare_test_data(self):
        if not self._use_all_data_to_train:
            self.log.info(f'Filtering test_data: start_test_date: {self._start_test_date}')
            self.log.info('Prepare Test Data...')
            try:
                self._test_data = self._all_data[(self._all_data['open_time'] > self._start_test_date)]
                self.log.info('info after filtering train_data: ') if self._verbose else None
                self._test_data.info() if self._verbose else None

                self.log.info(f'{self.pl}: Setup model - test_data.shape: {self._test_data.shape}')
                self.log.info(f'{self.pl}: Setup model - test_data: label stats: \n{self._test_data.groupby(myenv.label)[myenv.label].count()}')

            except Exception as e:
                self.log.error(e)

    # ML Pipeline functions
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

        self.log.info(f'{self.pl}: calculating regression_profit_and_loss - times: {self._times_regression_profit_and_loss} - stop_loss: {self._stop_loss}')
        self._all_data = utils.regression_PnL(
            data=self._all_data,
            label=myenv.label,
            diff_percent=self._stop_loss,
            max_regression_profit_and_loss=self._times_regression_profit_and_loss,
            drop_na=True,
            drop_calc_cols=True,
            strategy=None)
        self.log.info(f'{self.pl}: info after calculating regression_profit_and_loss: ')
        self._all_data.info() if self._verbose else None

        self._prepare_train_data()
        self._prepare_test_data()

    def _model_selection(self):
        aux_numeric_features = self._numeric_features.split(',')
        aux_numeric_features += self._features_added

        self._experiement = ClassificationExperiment()
        self._setup = self._experiement.setup(data=self._train_data,
                                              train_size=self._train_size,
                                              target=myenv.label,
                                              numeric_features=aux_numeric_features,
                                              date_features=['open_time'],
                                              create_date_columns=["hour", "day", "month"],
                                              fold_strategy='timeseries',
                                              fold=self._fold,
                                              session_id=123,
                                              normalize=self._normalize,
                                              use_gpu=self._use_gpu,
                                              verbose=self._verbose,
                                              n_jobs=self._n_jobs,
                                              log_experiment=self._verbose)

    def _model_training(self):
        # Accuracy	AUC	Recall	Prec.	F1	Kappa	MCC
        if self._compare_models:
            self.log.info(f'{self.pl}: comparing models...')
            self._model = self._setup.compare_models()
            self._estimator = self._setup.pull().index[0]
            self.log.info(f'{self.pl}: Best Model Estimator: {self._estimator}')
        else:
            self.log.info(f'{self.pl}: creating model...')
            self._model = self._setup.create_model(self._estimator)

    def _model_evaluation(self):
        return

    def _model_optimization(self):
        self._tune_model = self._model
        if not self._no_tune:
            self.log.info(f'{self.pl}: Tuning model...')
            self._tune_model = self._setup.tune_model(self._model)

        self.log.info(f'{self.pl}: Finalizing model...')
        self._final_model = self._setup.finalize_model(self._tune_model)

    def _finalize_training(self):
        model_name = '<< NOT SAVED >>'
        if self._save_model:
            utils.save_model(
                self._symbol,
                self._final_model,
                self._experiement,
                self._estimator,
                self._stop_loss,
                self._regression_times,
                self._times_regression_profit_and_loss)
            model_name = utils.get_model_name(
                self._symbol,
                self._estimator,
                self._stop_loss,
                self._regression_times,
                self._times_regression_profit_and_loss)

        res_score = None
        start_test_date = None
        end_test_date = None
        saldo_inicial = 0.0
        saldo_final = 0.0

        if not self._use_all_data_to_train:
            ajusted_test_data = self._test_data.drop(myenv.label, axis=1)
            df_final_predict, res_score = utils.validate_score_test_data(
                self._setup,
                self._final_model,
                myenv.label,
                self._test_data,
                ajusted_test_data)

            self.log.info(f'{self.pl}: simule trading...')
            start_test_date = df_final_predict['open_time'].min()
            end_test_date = df_final_predict['open_time'].max()

            self.log.info(f'{self.pl}: Min Data: {start_test_date}')
            self.log.info(f'{self.pl}: Max Data: end_test_date')
            saldo_inicial = 100.0
            saldo_final = utils.simule_trading_crypto(df_final_predict, start_test_date, end_test_date, saldo_inicial, self._stop_loss)

        utils.save_results(
            model_name,
            self._symbol,
            self._interval,
            self._estimator,
            self._train_size,
            self._start_train_date,
            start_test_date,
            self._numeric_features,
            self._regression_times,
            self._regression_features,
            self._times_regression_profit_and_loss,
            self._stop_loss,
            self._fold,
            saldo_inicial,
            saldo_final,
            self._use_all_data_to_train,
            self._no_tune,
            res_score,
            self._arguments)

    def run(self):
        result = 'SUCESS'
        try:
            self.log.info(f'{self.pl}: \n\nStart Trainign >>>>>>>>>>>>>>>>>>>>>>')
            self.log.info(f'{self.pl}: Start data_preprocessing...')
            self._data_preprocessing()
            self.log.info(f'{self.pl}: Start feature_engineering...')
            self._feature_engineering()
            self.log.info(f'{self.pl}: Start model_selection...')
            self._model_selection()
            self.log.info(f'{self.pl}: Start model_training...')
            self._model_training()
            self.log.info(f'{self.pl}: Start model_evaluation...')
            self._model_evaluation()
            self.log.info(f'{self.pl}: Start model_optimization...')
            self._model_optimization()
            self.log.info(f'{self.pl}: Start finalize_training...')
            self._finalize_training()
            self.log.info(f'{self.pl}: End Trainign <<<<<<<<<<<<<<<<<<<<<<\n\n')
        except Exception as e:
            self.log.exception(e)
            result = 'ERROR'
        return result
