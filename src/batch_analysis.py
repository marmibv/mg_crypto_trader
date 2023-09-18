from src.utils import *
from src.train import *
from src.calcEMA import calc_RSI

import src.myenv as myenv
import src.send_message as sm

import subprocess as sp


def train(calc_rsi,
          use_gpu,
          normalize,
          verbose,
          use_all_data_to_train,
          revert,
          no_tune,
          start_train_date,
          start_test_date,
          fold,
          n_jobs,
          symbol_list,
          estimator_list,
          stop_loss_list,
          numeric_features_list,
          times_regression_PnL_list,
          regression_times_list,
          regression_features_list):

    print('start_train_engine: parameters: ')
    print([{'calc_rsi': calc_rsi},
          {'use_gpu': use_gpu},
          {'normalize': normalize},
          {'verbose': verbose},
          {'use_all_data_to_train': use_all_data_to_train},
          {'revert': revert},
          {'no_tune': no_tune},
          {'start_train_date': start_train_date},
          {'start_test_date': start_test_date},
          {'fold': fold},
          {'n_jobs': n_jobs},
          {'symbol_list': symbol_list},
          {'estimator_list': estimator_list},
          {'stop_loss_list': stop_loss_list},
          {'numeric_features': numeric_features_list},
          {'times_regression_PnL_list': times_regression_PnL_list},
          {'regression_times_list': regression_times_list},
          {'regression_features': regression_features_list}])

    print('start_train_engine: prepare command...:')

    base_cmd = f'python . -train-model'
    base_cmd += ' -calc-rsi' if calc_rsi else ''
    base_cmd += ' -use-gpu' if use_gpu else ''
    base_cmd += ' -normalize' if normalize else ''
    base_cmd += ' -verbose' if verbose else ''
    base_cmd += ' -use-all-data-to-train' if use_all_data_to_train else ''
    base_cmd += ' -revert' if revert else ''
    base_cmd += ' -no-tune' if no_tune else ''
    base_cmd += ' -start-train-date=' + start_train_date
    base_cmd += ' -start-test-date=' + start_test_date
    base_cmd += ' -fold=' + str(fold)
    base_cmd += ' -n-jobs=' + str(n_jobs)

    command_list = []
    for symbol in symbol_list:
        for estimator in estimator_list:
            for stop_loss in stop_loss_list:
                for times_regression_PnL in times_regression_PnL_list:
                    for nf_list in numeric_features_list:  # Numeric Features
                        for rt_list in regression_times_list:
                            command = f'{base_cmd} -symbol={symbol}USDT -estimator={estimator} -stop-loss={stop_loss}'
                            command += f' -regression-profit-and-loss={times_regression_PnL} -numeric-features={nf_list} -regression-times={rt_list}'
                            if rt_list != '0':
                                command += f' -regression-features={regression_features_list}'
                            command_list.append(command)

    df = pd.DataFrame(data=command_list, columns=['command'], dtype=str)
    df.to_csv('command_list.csv', index=False, sep=';')

    for command in command_list:
        print('start_train_engine: command: ', command)
        exec(command)

    print(f'start_train_engine: Length of command_list: \n{len(command_list)}')


def exec(command):
    result = sp.run(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, text=True)

    print("Output:", result.stdout)
    print("Error:", result.stderr)
    print("Return Code:", result.returncode)


def get_symbol_list():
    result = []
    df = pd.read_csv(datadir + '/symbol_list.csv')
    for symbol in df['symbol']:
        result.append(symbol)
    return result


def main(args):
    # Boolean arguments
    calc_rsi = False
    use_gpu = False
    normalize = False
    verbose = False
    use_all_data_to_train = False
    revert = False
    no_tune = False

    # Single arguments
    start_train_date = '2010-01-01'
    start_test_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
    fold = 3
    n_jobs = myenv.n_jobs

    # List arguments
    symbol_list = get_symbol_list()

    estimator_list = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 'ransac', 'tr',
                      'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 'mlp', 'xgboost', 'lightgbm', 'catboost']
    stop_loss_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    numeric_features_list = prepare_numeric_features_list(myenv.data_numeric_fields)
    times_regression_PnL_list = [6, 12, 24]
    regression_times_list = [24, 360, 720]
    regression_features_list = combine_list(myenv.data_numeric_fields)

    for arg in args:
        # Boolean arguments
        if (arg.startswith('-calc-rsi')):
            calc_rsi = True
        if (arg.startswith('-use-gpu')):
            use_gpu = True
        if (arg.startswith('-normalize')):
            normalize = True
        if (arg.startswith('-verbose')):
            verbose = True
        if (arg.startswith('-use-all-data-to-train')):
            use_all_data_to_train = True
        if (arg.startswith('-revert')):
            revert = True
        if (arg.startswith('-no-tune')):
            no_tune = True

        # Single arguments
        if (arg.startswith('-start-train-date=')):
            start_train_date = arg.split('=')[1
                                              ]
        if (arg.startswith('-start-test-date=')):
            start_test_date = arg.split('=')[1]

        if (arg.startswith('-fold=')):
            fold = int(arg.split('=')[1])

        if (arg.startswith('-n-jobs=')):
            n_jobs = int(arg.split('=')[1])

        # List arguments
        if (arg.startswith('-symbol-list=')):
            aux = arg.split('=')[1]
            symbol_list = aux.split(',')

        if (arg.startswith('-estimator-list=')):
            aux = arg.split('=')[1]
            estimator_list = aux.split(',')

        if (arg.startswith('-stop-loss-list=')):
            aux = arg.split('=')[1]
            stop_loss_list = aux.split(',')

        if (arg.startswith('-numeric-features=')):
            aux = arg.split('=')[1]
            numeric_features_list = prepare_numeric_features_list(aux.split(','))

        if (arg.startswith('-regression-PnL-list=')):
            aux = arg.split('=')[1]
            times_regression_PnL_list = aux.split(',')

        if (arg.startswith('-regression-times-list=')):
            aux = arg.split('=')[1]
            regression_times_list = aux.split(',')

        if (arg.startswith('-regression-features=')):
            aux = arg.split('=')[1]
            regression_features_list = combine_list(aux.split(','))

    train(calc_rsi, use_gpu, normalize, verbose, use_all_data_to_train, revert, no_tune, start_train_date, start_test_date, fold, n_jobs, symbol_list,
          estimator_list, stop_loss_list, numeric_features_list, times_regression_PnL_list, regression_times_list, regression_features_list)


if __name__ == '__main__':
    main(sys.argv[1:])
