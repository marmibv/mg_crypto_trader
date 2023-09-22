import os
from pathlib import Path

file = Path(__file__)
parent = file.parent
os.chdir(parent)
print(file, parent, os.getcwd())


os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"

import src.train as tr
import src.batch_analysis as ba
import src.start_batch_training as bt
import src.start_robo_trader as bot
import src.myenv as myenv

import src.utils as utils
import sys
import logging
# Now you can use the 'argument' variable in your script
print("Argument provided:", sys.argv[1:])

log = None


def configure_log(log_level=logging.INFO):
  log_file_path = os.path.join(myenv.logdir, "main_process_trader.log")
  '''
    # Create a TimedRotatingFileHandler that rotates every day
    from logging.handlers import TimedRotatingFileHandler
    handler = TimedRotatingFileHandler(
        filename=log_file_path,
        when="midnight",  # Rotate at midnight
        interval=1,        # Daily rotation
        backupCount=7      # Keep up to 7 log files (adjust as needed)
    )
    '''
  logging.basicConfig(
      level=log_level,  # Set the minimum level to be logged
      format="%(asctime)s [%(levelname)s]: %(message)s",
      handlers=[
          logging.FileHandler(log_file_path, mode='a', delay=True),  # Log messages to a file
          logging.StreamHandler()  # Log messages to the console
      ]
  )
  return logging.getLogger("main_trader_log")
def configure_log(log_level):
  log_file_path = os.path.join(myenv.logdir, myenv.main_log_filename)
  logger = logging.getLogger('main_trader_log')
  logger.setLevel(log_level)
  fh = logging.FileHandler(log_file_path, mode='a', delay=True)
  fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
  fh.setLevel(log_level)
  logger.addHandler(fh)
  logger.addHandler(logging.StreamHandler())
  return logger

def main(args):
  log_level = logging.INFO
  interval_list = ['1h']

  # Generic Params
  for arg in args:
    if (arg.startswith('-log-level=DEBUG')):
      log_level = logging.DEBUG
    if (arg.startswith('-log-level=WARNING')):
      log_level = logging.WARNING
    if (arg.startswith('-log-level=INFO')):
      log_level = logging.INFO
    if (arg.startswith('-log-level=ERROR')):
      log_level = logging.ERROR
    if (arg.startswith('-interval-list=')):
      aux = arg.split('=')[1]
      interval_list = aux.split(',')

  log = configure_log(log_level)
  if '-download-data' in args:
    for interval in interval_list:
      log.info(f'Starting download data, in interval ({interval}) for all Symbols in database...')
      utils.download_data(save_database=True, parse_data=False, interval=interval)
    sys.exit(0)

  if '-prepare-best-parameters' in args:
    log.info('Starting prepare-best-parameters...')
    params = utils.prepare_best_params()
    log.info(params)
    sys.exit(0)

  if '-train-model' in args:
    log.info('Starting training...')
    if tr.main(args):
      log.info('Trainin completed ** SUCCESS **')
    else:
      log.info('Trainin ** FAILS **')
    sys.exit(0)

  if '-simule-trading' in args:
    tr.exec_simule_trading(args)
    sys.exit(0)

  if '-run-bot' in args:
    log.info('Starting bot...')
    bot.main(args)
    sys.exit(0)

  if '-batch-analysis' in args:
    log.info('Starting batch analysis...')
    ba.main(args)
    sys.exit(0)

  if '-batch-training' in args:
    log.info('Starting batch training...')
    bt.main(args)
    sys.exit(0)


if __name__ == '__main__':
  main(sys.argv[1:])
