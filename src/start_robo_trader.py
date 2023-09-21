import sys
from src.batch_robo_trader import BatchRoboTrader
from src.utils import *
from src.train import *

import src.myenv as myenv

import logging
logger = None


def configure_log(log_level):
  log_file_path = os.path.join(myenv.logdir, myenv.batch_robo_log_filename)
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
  return logging.getLogger("batch_robo_logger")


def main(args):
  # Boolean arguments
  verbose = False

  # Single arguments
  start_date = '2010-01-01'
  log_level = logging.DEBUG

  for arg in args:
    # Boolean arguments
    if (arg.startswith('-verbose')):
      verbose = True

    # Single arguments
    if (arg.startswith('-start-date=')):
      start_date = arg.split('=')[1
                                  ]
    if (arg.startswith('-log-level=DEBUG')):
      log_level = logging.DEBUG

    if (arg.startswith('-log-level=WARNING')):
      log_level = logging.WARNING

    if (arg.startswith('-log-level=INFO')):
      log_level = logging.INFO

    if (arg.startswith('-log-level=ERROR')):
      log_level = logging.ERROR

  logger = configure_log(log_level)

  brt = BatchRoboTrader(
      verbose,
      start_date,
      log_level)
  brt.run()


if __name__ == '__main__':
  main(sys.argv[1:])
