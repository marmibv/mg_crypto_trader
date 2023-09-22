import sys
from src.batch_robo_trader import BatchRoboTrader
from src.utils import *
from src.train import *

def main(args):
  # Boolean arguments
  verbose = False

  # Single arguments
  start_date = '2010-01-01'
  log_level = logging.INFO

  for arg in args:
    # Boolean arguments
    if (arg.startswith('-verbose')):
      verbose = True

    # Single arguments
    if (arg.startswith('-start-date=')):
      start_date = arg.split('=')[1]
      
    if (arg.startswith('-log-level=DEBUG')):
      log_level = logging.DEBUG

    if (arg.startswith('-log-level=WARNING')):
      log_level = logging.WARNING

    if (arg.startswith('-log-level=INFO')):
      log_level = logging.INFO

    if (arg.startswith('-log-level=ERROR')):
      log_level = logging.ERROR

  brt = BatchRoboTrader(
      verbose,
      start_date,
      log_level)
  brt.run()


if __name__ == '__main__':
  main(sys.argv[1:])
