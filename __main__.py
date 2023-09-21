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

import src.utils as utils
import sys
# Now you can use the 'argument' variable in your script
print("Argument provided:", sys.argv[1:])


def main(args):
  for arg in args:
    if (arg.startswith('-prepare-top-parameters')):
      print('Starting prepare-top-parameters...')
      params = utils.prepare_top_params()
      print(params)
      sys.exit(0)

  for arg in args:
    if (arg.startswith('-train-model')):
      print('Starting training...')
      if tr.main(args):
        print('Trainin completed ** SUCCESS **')
      else:
        print('Trainin ** FAILS **')
      sys.exit(0)

  for arg in args:
    if (arg.startswith('-simule-trading')):
      tr.exec_simule_trading(args)
      sys.exit(0)

  for arg in args:
    if (arg.startswith('-run-bot')):
      print('Starting bot...')
      bot.main(args)
      sys.exit(0)

  for arg in args:
    if (arg.startswith('-batch-analysis')):
      print('Starting batch analysis...')
      ba.main(args)
      sys.exit(0)

  for arg in args:
    if (arg.startswith('-batch-training')):
      print('Starting batch training...')
      bt.main(args)
      sys.exit(0)


if __name__ == '__main__':
  main(sys.argv[1:])
