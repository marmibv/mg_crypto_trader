import src.robo as bot
import src.train as tr
import time
import sys
import traceback

# Now you can use the 'argument' variable in your script
print("Argument provided:", sys.argv[1:])


def main(args):
    while True:
        try:
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

            for arg in args:
                if (arg.startswith('-run-bot')):
                    print('Starting bot...')
                    bot.main(args)

        except Exception as e:
            traceback.print_exc()
            time.sleep(60)


if __name__ == '__main__':
    main(sys.argv[1:])
