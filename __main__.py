import src.robo as bot
import src.train as tr
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
                    tr.main(args)

            for arg in args:
                if (arg.startswith('-run-bot')):
                    print('Starting bot...')
                    bot.main(args)

        except Exception as e:
            traceback.print_exc()


if __name__ == '__main__':
    main(sys.argv[1:])
