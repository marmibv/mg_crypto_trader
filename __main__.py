import src.robo as bot
import sys

# Now you can use the 'argument' variable in your script
print("Argument provided:", sys.argv[1:])

bot.main(sys.argv[1:])
