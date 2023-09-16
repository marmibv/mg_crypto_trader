#!/bin/bash
# This is a basic shell script template

# Author: Marcelo Lima Gomes
# Date: 11/09/2023

DATE_FORMATED=$(date +%Y%m%d_%H%M%S)

echo "Starting bot at $(date +%Y-%m-%d_%H:%M:%S)..."

#python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn -symbol=ADAUSDT > "bot_ADAUSDT_${DATE_FORMATED}.log" > "bot_ADAUSDT_${DATE_FORMATED}.log" &
#python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=1.5 -estimator=knn -symbol=BNBUSDT > "bot_BNBUSDT_${DATE_FORMATED}.log" > "bot_BNBUSDT_${DATE_FORMATED}.log" & 
#python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn -symbol=BTCUSDT > "bot_BTCUSDT_${DATE_FORMATED}.log" > "bot_BTCUSDT_${DATE_FORMATED}.log" & 
#python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=1 -estimator=knn -symbol=DOGEUSDT > "bot_DOGEUSDT_${DATE_FORMATED}.log" > "bot_DOGEUSDT_${DATE_FORMATED}.log" & 
#python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn -symbol=ETHUSDT > "bot_ETHUSDT_${DATE_FORMATED}.log" > "bot_ETHUSDT_${DATE_FORMATED}.log"  & 
#python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=3 -estimator=knn -symbol=SOLUSDT > "bot_SOLUSDT_${DATE_FORMATED}.log" > "bot_SOLUSDT_${DATE_FORMATED}.log" &
#python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=3 -estimator=knn -symbol=XRPUSDT > "bot_XRPUSDT_${DATE_FORMATED}.log" > "bot_XRPUSDT_${DATE_FORMATED}.log" &


ssh -f marcelo@192.168.31.161 '/home/marcelo/des/mg_crypto_trader/.env/bin/python3.10 /home/marcelo/des/mg_crypto_trader/. -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn  -symbol=ADAUSDT > /home/marcelo/des/mg_crypto_trader/bot_ADAUSDT.log'
ssh -f marcelo@192.168.31.161 '/home/marcelo/des/mg_crypto_trader/.env/bin/python3.10 /home/marcelo/des/mg_crypto_trader/. -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=1.5 -estimator=knn -symbol=BNBUSDT > /home/marcelo/des/mg_crypto_trader/bot_BNBUSDT.log'
ssh -f marcelo@192.168.31.161 '/home/marcelo/des/mg_crypto_trader/.env/bin/python3.10 /home/marcelo/des/mg_crypto_trader/.  -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn -symbol=BTCUSDT > /home/marcelo/des/mg_crypto_trader/bot_BTCUSDT.log' 
ssh -f marcelo@192.168.31.161 '/home/marcelo/des/mg_crypto_trader/.env/bin/python3.10 /home/marcelo/des/mg_crypto_trader/.  -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=1 -estimator=knn -symbol=DOGEUSDT > /home/marcelo/des/mg_crypto_trader/bot_DOGEUSDT.log' 
ssh -f marcelo@192.168.31.161 '/home/marcelo/des/mg_crypto_trader/.env/bin/python3.10 /home/marcelo/des/mg_crypto_trader/.  -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn -symbol=ETHUSDT > /home/marcelo/des/mg_crypto_trader/bot_ETHUSDT.log' 
ssh -f marcelo@192.168.31.161 '/home/marcelo/des/mg_crypto_trader/.env/bin/python3.10 /home/marcelo/des/mg_crypto_trader/.  -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=3 -estimator=knn -symbol=SOLUSDT > /home/marcelo/des/mg_crypto_trader/bot_SOLUSDT.log'
ssh -f marcelo@192.168.31.161 '/home/marcelo/des/mg_crypto_trader/.env/bin/python3.10 /home/marcelo/des/mg_crypto_trader/.  -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=3 -estimator=knn -symbol=XRPUSDT > /home/marcelo/des/mg_crypto_trader/bot_XRPUSDT.log'
ssh -f marcelo@192.168.31.161 '/home/marcelo/des/mg_crypto_trader/.env/bin/python3.10 /home/marcelo/des/mg_crypto_trader/.  -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn -symbol=AVAXUSDT > /home/marcelo/des/mg_crypto_trader/bot_AVAXUSDT.log'
ssh -f marcelo@192.168.31.161 '/home/marcelo/des/mg_crypto_trader/.env/bin/python3.10 /home/marcelo/des/mg_crypto_trader/.  -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=1.5 -estimator=knn -symbol=BCHUSDT > /home/marcelo/des/mg_crypto_trader/bot_BCHUSDT.log'
ssh -f marcelo@192.168.31.161 '/home/marcelo/des/mg_crypto_trader/.env/bin/python3.10 /home/marcelo/des/mg_crypto_trader/.  -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn -symbol=MATICUSDT > /home/marcelo/des/mg_crypto_trader/bot_MATICUSDT.log'
ssh -f marcelo@192.168.31.161 '/home/marcelo/des/mg_crypto_trader/.env/bin/python3.10 /home/marcelo/des/mg_crypto_trader/.  -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=1.5 -estimator=knn -symbol=TRXUSDT > /home/marcelo/des/mg_crypto_trader/bot_TRXUSDT.log'

exit 0