#!/bin/bash
# This is a basic shell script template

# Author: Marcelo Lima Gomes
# Date: 11/09/2023

DATE_FORMATED=$(date +%Y%m%d_%H%M%S)

echo "Starting bot at $(date +%Y-%m-%d_%H:%M:%S)..."

source .env/bin/activate

python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn -symbol=ADAUSDT >> bot_ADAUSDT.log
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn -symbol=AVAXUSDT >> bot_AVAXUSDT.log
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=1.5 -estimator=knn -symbol=BCHUSDT >> bot_BCHUSDT.log
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=1.5 -estimator=knn -symbol=BNBUSDT >> bot_BNBUSDT.log
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn -symbol=BTCUSDT >> bot_BTCUSDT.log
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=1.0 -estimator=knn -symbol=DOGEUSDT >> bot_DOGEUSDT.log
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn -symbol=ETHUSDT >> bot_ETHUSDT.log
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn -symbol=MATICUSDT >> bot_MATICUSDT.log
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=3.0 -estimator=knn -symbol=SOLUSDT >> bot_SOLUSDT.log
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=1.5 -estimator=knn -symbol=TRXUSDT >> bot_TRXUSDT.log
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=3.0 -estimator=knn -symbol=XRPUSDT >> bot_XRPUSDT.log

exit 0
