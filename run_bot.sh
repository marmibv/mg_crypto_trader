#!/bin/bash
# This is a basic shell script template

# Author: Marcelo Lima Gomes
# Date: 11/09/2023

DATE_FORMATED=$(date +%Y%m%d_%H%M%S)

echo "Starting bot at $(date +%Y-%m-%d_%H:%M:%S)..."

python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn -symbol=ADAUSDT > "bot_ADAUSDT_${DATE_FORMATED}.log" > "bot_ADAUSDT_${DATE_FORMATED}.log" &
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=1.5 -estimator=knn -symbol=BNBUSDT > "bot_BNBUSDT_${DATE_FORMATED}.log" > "bot_BNBUSDT_${DATE_FORMATED}.log" & 
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn -symbol=BTCUSDT > "bot_BTCUSDT_${DATE_FORMATED}.log" > "bot_BTCUSDT_${DATE_FORMATED}.log" & 
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=1 -estimator=knn -symbol=DOGEUSDT > "bot_DOGEUSDT_${DATE_FORMATED}.log" > "bot_DOGEUSDT_${DATE_FORMATED}.log" & 
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=2.5 -estimator=knn -symbol=ETHUSDT > "bot_ETHUSDT_${DATE_FORMATED}.log" > "bot_ETHUSDT_${DATE_FORMATED}.log"  & 
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=3 -estimator=knn -symbol=SOLUSDT > "bot_SOLUSDT_${DATE_FORMATED}.log" > "bot_SOLUSDT_${DATE_FORMATED}.log" &
python . -run-bot -calc-rsi -start-train-date=2023-06-01 -all-cols -regression-times=0 -stop-loss=3 -estimator=knn -symbol=XRPUSDT > "bot_XRPUSDT_${DATE_FORMATED}.log" > "bot_XRPUSDT_${DATE_FORMATED}.log" &

exit 0