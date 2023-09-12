#!/bin/bash
# This is a basic shell script template

# Author: Marcelo Lima Gomes
# Date: 11/09/2023
sudo swapon /newswap

python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=lr 
python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=knn 
python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=nb 
python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=dt 
python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=svm 
python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=mlp 
python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=ridge #
python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=rf #
python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=qda
python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=ada
python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=gbc
python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=lda
python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=et
python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=catboost

# Consome muita memória. Não executa em desktop
# python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=gpc 

#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=xgboost 
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=lightgbm 
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=rbfsvm 

# Utilizar all-data para determinadas moedas
# BTC
# ETH
# BNB
# XRP
# ADA
# DOGE
# SOL
python . -train-model -calc-rsi -n-jobs=20 -normalize -verbose -all-cols -regression-times=0 -use-all-data-to-train -stop-loss=1 -estimator=knn -symbol=BTCUSDT
python . -train-model -calc-rsi -n-jobs=20 -normalize -verbose -all-cols -regression-times=0 -use-all-data-to-train -stop-loss=1 -estimator=knn -symbol=ETHUSDT
python . -train-model -calc-rsi -n-jobs=20 -normalize -verbose -all-cols -regression-times=0 -use-all-data-to-train -stop-loss=1 -estimator=knn -symbol=BNBUSDT
python . -train-model -calc-rsi -n-jobs=20 -normalize -verbose -all-cols -regression-times=0 -use-all-data-to-train -stop-loss=1 -estimator=knn -symbol=XRPUSDT
python . -train-model -calc-rsi -n-jobs=20 -normalize -verbose -all-cols -regression-times=0 -use-all-data-to-train -stop-loss=1 -estimator=knn -symbol=ADAUSDT
python . -train-model -calc-rsi -n-jobs=20 -normalize -verbose -all-cols -regression-times=0 -use-all-data-to-train -stop-loss=1 -estimator=knn -symbol=DOGEUSDT
python . -train-model -calc-rsi -n-jobs=20 -normalize -verbose -all-cols -regression-times=0 -use-all-data-to-train -stop-loss=1 -estimator=knn -symbol=SOLUSDT

python . -train-model -calc-rsi -normalize -verbose -all-cols -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -estimator=knn -stop-loss=1 -use-all-data-to-train

python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-06-01 -estimator=knn

exit 0