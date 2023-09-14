#!/bin/bash
# This is a basic shell script template

# Author: Marcelo Lima Gomes
# Date: 11/09/2023
sudo swapon /newswap

#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=lr 
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=knn 
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=nb 
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=dt 
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=svm 
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=mlp 
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=ridge #
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=rf #
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=qda
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=ada
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=gbc
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=lda
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=et
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=catboost

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
python . -train-model -calc-rsi -n-jobs=20 -normalize -verbose -all-cols -regression-times=0 -use-all-data-to-train -stop-loss=2 -estimator=mlp -symbol=BTCUSDT
python . -train-model -calc-rsi -n-jobs=20 -normalize -verbose -all-cols -regression-times=0 -use-all-data-to-train -stop-loss=2 -estimator=mlp -symbol=ETHUSDT
python . -train-model -calc-rsi -n-jobs=20 -normalize -verbose -all-cols -regression-times=0 -use-all-data-to-train -stop-loss=2 -estimator=mlp -symbol=BNBUSDT
python . -train-model -calc-rsi -n-jobs=20 -normalize -verbose -all-cols -regression-times=0 -use-all-data-to-train -stop-loss=2 -estimator=mlp -symbol=XRPUSDT
python . -train-model -calc-rsi -n-jobs=20 -normalize -verbose -all-cols -regression-times=0 -use-all-data-to-train -stop-loss=2 -estimator=mlp -symbol=ADAUSDT
python . -train-model -calc-rsi -n-jobs=20 -normalize -verbose -all-cols -regression-times=0 -use-all-data-to-train -stop-loss=2 -estimator=mlp -symbol=DOGEUSDT
python . -train-model -calc-rsi -n-jobs=20 -normalize -verbose -all-cols -regression-times=0 -use-all-data-to-train -stop-loss=2 -estimator=mlp -symbol=SOLUSDT
#python . -train-model -calc-rsi -normalize -verbose -all-cols -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -estimator=knn -stop-loss=1 -use-all-data-to-train
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-06-01 -estimator=knn

!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=xgboost
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=lightgbm
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=rl
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=knn 
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=nb
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=dt 
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=svm 
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=mlp
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=ridge 
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=rf 
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=qda 
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=ada 
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=gbc 
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=lda 
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=et 
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=catboost 
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=rbfsvm
!python /content/drive/MyDrive/crypto_test/. -train-model -calc-rsi -normalize -verbose -numeric-features='close' -regression-features='close' -regression-times=24 -symbol=BTCUSDT -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=gpc

#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=knn 
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=nb 
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=dt 
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=svm 
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=mlp 
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=ridge #
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=rf #
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=qda
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=ada
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=gbc
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=lda
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=et
#python . -train-model -calc-rsi -normalize -verbose -all-cols  -symbol=BTCUSDT -n-jobs=20 -regression-times=0 -start-train-date=2010-01-01 -start-test-date=2023-01-01 -estimator=catboost

exit 0