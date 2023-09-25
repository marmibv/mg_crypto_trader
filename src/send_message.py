import datetime
import src.myenv as myenv
import httpx
import logging


def send_to_telegram(message):
  apiToken = myenv.telegram_key[0]
  chatID = '@mgcryptotrader'
  apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

  try:
    logging.getLogger().info(message)
    if myenv.producao:
      now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      with httpx.Client() as client:
        client.post(apiURL, json={'chat_id': chatID, 'text': f'[{now}]: {message}'})

      # response = await requests.post(apiURL, json={'chat_id': chatID, 'text': f'[{now}]: {message}'})
      # print(response.text)
  except Exception as e:
    logging.getLogger().exception(e)


def send_status_to_telegram(message):
  apiToken = myenv.telegram_key[0]
  chatID = '@statusmgcrypto'
  apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

  try:
    logging.getLogger().info(message)
    if myenv.producao:
      now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      with httpx.Client() as client:
        client.post(apiURL, json={'chat_id': chatID, 'text': f'[{now}]: {message}'})

      # response = await requests.post(apiURL, json={'chat_id': chatID, 'text': f'[{now}]: {message}'})
      # print(response.text)
  except Exception as e:
    logging.getLogger().exception(e)
