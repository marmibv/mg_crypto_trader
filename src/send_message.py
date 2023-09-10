import requests
import datetime


def send_to_telegram(message):
    producao = True

    apiToken = '5946293152:AAEIR1M3K_hriLGW3DkxWTI_5uaAV-4oNbU'
    chatID = '@mgcryptotrader'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

    try:
        if producao:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response = requests.post(apiURL, json={'chat_id': chatID, 'text': f'[{now}]: {message}'})
            print(response.text)
        print(message)
    except Exception as e:
        print(e)


def send_status_to_telegram(message):
    producao = False

    apiToken = '5946293152:AAEIR1M3K_hriLGW3DkxWTI_5uaAV-4oNbU'
    chatID = '@statusmgcrypto'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

    try:
        if producao:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response = requests.post(apiURL, json={'chat_id': chatID, 'text': f'[{now}]: {message}'})
            print(response.text)
        print(message)
    except Exception as e:
        print(e)
