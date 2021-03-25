from crypto_api import crypto_data
import numpy as np
import pandas as pd


from pyti.simple_moving_average import simple_moving_average as sma
from pyti.exponential_moving_average import exponential_moving_average as ema
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as macd
from pyti.stochastic import percent_k, percent_d
from pyti.on_balance_volume import on_balance_volume as obv


def prepare_data(symbol):
    
    coin = symbol
    coin_1 = crypto_data(coin)
    df = coin_1
    
    df['SMA_20'] = sma(df['Close'], 20)
    df['SMA_50'] = sma(df['Close'], 50)

    df['EMA_20'] = ema(df['Close'], 20)
    df['EMA_50'] = ema(df['Close'], 50)

    df['MACD'] = macd(df['Close'], 26, 12)

    df['per_k_stoch_10'] = percent_k(df['Close'], 10)
    df['per_d_stoch_10'] = percent_d(df['Close'], 10)

    df['OBV'] = obv(df['Close'], df['Volume'])
    
    
    
    fp = []
    for price in df['Close']:
        fp.append(price)

    fp.pop(0)
    fp.append(df['Close'].mean())
    df['FP'] = fp

    df_predict = df.tail(1)
    df.drop(df.tail(1).index,inplace=True)
    
    label = []
    for i in range(len(df)):
        if df['FP'][i] > df['Close'][i]:
            label.append(1)
        else:
            label.append(0)
        
    df['goes_up'] = label
    df = df.drop(['FP'],axis=1)
    df = df.fillna(df.mean())
    df_predict = df_predict.drop(['FP'],axis=1)
    
    return df, df_predict
    
    
    
    

