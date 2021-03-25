import numpy as np
import pandas as pd

from ml_data import prepare_data


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense





#-------------------- Models
#LSTM
def LSTM_NN(df, df_pred):
    df_X = df.drop('goes_up', axis=1)
    df_y = df['goes_up']
    X = df_X.to_numpy()
    y = df_y.to_numpy()
    pred_arr = df_pred.to_numpy()
    pred_arr  = np.nan_to_num(pred_arr)
    
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    pred_arr = np.reshape(pred_arr, (pred_arr.shape[0], 1, pred_arr.shape[1]))
    
    lstm = Sequential()

    lstm.add(LSTM(units=50, return_sequences = True, input_shape=(1, 11)))
    lstm.add(Dropout(0.2))

    lstm.add(LSTM(units=50, return_sequences = True))
    lstm.add(Dropout(0.2))

    lstm.add(LSTM(units=50))
    lstm.add(Dropout(0.2))

    lstm.add(Dense(units=1))

    lstm.compile(optimizer='adam', loss='mean_squared_error')

    lstm.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.10)
    
    df_LSTM = lstm.predict(pred_arr)
    
    return roc_auc_score(y_test, lstm.predict(X_test)), df_LSTM

#MLP
def MLP_NN(df, df_pred):
    df_X = df.drop('goes_up', axis=1)
    df_y = df['goes_up']
    X = df_X.to_numpy()
    y = df_y.to_numpy()
    pred_arr = df_pred.to_numpy()
    pred_arr  = np.nan_to_num(pred_arr)
    
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    pred_arr = np.reshape(pred_arr, (pred_arr.shape[0], 1, pred_arr.shape[1]))
    
    MLP = Sequential([
        Dense(100, input_shape=(1, 11), activation='sigmoid'),
        Dense(100, activation='sigmoid'),
        Dense(100, activation='sigmoid'),
        Dense(2, activation='softmax')
        ])
    
    MLP.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    
    MLP.fit(X_train, y_train, epochs=20)
    
    predictions = MLP.predict(X_test)
    y_pred = [predictions[i][0][1] for i in range(len(predictions))]
    y_pred = np.array(y_pred)
    
    df_MLP = MLP.predict(pred_arr)
    
    return roc_auc_score(y_test, y_pred), df_MLP[0][0][1]

def RandomForest(df, df_pred):
    df_X = df.drop('goes_up', axis=1)
    df_y = df['goes_up']
    X = df_X.to_numpy()
    y = df_y.to_numpy()
    pred_arr = df_pred.to_numpy()
    pred_arr  = np.nan_to_num(pred_arr)
    
    
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=42)
    
    #params = {'criterion':('gini', 'entropy'),
    #      'max_depth':(3, 5, 7, 9),
    #      'min_samples_leaf':(3, 5, 8, 10)
    #      }

    #clf = GridSearchCV(rfc(), param_grid=params, cv=5)
    #clf.fit(X_train, y_train)
    #clf.best_params_
    
    model_rfc = rfc(criterion='entropy', max_depth=5, random_state=0, min_samples_leaf=5)
    model_rfc.fit(X_train, y_train)
    
    df_RFC = model_rfc.predict(pred_arr)
    
    return roc_auc_score(y_test, model_rfc.predict(X_test)), df_RFC
    

    







    

    
    





