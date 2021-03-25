from ml_calculator import MLP_NN, LSTM_NN, RandomForest
from ml_data import prepare_data






print('Select a coin: ')
symbol = input()
df, df_pred = prepare_data(symbol)


def main():
    MLP_Score, MLP_Pred = MLP_NN(df, df_pred)
    LSTM_Score, LSTM_Pred = LSTM_NN(df, df_pred) 
    RF_Score, RF_Pred= RandomForest(df, df_pred)
    
    if MLP_Pred > 0.5:
        MLP = f'MLP Model: Close price of the {symbol} will be higher tomorrow, with {MLP_Score} probability.'
    else:
        MLP = f'MLP Model: Close price of the {symbol} will be lower tomorrow, with {MLP_Score} probability.'
    
    if LSTM_Pred > 0.5:
        LSTM = f'LSTM Model: Close price of the {symbol} will be higher tomorrow, with {LSTM_Score} probability.'
    else:
        LSTM = f'LSTM Model: Close price of the {symbol} will be lower tomorrow, with {LSTM_Score} probability.'
        
    if RF_Pred > 0.5:
        RF = f'Random Forest Model: Close price of the {symbol} will be higher tomorrow, with {RF_Score} probability.'
    else:
        RF = f'Random Forest Model: Close price of the {symbol} will be lower tomorrow, with {RF_Pred} probability.'
        
    
    return print('Result:', '\n',
                 MLP,'\n',
                 LSTM, '\n',
                 RF)
                 
    

if __name__ == "__main__":
    main()