from ntpath import realpath
import os
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.stream import Stream
import datetime as dt
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io 
import base64
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras import layers
from tensorflow.keras import activations
from sklearn.metrics import confusion_matrix, classification_report

def create_model(symbol_, start_date, end_date, result={}):
    
    try:
        #convert start_date and end_date to datetime objects
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
        
        print("Creating model for: " + symbol_ + " from " + start_date.strftime("%Y-%m-%d") + " to " + end_date.strftime("%Y-%m-%d"))
        # Import the API keys
        api_id = "PKLXRG5MYKH326I8J1DV"
        api_secret = "TxPfLNWc7HlulXh17NLCX0ser8wrlqXmbjYKriiB"

        # Create the REST API object for paper trading
        rest_api = REST(api_id, api_secret,'https://paper-api.alpaca.markets')
        
        print("Getting data for:", symbol_)
        # Get bar data for apple stock from the last 2 years
        bar_data = rest_api.get_bars(symbol_, TimeFrame.Minute, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

        #if bar_data is not empty, then save the data
        if len(bar_data) > 0:
            # Save the apple_bars dataframe to a pickle file
            bar_data.df.to_pickle(f"data/{symbol_}_bar_data.pkl")
        # else, return an error message
        else:
            print("No data found for:", symbol_)
            result['error'] = "No data found for: " + symbol_
            return
        
        #set bar_data to the dataframe from bar_data.df
        bar_data = bar_data.df
        
        print("Processing data for:", symbol_)
        #convert the dataframe index to a column
        bar_data['DateTime'] = bar_data.index.to_pydatetime()
        # remove the index from the dataframe
        bar_data = bar_data.reset_index(drop=True)
        
        
        #drop all columns except for the DateTime and the vwap columns
        bar_data = bar_data.drop(columns=['open', 'high', 'low', 'close', 'volume','trade_count'])

        #create an array of all the unique dates in the dataframe
        dates = bar_data['DateTime'].dt.date.unique()
        # make dates to list
        dates = dates.tolist()
        dates = list(map(lambda x: x.strftime('%Y-%m-%d'), dates))
        
        # in a new dataframe, for each date in the array, create a columns for every vwap value for that date
        tmp_df = pd.DataFrame(index=dates, columns=range(0, 1000))

        # get all vwap values for each date
        for date in dates:
            # get the vwap values for that date
            vwap = bar_data[bar_data['DateTime'].dt.strftime('%Y-%m-%d') == date]['vwap'].tolist()
            # fill vwap list with zeros to match the length of the columns
            vwap = vwap + [0] * (1000 - len(vwap))
            # create a column for that date
            tmp_df.loc[date] = vwap
        # set bar_data to the tmp_df
        bar_data = tmp_df

        #save the dataframe to a pickle file
        bar_data.to_pickle(f'data/{symbol_}_df.pkl')
        
        # set 'gain' column to 1 if column '0' is less than the last non-zero column in the row
        bar_data['gain'] = bar_data.apply(lambda x: True if x[0] < x[x[x>0].last_valid_index()] else False, axis=1)

        # pickle the dataframe
        bar_data.to_pickle(f'data/{symbol_}_df_ML.pkl')
        
        #drop index column
        bar_data.reset_index(drop=True, inplace=True)
        
        
        # function drops any rows with more than 40% 0 values
        def drop_rows_with_many_zeros(df):
            counts =(df==0).astype(int).sum(axis=1)
            df = df[counts < len(df.columns)*0.4]
            return df

        # function that replaces any nan values with the last non-nan value
        def replace_nan_with_last_non_nan(df):
            for row_indx in range(len(df)):
                last_non_nan_value = df.iloc[row_indx,0]
                for col_indx, val in enumerate(df.iloc[row_indx,:]):
                    if not pd.notna(val):
                        df.iloc[row_indx,col_indx] = last_non_nan_value
                    else:
                        last_non_nan_value = val
            return df

        #only keep values that are not equal to 0 except for the 'gain' column
        nan_value = float('NaN')

        #replace all 0s with NaN
        bar_data.replace(0, nan_value, inplace=True)

        #drop rows with more than 40% 0 values
        bar_data = drop_rows_with_many_zeros(bar_data)

        #replace nan values with the last non-nan value
        bar_data = replace_nan_with_last_non_nan(bar_data)


        # for the 'gain' column, replace all True with 1 and all False with 0
        bar_data.replace(True, 1, inplace=True)
        bar_data.replace(False, 0, inplace=True)

        # remove all columns between 360 and 999
        bar_data.drop(bar_data.columns[360:1000], axis=1, inplace=True)

        # save the dataframe to a pickle file
        bar_data.to_pickle(f'data/{symbol_}_df_ML.pkl')
        
        print("Data for:", symbol_, "successfully processed")
        result['success'] = True
   
    except Exception as e:
        print('ML Training - Error:', e)
        result['success'] = False
        result['error'] = str(e)
    

def train_model(symbol_, result={}):
    
    try:
        #load apple_df_ML.pkl
        bar_data = pickle.load(open(f'data/{symbol_}_df_ML.pkl', 'rb'))
        print('ML Training - Data loaded for:', symbol_)
        X = bar_data.drop('gain', axis=1).values
        y = bar_data['gain'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1204)
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = Sequential()

        model.add(Dense(units=750, activation=activations.gelu))
        model.add(Dropout(0.02))

        model.add(Dense(units=1, activation=activations.sigmoid))

        #binary_crossentropy
        model.compile(loss='binary_crossentropy', optimizer='adam')

        #train the model
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        
        print('ML Training - Beginning Model training for:', symbol_)
        #train the model
        model.fit(x=X_train, y=y_train, epochs=1000, batch_size=256, validation_data=(X_test, y_test), verbose=1, callbacks=[callback])
        print('ML Training - Done Model trained for:', symbol_)
        
        # if model.predict(X_test) > 0.44 then 1 else 0:
        predictions = model.predict(X_test)
        offset = .00
        predictions[predictions > (predictions.mean() + offset)] = 1
        predictions[predictions <= (predictions.mean() + offset)] = 0
        
        #save the model
        model.save(f'models/{symbol_}_model_ML.h5')
        print('ML Training - Model saved for:', symbol_)
        
        result['success'] = True

        #build seaborn heatmap of the confusion matrix for the model
        fig, ax = plt.subplots(figsize=(10,10))
        ax=sns.set(style="darkgrid")
        canvas = FigureCanvas(fig)
        sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues')
        #save the heatmap
        img = io.BytesIO()
        img_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static/resources/img/')
        fig.savefig(f'{img_dir}{symbol_}_ML_heatmap.png', format='png')
        img.seek(0)
        heatmap_img = f'{symbol_}_ML_heatmap.png'
        result['heatmap_img'] = heatmap_img
        
        
    except Exception as e:
        print('ML Training - Error:', e)
        result['success'] = False
        result['error'] = str(e)
    
def test():
  #  end = dt.datetime.now().date() - dt.timedelta(days=1)
  #  start = end - dt.timedelta(days=end.day)
  #  create_model("dfsdews", start, end)    
  train_model("aapl")  
    
if __name__ == "__main__":
    test()