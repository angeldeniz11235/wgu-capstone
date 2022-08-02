from ntpath import realpath
import os
from sqlite3 import Timestamp
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def get_stock_data(symbol_, start_, end_, result={}):
    try:
        #check if timeframe is set, if not set it to minute
        #hack to set a default timeframe
        timeframe = TimeFrame.Minute
            
        #convert start_date and end_date to datetime objects
        start_date = dt.datetime.strptime(start_, '%Y-%m-%d')
        end_date = dt.datetime.strptime(end_, '%Y-%m-%d')
        
        print("Creating model for: " + symbol_ + " from " + start_date.strftime("%Y-%m-%d") + " to " + end_date.strftime("%Y-%m-%d"))
        # Import the API keys
        api_id = "PKX7LJVBEOAWQB6L7HSR"
        api_secret = "tz7PzLbQgH3oFcrTUZSWTholaaOmQECPnIz5MbWV"

        # Create the REST API object for paper trading
        rest_api = REST(api_id, api_secret,'https://paper-api.alpaca.markets')
        
        print("Getting data for:", symbol_)
        # Get bar data from the API
        #if dates are the same, and dates are today, subtract 15 minutes from end date
        bar_data = []
        if dt.datetime.now().strftime("%Y-%m-%d") == start_date.strftime("%Y-%m-%d"):
            #start_date is beginning of the day, end_date is end of the day
            start_date = dt.datetime.combine(start_date, dt.time(0, 0, 0))
            end_date = (dt.datetime.now() - dt.timedelta(minutes=15)) 
            bar_data = rest_api.get_bars(symbol=symbol_, timeframe=timeframe, start=start_date.isoformat("T")+"Z", end=end_date.isoformat("T")+"Z")
        #if end date is today but start time is not today, subtract 15 minutes from end date
        elif (dt.datetime.now().strftime("%Y-%m-%d") == end_date.strftime("%Y-%m-%d")) and not (dt.datetime.now().strftime("%Y-%m-%d") == start_date.strftime("%Y-%m-%d")): 
            end_date = (dt.datetime.now() - dt.timedelta(minutes=15))
            bar_data = rest_api.get_bars(symbol=symbol_, timeframe=timeframe, start=start_date.isoformat("T")+"Z", end=end_date.isoformat("T")+"Z")
        else: 
            bar_data = rest_api.get_bars(symbol=symbol_, timeframe=timeframe, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

        #create plot from bar_data and save it to a file
        # adjust figure size
        plt.rcParams['figure.figsize'] = (10, 8)
        # show a plot of the data
        plot = bar_data.df.plot(use_index=True, y='open' )
        plot.add_line(plt.Line2D(bar_data.df.index, bar_data.df.close, color='red', linewidth=1, label='close'))
        # title the axes
        plot.set_title('Apple Stock Price')
        plot.set_ylabel('Price')
        plot.set_xlabel('Date')
        # save the plot to a file
        fig = plot.get_figure()
        img_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static/resources/img/')
        fig.savefig(img_dir + symbol_ + '_plot.png')
        result['plot_img'] =  symbol_ + '_plot.png'
        
        return bar_data
    
    except Exception as e:
        print("Error getting data for:", symbol_)
        print(e)
        #append error to errorLog.txt
        Timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("errorLog.txt", "a") as f:
            f.write(Timestamp + ": Error getting data for: " + symbol_ + "\n")
            f.write(str(e) + "\n")
        result['error'] = "Error getting data for: " + symbol_

#this function is used to process the data used for prediction       
def process_bar_data(bar_data, symbol_, result={}):
    #set bar_data to the dataframe from bar_data.df
        bar_data = bar_data.df
        
        print("Processing data for:", symbol_)
        #convert the dataframe index to a column
        bar_data['DateTime'] = pd.to_datetime(bar_data.index) 
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
        
        # set 'gain' column to 1 if column '0' is less than the last non-zero column in the row
        bar_data['gain'] = bar_data.apply(lambda x: True if x[0] < x[x[x>0].last_valid_index()] else False, axis=1)

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

        return bar_data
        
def process_data_for_new_model(symbol_, start_date, end_date, result={}):
    
    try:
        # Get bar data for the stock
        bar_data = get_stock_data(symbol_, start_date, end_date)
        
        #save the bar_data plot png name to result
        result['plot_img'] =  symbol_ + '_plot.png'
        
        #if bar_data is not empty, then save the data
        if len(bar_data) > 0:
            # Save the apple_bars dataframe to a pickle file
            bar_data.df.to_pickle(f"data/{symbol_}_bar_data.pkl")
        # else, return an error message
        else:
            print("No data found for:", symbol_)
            #append error to errorLog.txt
            Timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("errorLog.txt", "a") as f:
                f.write(Timestamp + ": Error getting data for: " + symbol_ + "\n")
                f.write(str(e) + "\n")
            result['error'] = "No data found for: " + symbol_
            return
        
        #set bar_data to the dataframe from bar_data.df
        bar_data = bar_data.df
        
        print("Processing data for:", symbol_)
        #convert the dataframe index to a column
        bar_data['DateTime'] = pd.to_datetime(bar_data.index) 
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
        #append error to errorLog.txt
        Timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("errorLog.txt", "a") as f:
            f.write(Timestamp + ": Error getting data for: " + symbol_ + "\n")
            f.write(str(e) + "\n")
        result['success'] = False
        result['error'] = str(e)
    

def train_model(symbol_, nn_layers=[], result={}):
    
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
        
        # create the neural network architecture
        model = Sequential()
        
        # add the first layer
        model.add(Dense(units=360, activation=activations.gelu, input_shape=(X_train.shape[1],)))
        
        # check if the nn_layers is empty
        if nn_layers != []:
            for num_nodes in nn_layers:
                model.add(Dense(units=num_nodes, activation=activations.gelu))
                model.add(Dropout(0.2))    
        else:
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
        
        # save the model history for the matplotlib plot
        model_history = model.history.history
        
        # if model.predict(X_test) > 0.44 then 1 else 0:
        predictions = model.predict(X_test) #this clears the model.history object
        offset = .00
        predictions[predictions > (predictions.mean() + offset)] = 1
        predictions[predictions <= (predictions.mean() + offset)] = 0
        
        #save the model
        model.save(f'models/{symbol_}_model_ML.h5')
        print('ML Training - Model saved for:', symbol_)
        
        result['success'] = True

        #save the model accuracy to the result dictionary
        result['model_accuracy'] = str(round(accuracy_score(y_test, predictions) * 100, 2)) + '%'

        #build seaborn heatmap of the confusion matrix for the model and save it to a png file named after the symbol
        fig, ax = plt.subplots(figsize=(10,10))
        ax=sns.set(style="darkgrid")
        canvas = FigureCanvas(fig)
        heatmap = sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues')
        #set labels for the heatmap
        fig.suptitle(f'{symbol_} ML Confusion Matrix', fontsize=20)
        #save the heatmap
        img = io.BytesIO()
        img_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static/resources/img/')
        fig.savefig(f'{img_dir}{symbol_}_ML_heatmap.png', format='png')
        img.seek(0)
        heatmap_img = f'{symbol_}_ML_heatmap.png'
        result['heatmap_img'] = heatmap_img
        
        #build a matlibplot plot of model loss and save it to a png file named after the symbol
        plot = pd.DataFrame(model_history).plot(figsize=(10,10))
        fig = plot.get_figure()
        fig.savefig(f'{img_dir}{symbol_}_ML_loss.png', format='png')
        result['model_loss_img'] = f'{symbol_}_ML_loss.png'
        
        
    except Exception as e:
        print('ML Training - Error:', e)
        #append error to errorLog.txt
        Timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("errorLog.txt", "a") as f:
            f.write(Timestamp + ": Error getting data for: " + symbol_ + "\n")
            f.write(str(e) + "\n")
        result['success'] = False
        result['error'] = str(e)

#this function is used to predict if the stock will gain or lose on a given day based on the model
def predict(symbol_, date_, result={}):
    try:
        #load the model
        model = tf.keras.models.load_model(f'models/{symbol_}_model_ML.h5')
        print('ML Prediction - Model loaded for:', symbol_)
        #download the data (start and end dates are the same) for the given symbol
        bar_data = get_stock_data(symbol_, date_, date_, result)
        bar_data = process_bar_data(bar_data, symbol_, result)
        print('ML Prediction - Data loaded for:', symbol_)
        #get the data for the given date
        X = bar_data.drop('gain', axis=1).values
        y = bar_data['gain'].values
        #scale the data
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        #predict the data
        prediction = model.predict(X)
        #if the prediction is greater than 0.5 then 1 else 0
        if prediction > 0.5:
            result['prediction'] = 'gain'
        else:
            result['prediction'] = 'loss'
        result['success'] = True
        print('ML Prediction - Prediction done for:', symbol_)
        print('ML Prediction - Prediction:', result['prediction'])
    except Exception as e:
        print('ML Prediction - Error:', e)
        #append error to errorLog.txt
        Timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("errorLog.txt", "a") as f:
            f.write(Timestamp + ": Error getting data for: " + symbol_ + "\n")
            f.write(str(e) + "\n")
        result['success'] = False
        result['error'] = str(e)
    
def test():
  #  end = dt.datetime.now().date() - dt.timedelta(days=1)
  #  start = end - dt.timedelta(days=end.day)
  # process_data_for_new_model("aapl", "2022-07-20", "2022-07-28", result={})    
  #train_model("aapl", [750, 100, 30], {}) 
  predict("aapl", "2022-08-02", {}) 
    
if __name__ == "__main__":
    test()