#Flask web based UI for the main program

#import flask
from concurrent.futures import thread
import os
import threading
from turtle import st
from urllib import response
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_session import Session

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfdaab0ba245'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

session = {}
# redirect from / to /login.html
@app.route('/')
def index():
    #if 'username' in session then redirect to /index.html
    if not 'username' in session:
        return redirect(url_for('login'))
    #get list of models in the /models folder
    dict_of_models = {"models": os.listdir('models')}
    return render_template('index.html', username=session['username'], models=dict_of_models)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username']
        session['password'] = request.form['password']
        if session['username'] == 'admin' and session['password'] == 'admin':
            return redirect(url_for('index'))
        else:
            session['invalid-login'] = True
    return render_template('login.html',session=session)

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    session.pop('password', None)
    session.clear()
    return redirect(url_for('login'))

@app.route('/createmodel', methods=['POST'])
def create_model():
    from ml_models import train_model, process_data_for_new_model
    import datetime as dt
    nn_layers = request.form.getlist('hiddenLayers')
    symbol = request.form['symbol']
    start = request.form['startDate']
    end = request.form['endDate']
    #dictionary to store the results
    mc_res = {}
    #create new thread to create model
    cm_thread = threading.Thread(target=process_data_for_new_model, args=(symbol, start, end, mc_res))
    cm_thread.start()
    cm_thread.join()
    #if model creation was not successful then return error message
    if 'error' in mc_res:
        return jsonify({'status':mc_res['error']})
    
    #dictionary to store the results
    tm_res = {}
    #thread to train model
    tm_thread = threading.Thread(target=train_model, args=(symbol, nn_layers, tm_res))
    tm_thread.start()
    tm_thread.join()
    #if model training was not successful then return error message
    if 'error' in tm_res:
        return jsonify({'status':tm_res['error']})
    
    #if successful then return success message
    # send response to client to let them know that the model was created
    res = {'status': 'Model for ' + symbol + ' created successfully.'}
    
    #add the img paths (plots/charts) to the response
    res['plot_img'] =  mc_res['plot_img']
    res['heatmap_img'] = tm_res['heatmap_img']
    res['model_loss_img'] = tm_res['model_loss_img']

    #add the model accuracy to the response
    res['model_accuracy'] = tm_res['model_accuracy']
    
    #add the model info to the response
    return jsonify(res)
    
#route to check if a date is a valid trading date 
@app.route('/check_valid_date', methods=['POST'])
def check_valid_date():
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar('NYSE')
    start = request.form['date']
    end = start
    nyse_dates = nyse.valid_days(start_date=start, end_date=end) 
    if len(nyse_dates) == 0:
        return jsonify({'valid': 'false'})
    else:
        return jsonify({'valid': 'true'})
    
#route to predict if a stock will go up or down on a given date
@app.route('/predict', methods=['POST'])
def predict():
    from ml_models import predict
    symbol = request.form['symbol']
    date = request.form['date']
    res = {}
    predict(symbol, date, res)
    return jsonify(res)
    
if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='
