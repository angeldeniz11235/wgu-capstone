#Flask web based UI for the main program

#import flask
from concurrent.futures import thread
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
    return render_template('index.html', username=session['username'])

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
    from ml_models import train_model, create_model
    import datetime as dt
    symbol = request.form['symbol']
    start = request.form['startDate']
    end = request.form['endDate']
    #dictionary to store the results
    mc_res = {}
    #create new thread to create model
    cm_thread = threading.Thread(target=create_model, args=(symbol, start, end, mc_res))
    cm_thread.start()
    cm_thread.join()
    #if model creation was not successful then return error message
    if 'error' in mc_res:
        return jsonify({'status':mc_res['error']})
    
    #dictionary to store the results
    tm_res = {}
    #thread to train model
    tm_thread = threading.Thread(target=train_model, args=(symbol,tm_res))
    tm_thread.start()
    tm_thread.join()
    #if model training was not successful then return error message
    if 'error' in tm_res:
        return jsonify({'status':tm_res['error']})
    
    #if successful then return success message
    # send response to client to let them know that the model was created
    res = {'status': 'Model for ' + symbol + ' created successfully.'}
    res['heatmap_url'] = tm_res['heatmap_url']
    return jsonify(res)
    

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='
