#Flask web based UI for the main program

#import flask
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

#@app.route('/home')


if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='
