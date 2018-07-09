from flask import Flask, render_template, flash, redirect, url_for, session, request, logging
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from flask_mysqldb import MySQL
# from passlib.hash import sha256_crypt
from functools import wraps
from twit import SentimentAnalysis
import time
import sentiment_module as ml_classifiers
from textblob import TextBlob as tb

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as web
from datetime import datetime as dt

# from twitterconnection import search_tweets

# #################################################################################################
# Steps needed to carry out the project
# 1. Take input from a user in terms of keywords [D]
# 2. Take the amount of tweets needed to be analyzed (default = 5) [D]
# 3. Stream the related tweets and the same amount of tweets
# 4. Process the tweet with some sort of api or method
# 5. Plot the results on the graph
# 6. Return the result effectively and display them
# 7. Extend functionality such as login, database connections, registeration if there is still time
# #################################################################################################

##########################################################################
###                         DEV COMMENTS
### I'm not sure if I should do dash here or make another module for it
###
###########################################################################

# Init Flask
app = Flask(__name__)

# Init Dash
dashApp = dash.Dash(__name__, server=app, url_base_pathname='/dashapp')

############################## Start Dash ####################################

dashApp.layout = html.Div(children=[
html.H1(children='Dash App')])

dashApp.layout = html.Div([
    html.H1('Live Twitter Sentiment'),
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Coke', 'value': 'COKE'},
            {'label': 'Tesla', 'value': 'TSLA'},
            {'label': 'Apple', 'value': 'AAPL'}
        ],
        value='COKE'
    ),
    dcc.Graph(id='my-graph')
])

@dashApp.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    return {
        'data': [{
            'x': [1,4,6,2,6,8,9,4,3,1],
            'y': [23,62,123,52,64,72,108,14,61,71]
        }]
}

############################## End Dash ######################################

# Config MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234'
app.config['MYSQL_DB'] = 'myflaskapp'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# init MYSQL
mysql = MySQL(app)

# Init VaderSentiment
twitter_analyzer = SentimentAnalysis()

# Index route
@app.route('/')
def index():
    return render_template('home.html')

# #########################################################
# PUBLIC Route
# NAME Twitter Sentiment
# DESC Fetch Tweets and Analyze them and plot pi chart
# #########################################################
@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    if request.method == 'POST':
        # Get Form Fields
        input = request.form['keyword']
        quantity = int(request.form['quantity'])
        if len(input) > 0 or quantity > 0 :
        # Fetch tweets from twitter and perform sentiment analysis on the given data
            twitter_analyzer.analyzeTwitter(input, quantity)

            return redirect(url_for('sentiment'))
        else:
            flash('Please enter some text to be analyzed', 'danger')
    return render_template('sentiment.html', title="Twitter Sentiment")


# #########################################################
# PUBLIC Route
# NAME Manual Sentiment
# DESC Allow users to enter data to be analyzed
# #########################################################
@app.route('/manualsentiment', methods=['GET', 'POST'])
def manualSentiment():
    if request.method == 'POST':
        # Get Form Fields
        input = request.form['keyword']
        if len(input) > 0:
        # 1. Get Input from the user
        # 2. Analyze it with the custom classifier, scikit algo, and textblob
        # 3. Return the value 
        # 4. Need to enter custom classifier value
            try:
                # result = cohesive_classify(input)
                # if(result == "pos"):
                #     flash("The sentence has a Positive sentiment", 'success')
                # elif(result == "neg"):
                #     flash("The sentence has a Negative sentiment", 'success')
                # else:
                #     flash("The sentence has an Ambiguous sentiment", 'success')
                result = ml_classifiers.sentiment(input)
                flash('The result is ' + str(result), 'info')
            except:
                flash('A server side error has occured', 'danger')


            return redirect(url_for('manualSentiment', title="Manual Sentiment"))
        else:
            flash('Please enter some text to be analyzed', 'danger')
    return render_template('manualsentiment.html')

def cohesive_classify(text):

    ml_result = ml_classifiers.sentiment(text)[0]
    tb_polarity = tb(text).sentiment.polarity

    if tb_polarity > 0:
        tb_result = "pos"
    elif tb_polarity < 0:
        tb_result = "neg"
    else:
        tb_result = "ambiguous"

    # customer_result = customer_classifier.classify(text)

    if tb_result and ml_result == 'pos':
        result = "pos"
    elif tb_result and ml_result == 'neg':
        result = "neg"
    else:
        result = "ambiguous"

    return result



# Twitter Analysis route
# @app.route('/twitter_analysis', methods=['GET', 'POST'])
# def twitter_analysis():
#     # Todo implementatoin here
#     render_template('twitter_analysis.html')

# Sentiment Analysis function
# def sentiment_of(someinput):
#     vader_sentiment = analyzer.polarity_scores(someinput)
#     if not vader_sentiment['neg'] > 0.1:
#         if vader_sentiment['pos']-vader_sentiment['neg'] > 0:
#             # This line does not seem right, we should not consider just the
#             # positive value but the overall compoud value and somehow make a score.
#             return ("Positive sentiment with a sentiment score of %s%%" % (vader_sentiment['pos']*100))
#     elif not vader_sentiment['pos'] > 0.1:
#         if vader_sentiment['pos']-vader_sentiment['neg'] <= 0:
#             # This line does not seem right, we should not consider just the
#             # negative value but the overall compoud value and somehow make a score.
#             return ("Negative sentiment with a sentiment score of %s%%" % (vader_sentiment['neg']*100))
#     else:
#         return ("This might be a neutral sentiment lol.")

class RegisterForm(Form):
    name = StringField('Name', [validators.Length(min=1, max=50)])
    username = StringField('Username', [validators.Length(min=4, max=25)])
    email = StringField('Email', [validators.Length(min=6, max=50)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords do not match')
    ])
    confirm = PasswordField('Confirm Password')


# Register route
# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     form = RegisterForm(request.form)
#     if request.method == 'POST' and form.validate():
#         # Todo implementation here
#         name = form.name.data
#         email = form.email.data
#         username = form.username.data
#         password = sha256_crypt.encrypt(str(form.password.data))

#         # create cursor
#         cur = mysql.connection.cursor()

#         # execute query
#         cur.execute('INSERT INTO users(name, email, username, password) VALUES(%s, %s, %s, %s)' , (name, email, username, password))

#         # commit to DB
#         mysql.connection.commit()

#         # close cursor
#         cur.close()

#         # display success
#         flash('You have successfully registered!', 'success')

#         return redirect(url_for('index'))
#     return render_template('register.html', form=form)

# login route
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         # Get form fields
#         username = request.form['username']
#         password_candidate = request.form['password']

#         # Create cursor
#         cur = mysql.connection.cusror()

#         # Get user by Username
#         result = cur.execute('SELECT * FROM users WHERE username = %s', [username])

#         if result > 0:
#             # Get stored hash
#             data = cur.fetchone()
#             password = data['password']

#             # Compare Passwords
#             if sha256_crypt.verify(password_candidate, password):
#                 # Passed
#                 session['logged_in'] = True
#                 session['username'] = username

#                 flash('You are now logged in', 'success')
#                 return redirect(url_for('dashboard'))
#             else:
#                 error = 'Invalid login'
#                 return render_template('login.html', error=error)
#             # Close connection
#             cur.close()
#         else:
#             error = 'Username not found'
#             return render_template('login.html', error=error)


#         return redirect(url_for('index'))
#     return render_template('login.html')

# Dashboard route
# @app.route('/dashboard')
# def dashboard():
#     return render_template('dashboard.html')



# Call main method/run app
if __name__ == '__main__':
    app.secret_key = 'secret123'
    app.run(debug=True, use_reloader=False)
