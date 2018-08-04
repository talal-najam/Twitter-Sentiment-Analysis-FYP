from flask import Flask, render_template, flash, redirect, url_for, session, request, logging
from flask_mysqldb import MySQL
# from passlib.hash import sha256_crypt (for password hasing in database)
from functools import wraps
from twit import SentimentAnalysis
import time
from sentiment_shifter import should_invert
import sentiment_module as ml_classifiers
from textblob import TextBlob as tb
from forms import RegisterForm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from score_calculator import sentiment_score

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
# import pandas as pd
# pd.core.common.is_list_like = pd.api.types.is_list_like
# from pandas_datareader import data as web

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
# 8. Use the preProcessText function 
# #################################################################################################

##########################################################################
###                         DEV COMMENTS
### > I'm not sure if I should do dash here or make another module for it
### > Added negation handler and score_calculator 23/7/2018
### > Need to integrate the lexicon score_calcualtor and negation handler
###   to main.py to either calculate every time or when conf < 1
###########################################################################

# Init Flask
app = Flask(__name__)

# Init Dash
dashApp = dash.Dash(__name__, server=app, url_base_pathname='/dashapp')

############################## Start Dash ####################################

dashApp.layout = html.Div(children=[
    html.H1(children='Sentiment Graph'),

    html.Div(children='''
        Dash: A web application framework
    '''),

    dcc.Graph(
        id='my-graph',
        figure={
            'data': [
                {'x': [1,2,3], 'y': [4,1,2], 'type': 'line', 'name': 'sf'},
                {'x': [1,2,3], 'y': [2,4,5], 'type': 'bar', 'name': 'karachi'},
            ],
            'layout': {
                'title': 'Sentiment Visualization Graph'
            }
        }
    )
])


@dashApp.callback(Output('my-graph', 'figure'), 
    # [Input('my-dropdown', 'value')]
    )
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

stop_words = set(stopwords.words("english"))

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
        try:
            input = request.form['keyword']
            quantity = int(request.form['quantity'])
            flash('Please enter some text to be analyzed', 'danger')
            if len(input) > 0 or quantity > 0 :
            # Fetch tweets from twitter and perform sentiment analysis on the given data
                pos_tweeets, neg_tweets = twitter_analyzer.analyzeTwitter(input, quantity)
            else:
                flash('Please enter some text to be analyzed', 'danger')
        except ValueError:
            flash('Please enter valid text to be analyzed', 'danger')
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

                # Check for adjectives
                allowed_word_types = ["J", "V"]
                pos = nltk.pos_tag(input)
                has_allowed_words = False
                all_words = []
                for w in pos:
                    if w[1][0] in stop_words:
                        continue
                    elif w[1][0] in allowed_word_types:
                        all_words.append(w[0].lower())
                        has_allowed_words = True

                if has_allowed_words:
                    # Classify with machine learning classifier
                    result = ml_classifiers.sentiment(input)


                    # Negation Handling
                    # Check for sentiment inversion
                    if result[0] == 'pos' and should_invert(input):
                        result = ('neg', result[1])
                    elif result[0] == 'neg' and should_invert(input):
                        result = ('pos', result[1])


                    # Sentiment score calculation
                    # Might need filtering before calculating the score
                    score = sentiment_score(input)
                    if score > 0 and should_invert(input):
                        score *= -1
                    elif score < 0 and should_invert(input):
                        score *= 1


                    if result[0] == 'pos':
                        flash('The sentiment of the text is Positive with the confidence of ' + str(result[1]*100) + "%" + ". Manual calculated score = " + str(score), 'info')
                    elif result[0] == 'neg':
                        flash('The sentiment of the text is Negative with the confidence of ' + str(result[1]*100) + "%" + ". Manual calculated score = " + str(score), 'info')
                else:
                    flash('The sentiment of the text is Neutral with the confidence of 100%. Manual calculated score = 0', 'info')


            except Exception as e:
                flash('A server side error has occured ' + "Exception: \"" + str(e) + "\"" , 'danger')


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


def preprocess_text(text):
    filtered_sentence = [w for w in text if not w in set(nltk.corpus.stopwords.words("english"))]
    return filtered_sentence


# Call main method/run app
if __name__ == '__main__':
    app.secret_key = 'secret123'
    app.run(debug=True, use_reloader=False)