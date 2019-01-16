# #################################################################################################
# Author: Talal Najam
# Date  : 21/12/2018
# Github: https://github.com/mistat44
# #################################################################################################

from flask import Flask, render_template, flash, redirect, url_for, session, request, logging, jsonify
import twit as SentimentAnalysis
from sentiment_shifter import should_invert
import sentiment_module as ml_classifiers
from textblob import TextBlob as tb
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from score_calculator import sentiment_score

import requests

# Steps:- 
# 1. Take input from a user in terms of keywords [D]
# 2. Take the amount of tweets needed to be analyzed (default = 5) [D]
# 3. Stream the related tweets and the same amount of tweets
# 4. Process the tweet with some sort of api or method
# 5. Plot the results on the graph
# 6. Return the result effectively and display them
# 8. Use the preProcessText function 


# Init Flask
app = Flask(__name__)

stop_words = set(stopwords.words("english"))


# #########################################################
# PUBLIC Route
# NAME Index
# DESC Fetch Tweets and Analyze them and plot pie chart
# #########################################################
@app.route('/')
@app.route('/home')
def index():
    return render_template('home.html', title="Sentiment Analysis")


# #########################################################
# PUBLIC Route
# NAME Twitter Sentiment
# DESC Fetch Tweets and Analyze them and visualize data
# #########################################################
@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    # Handling POST request 
    if request.method == 'POST':
        try:
            # Get Form Fields
            keyword = request.form['keyword']
            quantity = int(request.form['quantity'])
            if len(keyword) > 0 or quantity > 0 :

                # Fetch tweets from twitter and perform sentiment analysis on the given data
                pos_tweeets, neg_tweets, total_tweets, chartdict,  positive_percentage, negative_percentage, neutral_percentage = SentimentAnalysis.analyzeTwitter(keyword, quantity)

                return render_template('test2.html', total_tweets=total_tweets,
                                        keyword=keyword.upper(), 
                                        number=quantity, 
                                        positive_percentage=positive_percentage,
                                        negative_percentage=negative_percentage,
                                        neutral_percentage=neutral_percentage)
           
        except ValueError as e:  
            flash('Please enter valid text to be analyzed', 'danger')
    
    # Default form handling GET request
    return render_template('sentiment.html', title="Twitter Sentiment")


# #########################################################
# PUBLIC Route
# NAME Manual Sentiment
# DESC Analyze the sentiment of manual text input
# #########################################################
@app.route('/manualsentiment', methods=['GET', 'POST'])
def manualSentiment():
    if request.method == 'POST':
        
        # Get Form Fields
        input = request.form['keyword']
        
        # Basic validation
        if len(input) > 0:
            # Extract verbs and adjectives
            try:
                allowed_word_types = ["J", "V"]
                tokens = word_tokenize(input)
                pos = nltk.pos_tag(tokens)
                has_allowed_words = False
                all_words = []

                # Filter out stop-words (e.g. the, on, at)
                for w in pos:
                    if w[1][0] in stop_words:
                        continue
                    elif w[1][0] in allowed_word_types:
                        all_words.append(w[0].lower())
                        has_allowed_words = True

                # Combine the filtered sentence
                filtered_input = ' '.join(str(x) for x in all_words)

                print(all_words)
                print(filtered_input)

                # Check if sentence contains sentiment words
                if has_allowed_words:

                    # Classify with machine learning classifier
                    result = ml_classifiers.sentiment(filtered_input)

                    # Negation Handling & Check for sentiment inversion
                    if result[0] == 'pos' and should_invert(input):
                        result = ('neg', result[1])
                    elif result[0] == 'neg' and should_invert(input):
                        result = ('pos', result[1])

                    # Sentiment score calculation
                    score = sentiment_score(input)

                    # Check for polarity flips
                    if should_invert(input):
                        score *= -1
                    
                    # Form a list of top notable features
                    j = 1
                    showable_feature_list = []

                    for word in all_words:
                        if word in showable_feature_list:
                            j+=1
                            continue
                        else:
                            number = str(j) + ". " + word
                            showable_feature_list.append(number)
                            j+=1

                    # Generate list of features
                    important_features = ', '.join(str(x) for x in showable_feature_list)

                    # Display result
                    if result[0] == 'pos':
                        flash('The sentiment of the text is Positive with the confidence of ' + str(round(result[1]*100, 2)) + "%" + ". Manual calculated score = " + str(round(score, 2)), 'success')
                        flash("Top features:\t" + important_features, 'primary')
                    elif result[0] == 'neg':
                        flash('The sentiment of the text is Negative with the confidence of ' + str(round(result[1]*100, 2)) + "%" + ". Manual calculated score = " + str(round(score, 2)), 'danger')
                        flash("Top features:\t" + important_features, 'primary')
                else:
                    flash('The sentiment of the text is Neutral with the confidence of 100%. Manual calculated score = 0', 'warning')
                    flash('Make sure to use words which express sentiment, e.g. adjectives and verbs to make the best use of this system', 'warning')


            except Exception as e:
                flash('A server side error has occured ' + "Exception: \"" + str(e) + "\"" , 'warning')

            return redirect(url_for('manualSentiment', title="Manual Sentiment"))
        else:
            flash('Please enter some text to be analyzed', 'warning')

    return render_template('manualsentiment.html', title="Manual Sentiment")


@app.route('/about')
def about():
    return render_template('about.html', title="About This Project")


def cohesive_classify(text):

    ml_result = ml_classifiers.sentiment(text)[0]
    tb_polarity = tb(text).sentiment.polarity

    if tb_polarity > 0:
        tb_result = "pos"
    elif tb_polarity < 0:
        tb_result = "neg"
    else:
        tb_result = "ambiguous"

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

# Run app
if __name__ == '__main__':
    app.secret_key = 'secret123'
    app.run(debug=True, use_reloader=False)