# #################################################################################################
# Author: Talal Najam
# Date  : 21/12/2018
# Github: https://github.com/mistat44
# #################################################################################################

import sys,tweepy,csv,re
from textblob import TextBlob
import matplotlib.pyplot as plt
import sentiment_module as ml_classifiers
from sentiment_shifter import should_invert
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from score_calculator import sentiment_score
from config import c_key, c_secret, a_token, a_token_secret
import os
 
# Create a set for stop words
stop_words = set(stopwords.words("english"))

def analyzeTwitter(keyword, number):

    # auth
    consumerKey = os.env.get("consumerKey")
    consumerSecret = os.env.get("consumerSecret")
    accessToken = os.env.get("accessToken")
    accessTokenSecret = os.env.get("accessTokenSecret")

    tweets = []
    positiveTweets = []
    negativeTweets = []
    total_tweets = []
    total_tweets_analyzed = []
    chart_classifier_results = []

    chartdict = {}

    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)

    # searching for tweets
    tweets = tweepy.Cursor(api.search, q=keyword, lang = "en").items(number)

    # creating some variables to store info
    polarity = 0
    positive = 0
    negative = 0
    neutral = 0
    result = 0


    # iterating through tweets fetched
    for tweet in tweets:

        #################CLASSIFICATION#################
        # Clean the tweet text
        text_tweet = tweet.text

        # Analyze with custom voted classifier
        analysis = manual_rules(text_tweet.lower())

        # Sample : (('pos', 1), 1)
        if analysis[0][0] == 'pos' and analysis[1] >= 0:
            result = 1 
            positive += 1
        elif analysis[0][0] == 'neg' and analysis[1] < 0:
            result = -1
            negative += 1
        else:
            result = 0
            neutral += 1 

        print("TWEET : " + text_tweet + " | ANALYSIS | " + str(analysis) + " | FINAL RESULT| " + str(result) + "\n")

        ################################################

        total_tweets.append((text_tweet, result))
        total_tweets_analyzed.append(analysis[1])
        chart_classifier_results.append(result)


        # print("Tweet #" + str(counter) + " : " + str(clean_tweet)[2:])


    chartdict['x'] = [i for i, tweet in enumerate(total_tweets, 1)]
    # Manually calculated scores
    chartdict['y1'] = total_tweets_analyzed
    # Classifier result
    chartdict['y2'] = chart_classifier_results


    # finding average of how people are reacting
    positive_percentage = percentage(positive, number)
    negative_percentage = percentage(negative, number)
    neutral_percentage = percentage(neutral, number)

    # finding average reaction
    polarity = polarity / number

    return positiveTweets, negativeTweets, total_tweets, chartdict, positive_percentage, negative_percentage, neutral_percentage


# function to calculate percentage
def percentage(part, whole):
    temp = 100 * float(part) / float(whole)
    return format(temp, '.2f')

def manual_rules(input):
    allowed_word_types = ["J", "V"]
    tokens = word_tokenize(input)
    pos = nltk.pos_tag(tokens)
    has_allowed_words = False
    all_words = []
    for w in pos:
        if w[1][0] in stop_words:
            continue
        elif w[1][0] not in allowed_word_types:
            continue
        else:
            all_words.append(w[0].lower())
            has_allowed_words = True

    filtered_input = ' '.join(str(x) for x in all_words)

    print("ALL WORDS",all_words)
    print("FILTERED INPUT",filtered_input)

    if has_allowed_words:
        # Classify with machine learning classifier
        result = ml_classifiers.sentiment(filtered_input)

        # Negation Handling
        # Check for sentiment inversion
        if result[0] == 'pos' and should_invert(input):
            result = ('neg', result[1])
        elif result[0] == 'neg' and should_invert(input):
            result = ('pos', result[1])

        # Sentiment score calculation
        # Might need filtering before calculating the score
        score = sentiment_score(input)

        # Check for polarity flips
        if should_invert(input):
            score *= -1

        # Assign result values
        if result[0] == 'pos':
            return (result, score)
        elif result[0] == 'neg':
            return (result, score)
    else:
        return (('neutral', 100), 0)



if __name__== "__main__":
    print("Running script")
    analyzeTwitter("westworld", 3)
    
