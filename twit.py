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


# #################################################################################################
# Author: Talal Najam
# Date  : 21/12/2018
# Github: https://github.com/mistat44
# 
# 
# Steps needed to carry out the project
# 1. Add the two parameters of keyword and number of tweets to search to the download data function. As we want to take the parameters directly as a request from the web page by the user.
# 2. Somehow import the class into the main package and try working out with textblob to see if we can perform sentiment analysis on the given input from user
# 3. Alternatively, somehow bring the input fields from that classes and call them here, perform the data and then return the results there, but that doesn't seem right for some reason
# 4. Need to display the result on the webpage rather than just plotting on IDE, should be done by somehow returning the result to the webpage and use chartJS or some tech to plot graph dynamically on the webpage.
# #################################################################################################

# Create a set for stop words
stop_words = set(stopwords.words("english"))

def analyzeTwitter(keyword, number):

    # auth
    consumerKey = "vPocvAU7dFdrGhbBxkt4Ab6tu"
    consumerSecret = "V2qoxlzAqtRs5NQ5ifJI3Tx0DsaCycUpBaXvKTSi1LCTWV2VG2"
    accessToken = "1139826878-qBQpLdy1E96PJrMLpg9IFbT11QuZDbV3dydHIHp"
    accessTokenSecret = "fu0igad1yHFFvP2rvNPLk2aOjTaAPaCOJaL2JXKVpn5T6"

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

# # # # # # USED GOOGLE CHARTS INSTEAD OF MATPLOTLIB # # # # # #
# 
#  
# def plotPieChart(positive, negative, neutral, keyword, number):
#     labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]','Negative [' + str(negative) + '%]']
#     sizes = [positive, neutral, negative]
#     colors = ['darkgreen', 'gold', 'red']
#     patches, texts = plt.pie(sizes, colors=colors, startangle=90)
#     plt.legend(patches, labels, loc="best")
#     plt.title('How people are reacting on "' + keyword.upper() + '" by analyzing ' + str(number) + ' Tweets.')
#     plt.axis('equal')
#     plt.tight_layout()
#     plt.show()

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
    
