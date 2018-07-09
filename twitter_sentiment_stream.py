from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from unidecode import unidecode

import sqlite3

import sentiment_module as ml_classifiers
from textblob import TextBlob as tb

import time


# DB Connection
conn = sqlite3.connect('twitter.db')
c = conn.cursor()


def create_table():
	c.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, tweet TEXT, sentiment REAL)")
	conn.commit()


create_table()


# Tweepy
# Authorization
ckey="vPocvAU7dFdrGhbBxkt4Ab6tu"
csecret="V2qoxlzAqtRs5NQ5ifJI3Tx0DsaCycUpBaXvKTSi1LCTWV2VG2"
atoken="1139826878-qBQpLdy1E96PJrMLpg9IFbT11QuZDbV3dydHIHp"
asecret="fu0igad1yHFFvP2rvNPLk2aOjTaAPaCOJaL2JXKVpn5T6"


class listener(StreamListener):
	def on_data(self, data):
		try:
			data = json.loads(data)
			tweet = unidecode(data['text'])
			time_ms = data['timestamp_ms']
			# sentiment = cohesive_classify(tweet)
			sentiment = tb(tweet).sentiment.polarity
			print(time_ms, tweet, sentiment)
			c.execute("INSERT INTO sentiment(unix, tweet, sentiment) VALUES (?, ?, ?)", (time_ms, tweet, sentiment))
			conn.commit()

		except KeyError as e:
			print(str(e))				
		return(True)

	def on_error(self, status):
		print(status)


# Sentiment Classification
def cohesive_classify(text):
    ml_result = ml_classifiers.sentiment(text)[0]
    tb_polarity = tb(text).sentiment.polarity

    if tb_polarity > 0:
        tb_result = "pos"
    elif tb_polarity < 0:
        tb_result = "neg"
    else:
        tb_result = "neutral"
    # customer_result = customer_classifier.classify(text)
    if tb_result and ml_result == 'pos':
        result = "pos"
    elif tb_result and ml_result == 'neg':
        result = "neg"
    else:
        result = "neutral"
    return result


# Infinite Loop to counter api break
while True:
	try:
		auth = OAuthHandler(ckey, csecret)
		auth.set_access_token(atoken, asecret)
		twitterStream = Stream(auth, listener())
		twitterStream.filter(track=["a", "e", "i", "o", "u"])
	except Exception as e:
		print(str(e))
		time.sleep(5)