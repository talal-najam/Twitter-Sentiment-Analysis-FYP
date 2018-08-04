import sys,tweepy,csv,re
from textblob import TextBlob
import matplotlib.pyplot as plt

"""
TODO: 
> Add the two parameters of keyword and number of tweets to search to the download data function.
  As we want to take the parameters directly as a request from the web page by the user.
> Somehow import the class into the main package and try working out with textblob to see
  if we can perform sentiment analysis on the given input from user
> Alternatively, somehow bring the input fields from that classes and call them here, perform the 
  data and then return the results there, but that doesn't seem right for some reason
> Need to display the result on the webpage rather than just plotting on IDE, should be done
  by somehow returning the result to the webpage and use chartJS or some tech to plot graph 
  dynamically on the webpage.

"""

class SentimentAnalysis:

    def __init__(self):
        self.tweets = []
        self.tweetText = []
        self.positiveTweets = []
        self.negativeTweets = []

    def analyzeTwitter(self, keyword, number):

        # auth
        consumerKey = "vPocvAU7dFdrGhbBxkt4Ab6tu"
        consumerSecret = "V2qoxlzAqtRs5NQ5ifJI3Tx0DsaCycUpBaXvKTSi1LCTWV2VG2"
        accessToken = "1139826878-qBQpLdy1E96PJrMLpg9IFbT11QuZDbV3dydHIHp"
        accessTokenSecret = "fu0igad1yHFFvP2rvNPLk2aOjTaAPaCOJaL2JXKVpn5T6"

        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth)

        # input for term to be searched and how many tweets to search
        # searchTerm = input("Enter Keyword/Tag to search about: ")
        # NoOfTerms = int(input("Enter how many tweets to search: "))

        # searching for tweets
        self.tweets = tweepy.Cursor(api.search, q=keyword, lang = "en").items(number)

        # Open/create a file to append data to
        # csvFile = open('result.csv', 'a')

        # Use csv writer
        # csvWriter = csv.writer(csvFile)


        # creating some variables to store info
        polarity = 0
        positive = 0
        wpositive = 0
        spositive = 0
        negative = 0
        wnegative = 0
        snegative = 0
        neutral = 0


        # iterating through tweets fetched
        for tweet in self.tweets:
            #Append to temp so that we can store in csv later. I use encode UTF-8
            self.tweetText.append(self.cleanTweet(tweet.text).encode('utf-8'))
            # print (tweet.text.translate(non_bmp_map))    #print tweet's text
            analysis = TextBlob(tweet.text)
            # print(analysis.sentiment)  # print tweet's polarity
            polarity += analysis.sentiment.polarity  # adding up polarities to find the average later

            if analysis.sentiment.polarity >= 0:
                self.positiveTweets.append(tweet.text)
            else:
                self.negativeTweets.append(tweet.text)

            if (analysis.sentiment.polarity == 0):  # adding reaction of how people are reacting to find average later
                neutral += 1
            elif (analysis.sentiment.polarity > 0 and analysis.sentiment.polarity <= 0.3):
                wpositive += 1
            elif (analysis.sentiment.polarity > 0.3 and analysis.sentiment.polarity <= 0.6):
                positive += 1
            elif (analysis.sentiment.polarity > 0.6 and analysis.sentiment.polarity <= 1):
                spositive += 1
            elif (analysis.sentiment.polarity > -0.3 and analysis.sentiment.polarity <= 0):
                wnegative += 1
            elif (analysis.sentiment.polarity > -0.6 and analysis.sentiment.polarity <= -0.3):
                negative += 1
            elif (analysis.sentiment.polarity > -1 and analysis.sentiment.polarity <= -0.6):
                snegative += 1


        # Write to csv and close csv file
        # csvWriter.writerow(self.tweetText)
        # csvFile.close()

        # finding average of how people are reacting
        positive = self.percentage(positive, number)
        wpositive = self.percentage(wpositive, number)
        spositive = self.percentage(spositive, number)
        negative = self.percentage(negative, number)
        wnegative = self.percentage(wnegative, number)
        snegative = self.percentage(snegative, number)
        neutral = self.percentage(neutral, number)

        # finding average reaction
        polarity = polarity / number

        # # printing out data
        # print("How people are reacting on " + keyword + " by analyzing " + str(number) + " tweets.")
        # print()
        # print("General Report: ")

        # if (polarity == 0):
        #     print("Neutral")
        # elif (polarity > 0 and polarity <= 0.3):
        #     print("Weakly Positive")
        # elif (polarity > 0.3 and polarity <= 0.6):
        #     print("Positive")
        # elif (polarity > 0.6 and polarity <= 1):
        #     print("Strongly Positive")
        # elif (polarity > -0.3 and polarity <= 0):
        #     print("Weakly Negative")
        # elif (polarity > -0.6 and polarity <= -0.3):
        #     print("Negative")
        # elif (polarity > -1 and polarity <= -0.6):
        #     print("Strongly Negative")

        # print()
        # print("Detailed Report: ")
        # print(str(positive) + "% people thought it was positive")
        # print(str(wpositive) + "% people thought it was weakly positive")
        # print(str(spositive) + "% people thought it was strongly positive")
        # print(str(negative) + "% people thought it was negative")
        # print(str(wnegative) + "% people thought it was weakly negative")
        # print(str(snegative) + "% people thought it was strongly negative")
        # print(str(neutral) + "% people thought it was neutral")

        self.plotPieChart(positive, wpositive, spositive, negative, wnegative, snegative, neutral, keyword, number)

        return self.positiveTweets, self.negativeTweets


    def cleanTweet(self, tweet):
        # Remove Links, Special Characters etc from tweet
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())

    # function to calculate percentage
    def percentage(self, part, whole):
        temp = 100 * float(part) / float(whole)
        return format(temp, '.2f')

    def plotPieChart(self, positive, wpositive, spositive, negative, wnegative, snegative, neutral, keyword, number):
        labels = ['Positive [' + str(positive) + '%]', 'Weakly Positive [' + str(wpositive) + '%]','Strongly Positive [' + str(spositive) + '%]', 'Neutral [' + str(neutral) + '%]',
                  'Negative [' + str(negative) + '%]', 'Weakly Negative [' + str(wnegative) + '%]', 'Strongly Negative [' + str(snegative) + '%]']
        sizes = [positive, wpositive, spositive, neutral, negative, wnegative, snegative]
        colors = ['yellowgreen','lightgreen','darkgreen', 'gold', 'red','lightsalmon','darkred']
        patches, texts = plt.pie(sizes, colors=colors, startangle=90)
        plt.legend(patches, labels, loc="best")
        plt.title('How people are reacting on ' + keyword + ' by analyzing ' + str(number) + ' Tweets.')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()



if __name__== "__main__":
    print("Running script")