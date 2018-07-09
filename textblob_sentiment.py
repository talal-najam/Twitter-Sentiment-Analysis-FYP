# pip install textblob vadersentiment
# from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import time
analyzer = SentimentIntensityAnalyzer()


# Positive data set
pos_count = 0
pos_correct = 0

with open("positive.txt","r") as f:
    for line in f.read().split('\n'):
        vs = analyzer.polarity_scores(line)
        if not vs['neg'] > 0.1:
            if vs['pos']-vs['neg'] > 0:
                pos_correct += 1
            pos_count += 1        

# Negative data set
neg_count = 0
neg_correct = 0

with open("negative.txt","r") as f:
    for line in f.read().split('\n'):
        vs = analyzer.polarity_scores(line)
        if not vs['pos'] > 0.1:
            if vs['pos']-vs['neg'] <= 0:
                neg_correct += 1
            neg_count += 1        


print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100, neg_count))



# threshold = 0.001
# pos_count = 0
# pos_correct = 0

# with open("positive.txt","r") as f:
#     for line in f.read().split('\n'):
#         analysis = TextBlob(line)
        
#         if analysis.sentiment.polarity >= threshold:
#             if analysis.sentiment.polarity > 0:
#                 pos_correct += 1
#             pos_count += 1

# neg_count = 0
# neg_correct = 0

# with open("negative.txt","r") as f:
#     for line in f.read().split('\n'):
#         analysis = TextBlob(line)
        
#         if analysis.sentiment.polarity <= -threshold:
#             if analysis.sentiment.polarity <= 0:
#                 neg_correct += 1
#             neg_count += 1

# print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100, pos_count))
# print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100, neg_count))
