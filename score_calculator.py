import os
from xml.etree import cElementTree
from nltk.tokenize import word_tokenize, sent_tokenize
# from sentiment_shifter import should_invert

file_name = 'en-sentiment.xml'
full_file = os.path.abspath(os.path.join('data', file_name))

xml = cElementTree.parse(full_file)
xml = xml.getroot()

all_words = []

# short_pos = open("short_reviews/positive.txt", "r").read()

# tweets = [tweet for tweet in short_pos.split('\n')[:10]]

# TODO:- Calculate Relative Proportional difference, bounds: [-1, 1]

def sentiment_score(tweet):
    words = word_tokenize(tweet)
    for word in words:
        all_words.append(word)
    
    acum_score = 0
    positive_score = 0
    negative_score = 0

    # for word in all_words:
    #     for w in xml.findall('word'):
    #         w, pos, p= (w.attrib.get("form"), w.attrib.get("pos"), w.attrib.get("polarity", 0.0))
    #         if word == str(w) and float(p) >= 0:
    #             positive_score += float(p)
    #         elif word == str(w) and float(p) < 0:
    #             negative_score += float(p)
    
    for word in all_words:
        for w in xml.findall('word'):
            w, pos, p= (w.attrib.get("form"), w.attrib.get("pos"), w.attrib.get("polarity", 0.0))
            if word == str(w) and float(p) >= 0:
                acum_score += float(p)
            elif word == str(w) and float(p) < 0:
                acum_score += float(p)
    
    # TODO:- Calculate Relative Proportional difference, bounds: [-1, 1]
    # print(positive_score)
    # print(negative_score)
    # if positive_score != 0 and negative_score != 0:
    # acum_score = (positive_score - negative_score) / (positive_score + negative_score)
    # else:
        # acum_score = 0

    # print(acum_score)

    # Calculate accumulated score
    if acum_score > 1:
        acum_score = 1
    elif acum_score < -1:
        acum_score = -1

    return acum_score

    # if should_invert(tweet):
    #     return acum_score * -1
    # else:
    #     return acum_score


# print(sentiment_score("I did not disliked this movie"))