import os
from xml.etree import cElementTree
from nltk.tokenize import word_tokenize, sent_tokenize
# from sentiment_shifter import should_invert

file_name = 'en-sentiment.xml'
full_file = os.path.abspath(os.path.join('data', file_name))

xml = cElementTree.parse(full_file)
xml = xml.getroot()

def sentiment_score(tweet):
    words = word_tokenize(tweet)

    print(words)

    acum_score = 0

    for word in words:
        for w in xml.findall('word'):
            w, pos, p= (w.attrib.get("form"), w.attrib.get("pos"), w.attrib.get("polarity", 0.0))
            if word == str(w):
                acum_score += float(p)

    # Calculate accumulated score
    if acum_score > 1:
        acum_score = 1
    elif acum_score < -1:
        acum_score = -1

    return acum_score