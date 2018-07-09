## @Talal Najam
## Sentdex NLTK Series p.19
## Sentiment Analysis module

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode, StatisticsError

from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        try:
            return mode(votes)
        except Exception as e:
            return ("CAUGHT AN EXCEPTION " + str(e))


    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
        
short_pos = open("short_reviews/positive.txt","r").read()
short_neg = open("short_reviews/negative.txt","r").read()

documents_f = open("LIMITED_PICKLES/documents_POS.pickle","rb")
documents = pickle.load(documents_f)
documents_f.close()


all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)        

word_features_f = open("LIMITED_PICKLES/word_features5k.pickle","rb")
word_features = pickle.load(word_features_f)
word_features_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets_f = open("LIMITED_PICKLES/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)
# print(len(featuresets))

# positive data example:      
training_set = featuresets[:3000]
testing_set =  featuresets[3000:4000]

open_file = open("LIMITED_PICKLES/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open("LIMITED_PICKLES/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("LIMITED_PICKLES/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("LIMITED_PICKLES/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("LIMITED_PICKLES/SGDClassifier_classifier5k.pickle", "rb")
SGDClassifier_classifier = pickle.load(open_file)
open_file.close()

open_file = open("LIMITED_PICKLES/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

# open_file = open("sentdex_pickled_algos/NuSVC_classifier5k.pickle", "rb")
# NuSVC_classifier = pickle.load(open_file)
# open_file.close()

voted_classifier = VoteClassifier(
                                #   NuSVC_classifier,
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)
