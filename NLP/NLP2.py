import random
import nltk
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import pickle

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for classifier in self._classifiers:
			vote = classifier.classify(features)
			votes.append(vote)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for classifier in self._classifiers:
			vote = classifier.classify(features)
			votes.append(vote)

		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf


# List of tuples that contains list of words of reviews with the category - pos/neg
documents = [(list(movie_reviews.words(fileid)), category)
			 for category in movie_reviews.categories()
			 for fileid in movie_reviews.fileids(category)]

# Shuffling the documents since they are sorted by default
# random.shuffle(documents)

# Bag of words model
all_words = [] 
for word in movie_reviews.words():
	word = word.lower()
	all_words.append(word)

# Performing frequency distribution of the all_words list
all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))

# preparing the word features to feed the classifier
word_features = []
for i in all_words.most_common(3000):
    word_features.append(i[0])

# find feature function that takes a document as an argument
def find_features(document):
	words = set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	return features

# Creating featuresets
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# your gonna have a labelled dataset
# positive data example:
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# Naive bayes algorithm - good because its scalable and basic
# posterior = prior occurences * likelihood / evidence

# classifier = nltk.NaiveBayesClassifier.train(training_set)

# Loading a saved classifier object into a variable file by using the pickle module
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Original Naive bayes algo accuracy: ", (nltk.classify.accuracy(classifier, testing_set))* 100)
classifier.show_most_informative_features(15)

# Testing out all the different algos from the sklearn module

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier, testing_set))* 100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BNB_classifier accuracy percent: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) *100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) *100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) *100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) *100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) *100)



voted_classifier = VoteClassifier(classifier, MNB_classifier, BernoulliNB_classifier, 
							      LogisticRegression_classifier, 
							      LinearSVC_classifier, NuSVC_classifier)
# print("Voted classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set)) *100)
# print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0]))
# print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:", voted_classifier.confidence(testing_set[1][0]))
# print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:", voted_classifier.confidence(testing_set[2][0]))
# print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:", voted_classifier.confidence(testing_set[3][0]))
# print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:", voted_classifier.confidence(testing_set[4][0]))
# print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:", voted_classifier.confidence(testing_set[5][0]))
