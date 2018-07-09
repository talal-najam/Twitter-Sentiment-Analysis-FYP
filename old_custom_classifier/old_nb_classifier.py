from nltk import word_tokenize, sent_tokenize
import math
import random
from collections import Counter, defaultdict

# 10664 total reviews
# 5332 positive reviews

#########################################################################################
# Naive Bayes Rule  		Pr[X|Y] = Pr[Y|X] * Pr[X] 
#									  ---------------
#							  	   		   Pr[Y]
#########################################################################################

#########################################################################################
# The core concept of ML is you make a prediction based on a given set of observations
# So the equation above can be rewritten as follows
#########################################################################################

#########################################################################################
# Naive Bayes Rule  		Pr[class|observations] = Pr[observations|class] * Pr[class] 
#									 				 ----------------------------------
#							  	   		  					  Pr[observations]
# 						Prediction = Posterior       Likelihood                Priors
#########################################################################################

# Open dataset with positive labelled reviews
positive_reviews = open("../short_reviews/positive.txt","r").read()

# Open dataset with negative labelled reviews
negative_reviews = open("../short_reviews/negative.txt","r").read()

# Create an array of tuples containing reviews and appending all shuffeled reviews
all_reviews = []

for review in positive_reviews.split('\n')[:3000]:
	all_reviews.append((review, 'pos'))

for review in negative_reviews.split('\n')[:3000]:
	all_reviews.append((review, 'neg'))

random.shuffle(all_reviews)


# Creating training and testing set
training_set = []
testing_set = []

for review in all_reviews[:5000]:
	training_set.append(review)

for review in all_reviews[5000:]:
	testing_set.append(review)

# Create training and testing set with a tab seperated values (.tsv) file
def create_training_set(filename):
	with open(filename,"w") as f:
		for feature in training_set:
			f.write(feature[0] + "\t" + feature[1] + "\n")
		f.close()

def create_testing_set(filename):
	with open(filename,"w") as f:
		for feature in testing_set:
			f.write(feature[0] + "\t" + feature[1] + "\n")
		f.close()

def tokenize(words):
	"""Break up words into text"""
	"""Not using regular expressions"""
	result = word_tokenize(words)
	return result

def read_training_file(filename):
	priors = Counter()
	likelihood = defaultdict(Counter)

	with open(filename) as f:
		for line in f:
			parts = line.strip().split('\t')
			priors[parts[1]] += 1
			for word in tokenize(parts[0]):
				likelihood[parts[1]][word] += 1
	return (priors, likelihood)

def read_testing_file(filename):
	return [line.strip().split('\t') for line in open(filename).readlines()]

def classify_random(line, priors, likelihood):
	""" Return a random category """
	categories = list(priors.keys())
	return categories[int(random.random() * len(categories))]

def classify_max_prior(line, priors, likelihood):
	""" Return the biggest category """
	return max(priors, key=lambda x: priors[x])

def classify_bayesian(line, priors, likelihood):
	""" Returns the class that maximizes the posterior """
	# priors = category
	# likelihood = words
	max_class = (-1E6, '')
	for c in priors.keys():
		p = math.log(priors[c])
		n = float(sum(likelihood[c].values())) 
		for word in tokenize(line[0]):
			p = p + math.log(max(1E-6, likelihood[c][word] / n)) 

		if p > max_class[0]:
			max_class = (p,c)
	print(max_class)
	return max_class[1]

# defining main method
def main(training_file, testing_file):
	create_training_set(training_file)
	create_testing_set(testing_file)

	(priors, likelihood) = read_training_file(training_file)
	lines = read_testing_file(testing_file)

	num_correct = 0
	for line in lines:
		if classify_bayesian(line, priors, likelihood) == line[1]:
			num_correct += 1 

	# Writing likelihood (every word in categories, to a log.txt file)
	# f = open("log.txt", "w")
	# # f.write(str(likelihood))
	# f.close()


	print("Classified %d correctly out of %d for an accuracy of %f" %(num_correct, len(lines), float(num_correct)/len(lines)))
	print(priors)


if __name__ == '__main__':
	main('train.tsv','test.tsv')

