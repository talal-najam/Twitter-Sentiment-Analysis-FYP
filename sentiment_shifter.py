from nltk.tokenize import word_tokenize
# short_pos = open("short_reviews/positive.txt", "r").read()

# TODO:-
# 1. Take a review
# 2. split the words
# 3. handle negation

tweet = "I didn't like this movie"


def handle_negation(review):
    # Check if words contain "n't", "not"
    # Add string "NOT_" to the following words in the sentence
    # Check if mod count(NOT_) 2 == 0? pos : neg
    # review = "I did not like this movie at all"

    negations = ["n't", "not", "couldnt", "shouldnt", "wont"]
    punctuations = [".", ","]
    all_words = []

    review = review.lower()
    neg_word_count = 0
    negative = False
    words = word_tokenize(review)

    for word in words:
        if word in negations:
            neg_word_count += 1

    # TODO aggregate the NOT_s in the sentence, if mod % 2 == 0, it is positive else negative
    if neg_word_count % 2 == 0:
        return words
    else:
        for word in words:
            if negative == False and word in negations:
                negative = True
                all_words.append(word)
                continue

            if negative and word not in punctuations:
                word = "NOT_" + word
                all_words.append(word)
            else:
                negative = False
                all_words.append(word)

        return all_words

# print(handle_negation("Hi, I'm talal. I don't like you and any of this movies."))

def should_invert(text):
    # Check if negations mod 2 == 0? pos : neg
    # text = "I did not not like this movie at all"

    negations = ["n't", "not", "couldnt", "shouldnt", "wont"]
    punctuations = [".", ","]
    all_words = []

    text = text.lower()
    neg_word_count = 0
    negative = False
    words = word_tokenize(text)

    for word in words:
        if word in negations:
            neg_word_count += 1
        if word == ".":
            neg_word_count = 0

    if neg_word_count % 2 == 0:
        return False
    else:
        return True

