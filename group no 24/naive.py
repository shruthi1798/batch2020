from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import six.moves.cPickle as pickle
from prodReview import extract_words
from nltk.corpus import stopwords

# Load All Reviews in train and test datasets
f = open('train.pkl', 'rb')
reviews = pickle.load(f)
f.close()

f = open('test.pkl', 'rb')
test = pickle.load(f)
f.close()


# Generate counts from text using a vectorizer.  
# There are other vectorizers available, and lots of options you can set.
# This performs our step of computing word counts.
vectorizer = CountVectorizer()
train_features = vectorizer.fit_transform([r for r in reviews[0]])
test_features = vectorizer.transform([r for r in test[0]])

# Fit a naive bayes model to the training data.
# This will train the model using the word counts we computer, 
#       and the existing classifications in the training set.
nb = MultinomialNB()
nb.fit(train_features, [int(r) for r in reviews[1]])

# Now we can use the model to predict classifications for our test features.
predictions = nb.predict(test_features)

# Compute the error.  
def accur(num1):
    print(metrics.classification_report(test[1], predictions))
    print("accuracy: {0}".format(metrics.accuracy_score(test[1], predictions)))
    sentence = num1

    sentences = []
    sentences.append(sentence)
    input_features = vectorizer.transform(extract_words(sentences))
    prediction = nb.predict(input_features)

def naive(sentence):
    sentences = []
    sentences.append(sentence)
    input_features = vectorizer.transform(extract_words(sentences))
    prediction = nb.predict(input_features)
    return prediction[0]

