#imports
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np

#data
fakes = open("/Users/adatur/Downloads/McGill/W2024/COMP550 - NLP/A1/fakes.txt")
facts = open("/Users/adatur/Downloads/McGill/W2024/COMP550 - NLP/A1/facts.txt")
fake_lines = fakes.readlines()
fact_lines = facts.readlines()

#preprocessing funcs
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
vectorizer = CountVectorizer(stop_words='english') #can add stop_words='english' if we want to remove stopwords

def lemmatize(text):
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = " ".join(lemmatized_words)
    return lemmatized_text

#preprocessing
lemmatized = []
stemmed = []
Y_vec = []
total = []


for fact_line in fact_lines:
    Y_vec.append(1)

    #stemming
    #stemm = [stemmer.stem(word) for word in fact_line.split()]
    #stemmed.append(' '.join(stemm))

    #lemmatizing
    #lemmatized.append(lemmatize(fact_line))

for fake_line in fake_lines:
    Y_vec.append(-1)

    #stemming
    #stemm = [stemmer.stem(word) for word in fake_line.split()]
    #stemmed.append(' '.join(stemm))

    # lemmatizing
    #lemmatized.append(lemmatize(fake_line))

total = fact_lines + fake_lines

X = vectorizer.fit_transform(total)

#training classifier
X_tmp, X_test, y_tmp, y_test = train_test_split(X, Y_vec, test_size=0.2, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=42)

classifier = MultinomialNB(alpha=0.4)
#classifier = LogisticRegression(penalty='l2', C=0.9, tol=0.001)
#classifier = svm.SVC(C=1.0, kernel='linear', shrinking=False)
classifier.fit(X_train, y_train)

#evaluation on dev set
"""
y_pred = classifier.predict(X_dev)
accuracy = accuracy_score(y_dev, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_dev, y_pred))
"""

#testing classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
