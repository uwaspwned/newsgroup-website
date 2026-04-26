import pickle

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

newsgroups = fetch_20newsgroups()

X, y = newsgroups.data, newsgroups.target # type: ignore

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('model', MultinomialNB())
])

pipeline.fit(X_train, y_train)

with open('model.pkl', 'wb') as file:
    pickle.dump(pipeline, file)