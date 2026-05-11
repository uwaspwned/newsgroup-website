import json
import pickle

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from app.config import Config

MODEL_PATH = Config.MODEL_PATH
MODEL_CONFIG_PATH = Config.MODEL_CONFIG_PATH

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

newsgroups = fetch_20newsgroups()

X, y = newsgroups.data, newsgroups.target  # type: ignore

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("model", MultinomialNB()),
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

model_configuration = {
    "model": {
        "type": "MultinomialNB",
        "vectorizer": "TfidfVectorizer",
        "dataset": "20newsgroups",
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True,
    },
    "artifacts": {
        "model_path": str(MODEL_PATH),
    },
    "metrics": {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=newsgroups.target_names, # type: ignore
            output_dict=True,
            zero_division=0,
        ),
    },
    "classes": {
        str(index): name
        for index, name in enumerate(newsgroups.target_names) # type: ignore
    },
}

with open(MODEL_PATH, "wb") as file:
    pickle.dump(pipeline, file)

with open(MODEL_CONFIG_PATH, "w", encoding="utf-8") as file:
    json.dump(model_configuration, file, indent=2)