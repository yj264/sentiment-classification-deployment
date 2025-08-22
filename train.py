import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os, shutil
import pandas as pd
from sklearn.datasets import load_files

# load the IMDB dataset
dataset = load_files('aclImdb/train', categories=['pos', 'neg'], shuffle=True, random_state=42)
texts, labels = dataset.data, dataset.target
texts = [t.decode('utf-8', errors='ignore') for t in texts]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

pipeline = make_pipeline(TfidfVectorizer(max_features=5000), LogisticRegression(max_iter=1000))

with mlflow.start_run() as run:
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(pipeline, "model")

    print(f"Logged model with accuracy: {acc}, run_id: {run.info.run_id}")

    # save the model to a specific path
    export_path = "mlruns/latest_model"
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
    mlflow.sklearn.save_model(pipeline, export_path)

    # save word distribution for drift detection
    vectorizer = pipeline.named_steps['tfidfvectorizer']
    vocab = vectorizer.get_feature_names_out()
    word_freq = vectorizer.transform(X_train).sum(axis=0).A1
    drift_data = pd.DataFrame({"word": vocab, "freq": word_freq})
    drift_data.to_csv("mlruns/latest_model/train_word_dist.csv", index=False)
