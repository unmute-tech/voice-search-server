import click
import glob
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore") 

def load_data(data_dir):
    labels = {}
    X = []
    y = []

    for path in glob.glob(f'{data_dir}/*.label'):
        with open(path, 'r') as f:
            label = f.readline().strip()

        with open(path.replace('.label', '.txt'), 'r') as f:
            text = f.readline().strip()

        if label not in labels:
            labels[label] = len(labels)
        
        X.append(text)
        y.append(labels[label])

    return X, y, labels

@click.command()
@click.argument('data_dir')
@click.argument('model_dir')
def train(data_dir, model_dir):
    X, y, labels = load_data(data_dir)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), stop_words=None, norm='l2')),
        ('SVC', SVC(probability=True, kernel='rbf', random_state=0))
    ])
    pipeline.fit(X, y)

    joblib.dump(pipeline, f'{model_dir}/svm_pipeline.pkl', compress=True)
    joblib.dump(labels, f'{model_dir}/labels.pkl', compress=True)
    print("Model trained successfully")


if __name__ == '__main__':
    train()
