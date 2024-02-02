import glob
import joblib
import os.path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


def create_classifier(model_path):
    last_change_time = os.path.getctime(f'{model_path}/lock')
    svm_model = joblib.load(f'{model_path}/svm_pipeline.pkl')
    label_to_id = joblib.load(f'{model_path}/labels.pkl')

    return Classifier(model_path, last_change_time, svm_model, label_to_id)

class Classifier:

    def __init__(self, path, last_change_time, svm_model, label_to_id):
        self.path = path
        self.last_change_time = last_change_time
        self.svm_model = svm_model
        self.label_to_id = label_to_id
        self.id_to_label = {v:k for k,v in label_to_id.items()}

    def reload_model_if_newer_model_exists(self):
        if os.path.getctime(f'{self.path}/lock') <= self.last_change_time:
             return

        print('New model exists, reloading.')
        new_classifier = create_classifier(self.path)
        self.last_change_time = new_classifier.last_change_time
        self.svm_model = new_classifier.svm_model
        self.label_to_id = new_classifier.label_to_id
        self.id_to_label = new_classifier.id_to_label

    def classify(self, transcript):
        self.reload_model_if_newer_model_exists()

        probs = self.svm_model.predict_proba([transcript])[0]
        predictions = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)

        return [(prob, self.id_to_label[i]) for i, prob in predictions]

    def train(self, data_dir):
        X, y, labels = self.load_data(data_dir)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 3), stop_words=None, norm='l2')),
            ('SVC', SVC(probability=True, kernel='rbf', random_state=0))
        ])
        pipeline.fit(X, y)

        joblib.dump(pipeline, f'{self.path}/svm_pipeline.pkl', compress=True)
        joblib.dump(labels, f'{self.path}/labels.pkl', compress=True)
        open(f'{self.path}/lock', 'w').close()

    def load_data(self, data_dir):
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
