import glob
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


def create_classifier(model_path):
    svm_model = joblib.load(f'{model_path}/svm_pipeline.pkl')
    label_to_id = joblib.load(f'{model_path}/labels.pkl')

    return Classifier(path, svm_model, label_to_id)

class Classifier:

    def __init__(self, path, svm_model, label_to_id):
        self.path = path
        self.svm_model = svm_model
        self.label_to_id = label_to_id
        self.id_to_label = {v:k for k,v in label_to_id.items()}

    def classify(self, transcript):
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
