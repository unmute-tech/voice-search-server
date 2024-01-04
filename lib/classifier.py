import joblib

def create_classifier(model_path):
    svm_model = joblib.load(f'{model_path}/svm_pipeline.pkl')
    label_to_id = joblib.load(f'{model_path}/labels.pkl')
    id_to_label = {v:k for k,v in label_to_id.items()}

    return Classifier(svm_model, id_to_label)

class Classifier:

    def __init__(self, svm_model, id_to_label):
        self.svm_model = svm_model
        self.id_to_label = id_to_label

    def classify(self, transcript):
        probs = self.svm_model.predict_proba([transcript])[0]
        predictions = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)

        return [(prob, self.id_to_label[i]) for i, prob in predictions]
