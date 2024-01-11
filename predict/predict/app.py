from flask import Flask, request
from predict.predict import run as run_predict
from train.train import run as run_train
import json

app = Flask(__name__)

@app.route('/predict', methods=["GET"])
def predict():
    artefact_path = "C:/Users/malob/Desktop/EPF 5A/EPF 5A/poc_to_prod/poc-to-prod-capstone/poc-to-prod-capstone/train/data/artefacts/2024-01-09-11-32-27"
    list_text = ['test']
    model = run_predict.TextPredictionModel.from_artefacts(artefact_path)
    prediction = model.predict(list_text, top_k=5)
    names = [model.labels_to_index[str(idx)] for idx in prediction[0]]
    result_json = json.dumps(names)
    return result_json


if __name__ == '__main__':
    app.run(debug=True)