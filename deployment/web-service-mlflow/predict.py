import pickle
import mlflow
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction import DictVectorizer

RUN_ID = "1a52002c34b44032992e9210dc149c10"
logged_model = f'{RUN_ID}/artifacts/models_mlflow/'
model = mlflow.pyfunc.load_model(logged_model)
preprocessor_path = f'{RUN_ID}/artifacts/preprocessor/preprocessor.b'
with open(preprocessor_path, 'rb') as f_in:
    dv = pickle.load(f_in)


def prepare_features(ride):
    print(f'ride: {ride}')
    features = {}
    features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return f"{preds[0]:.2f}"


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
