import pickle
from flask import Flask, request, jsonify
import numpy as np

# declare constants
HOST = '0.0.0.0'
PORT = 5000

# initialize flask application
app = Flask(__name__)

#loading linear model from file
with open("./static/model/lm_model.pkl","rb") as m:
    model = pickle.load(m)

# api endpoint to use linear model to make price prediction
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    # get data from post request
    data = request.get_json()
    if data.keys() >= {'LotArea', 'YearRemodAdd', 'GarageArea', 'OverallQual', 'YearBuilt'}:
        # make prediction
        arr = np.array([[data['LotArea'], data['YearRemodAdd'],
                         data['GarageArea'], data['OverallQual'],
                         data['YearBuilt']]])
        prediction = model.predict(arr).item(0)
        return jsonify(status=201, prediction=round(prediction, 2))
    else:
        # return error status
        return jsonify(status=400)


if __name__ == '__main__':
    app.run(host=HOST,
            debug=True,
            port=PORT)
