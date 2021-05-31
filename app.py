from flask import Flask, request, jsonify
from get_pred_on_single_video import Predictor

app = Flask(__name__)

pred = Predictor()

@app.route('/', methods=['GET'])
def index():
    return 'Barfi'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    output = pred.get_final_labels_in_video(data['url'])
    return jsonify(output)

if __name__ == '__main__':
    app.run()
