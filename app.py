from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the trained model
# Replace this with the actual path to your trained model
model = RandomForestClassifier()
model.fit(X_train, y_train)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the incoming JSON data
        data = request.get_json()

        # Convert the JSON data to a DataFrame
        input_data = pd.DataFrame([data])

        # Make predictions using the model
        predictions = model.predict(input_data)

        # Convert predictions to a response JSON
        response = {'prediction': predictions.tolist()}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
