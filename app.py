from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the ML model
with open('wine.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            # Get input values from the form
            features = [
                float(request.form["fixed_acidity"]),
                float(request.form["volatile_acidity"]),
                float(request.form["citric_acid"]),
                float(request.form["residual_sugar"]),
                float(request.form["chlorides"]),
                float(request.form["free_sulfur_dioxide"]),
                float(request.form["total_sulfur_dioxide"]),
                float(request.form["density"]),
                float(request.form["pH"]),
                float(request.form["sulphates"]),
                float(request.form["alcohol"]),
            ]
            # Convert input data to numpy array and reshape for prediction
            input_data = np.array(features).reshape(1, -1)
            prediction = model.predict(input_data)
            # Set result based on prediction
            if prediction[0] == 1:
                result = "Good Quality Wine 🍷"
            else:
                result = "Bad Quality Wine 🍇"
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
