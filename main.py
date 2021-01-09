# importing the necessary dependencies
from flask import Flask, render_template, request, jsonify
import sklearn
import pickle
import joblib

app = Flask(__name__)  # initializing a flask app


@app.route('/')  # route to display the home page
def homePage():
    return render_template("index.html")




@app.route('/predict', methods=['POST'])  # route to show the predictions in a web UI
def index():
    # # reading the inputs given by the user
    fixed_acidity = (request.form['fixed_acidity'])
    volatile_acidity = (request.form['volatile_acidity'])
    citric_acid = (request.form['citric_acid'])
    residual_sugar = (request.form['residual_sugar'])
    chlorides = (request.form['chlorides'])
    free_sulfur_dioxide = (request.form['free_sulfur_dioxide'])
    total_sulfur_dioxide = (request.form['total_sulfur_dioxide'])
    density = (request.form['density'])
    pH = (request.form['pH'])
    sulphates = (request.form['sulphates'])
    alcohol = (request.form['alcohol'])

    filename = "suraj.sav"
    loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
    # predictions using the loaded model file
    prediction = loaded_model.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
]])
    print('prediction is', prediction)
    # showing the prediction results in a UI
    return render_template('results.html', prediction=prediction[0])


# @app.route('/po', methods=['POST', 'GET'])  # route to show the predictions in a web UI
# def postm():
#
#     print(request.get_json(force=True))
#     data= request.get_json(force=True)
#     fixed_acidity = (data['fixed_acidity'])
#     volatile_acidity = (data['volatile_acidity'])
#     citric_acid = (data['citric_acid'])
#     residual_sugar = (data['residual_sugar'])
#     chlorides = (data['chlorides'])
#     free_sulfur = (data['free_sulfur'])
#     total_sulfur_dioxide = (data['total_sulfur'])
#     density = (data['density'])
#     pH = (data['pH'])
#     sulphates = (data['sulphates'])
#     alcohol = (data['alcohol'])
#
#     filename = "suraj.sav"
#     loaded_model = joblib.load(open(filename, 'rb'))  # loading the model file from the storage
#     # predictions using the loaded model file
#     prediction = loaded_model.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,chlorides, free_sulfur, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
#     print('prediction is', prediction)
#     prediction = (prediction[0])
#     # showing the prediction results in a UI
#     return jsonify({'Prediction': prediction})


if __name__ == "__main__":
    app.run(debug=True)  # running the app
