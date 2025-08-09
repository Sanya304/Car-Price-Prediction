

# from flask import Flask, render_template, request
# from flask_cors import CORS, cross_origin
# import pickle
# import pandas as pd
# import numpy as np

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Load model and dataset
# model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
# car = pd.read_csv('Cleaned_Car_data.csv')

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     companies = sorted(car['company'].unique())
#     car_models = sorted(car['name'].unique())
#     years = sorted(car['year'].unique(), reverse=True)
#     fuel_types = car['fuel_type'].unique()

#     companies.insert(0, 'Select Company')
#     return render_template(
#         'index.html',
#         companies=companies,
#         car_models=car_models,
#         years=years,
#         fuel_types=fuel_types
#     )

# @app.route('/predict', methods=['POST'])
# @cross_origin()
# def predict():
#     try:
#         company = request.form.get('company')
#         car_model = request.form.get('car_models')
#         year = request.form.get('year')
#         fuel_type = request.form.get('fuel_type')
#         driven = request.form.get('kilo_driven')

#         # Create DataFrame with exact same column names as training
#         input_df = pd.DataFrame({
#             'name': [car_model],
#             'company': [company],
#             'year': [int(year)],
#             'kms_driven': [int(driven)],
#             'fuel_type': [fuel_type]
#         })

#         prediction = model.predict(input_df)
#         output = np.round(prediction[0], 2)

#         return str(output)

#     except Exception as e:
#         return f"Error: {str(e)}"

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model and dataset
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types
    )

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')
        driven = request.form.get('kilo_driven')

        # Create DataFrame with exact same column names as training
        input_df = pd.DataFrame({
            'name': [car_model],
            'company': [company],
            'year': [int(year)],
            'kms_driven': [int(driven)],
            'fuel_type': [fuel_type]
        })

        prediction = model.predict(input_df)
        output = np.round(prediction[0], 2)

        return str(output)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    # Use PORT from environment (for Render) or 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
