import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request

# Load and prepare the data
data = pd.read_csv('Housing.csv')
data.dropna(inplace=True)

# Map categorical variables to numeric
data['mainroad'] = data['mainroad'].map({'yes': 1, 'no': 0})
data['guestroom'] = data['guestroom'].map({'yes': 0.1, 'no': 0.0})
data['basement'] = data['basement'].map({'no': 0, 'yes': 1})
data['hotwaterheating'] = data['hotwaterheating'].map({'no': 0, 'yes': 1})
data['airconditioning'] = data['airconditioning'].map({'yes': 1, 'no': 0})
data['furnishingstatus'] = data['furnishingstatus'].map({'unfurnished': 0.0, 'semi-furnished': 0.1, 'furnished': 0.2})
data['prefarea'] = data['prefarea'].map({'yes': 1, 'no': 0})

# Define features (X) and target (y)
x = data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'mainroad', 'guestroom',
          'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']]
y = data[['price']]

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Train the model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories = int(request.form['stories'])
    parking = int(request.form['parking'])
    mainroad = int(request.form['mainroad'])
    guestroom = float(request.form['guestroom'])
    basement = int(request.form['basement'])
    hotwaterheating = int(request.form['hotwaterheating'])
    airconditioning = int(request.form['airconditioning'])
    prefarea = int(request.form['prefarea'])
    furnishingstatus = float(request.form['furnishingstatus'])

    # Prepare the input data for prediction
    input_data = np.array([[area, bedrooms, bathrooms, stories, parking, mainroad, guestroom,
                            basement, hotwaterheating, airconditioning, prefarea, furnishingstatus]])

    # Predict the house price
    predicted_price = lr.predict(input_data)

    # Return the result to the result.html template
    return render_template('result.html', prediction=predicted_price[0][0])

if __name__ == '__main__':
    app.run(debug=True)