
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Global variables to store model, scaler, and feature columns
model_ttf = None
scaler = None
feature_columns = []

# Function to read and train model from uploaded file
def train_model(file_path):
    global model_ttf, scaler, feature_columns

    # Read dataset from uploaded file
    data = pd.read_csv(file_path)
    
    # Data cleaning
    data = data.dropna()
    
    # Define feature columns
    feature_columns = ['cycle', 'setting1', 'setting2', 'setting3']
    feature_columns += [f's{i}' for i in range(1, 22)]
    feature_columns += [f'av{i}' for i in range(1, 22)]
    feature_columns += [f'sd{i}' for i in range(1, 22)]
    
    # Normalize data
    scaler = StandardScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    
    # Data distribution
    X = data[feature_columns]
    y_ttf = data['ttf']
    
    # Split data into training and testing set
    X_train, X_test, y_ttf_train, y_ttf_test = train_test_split(X, y_ttf, test_size=0.2, random_state=42)
    
    # Build TTF estimation model
    model_ttf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_ttf.fit(X_train, y_ttf_train)
    
    # Predict TTF for testing data
    y_ttf_pred = model_ttf.predict(X_test)
    
    # Evaluate TTF model
    mse = mean_squared_error(y_ttf_test, y_ttf_pred)
    rmse = np.sqrt(mse)
    return rmse

# Main page for uploading file and entering data
@app.route('/', methods=['GET', 'POST'])
def index():
    rmse = None
    result = None

    if request.method == 'POST':
        # Upload reference file
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                rmse = train_model(file_path)

        # Manual data input
        if 'cycle' in request.form:
            global model_ttf, scaler, feature_columns
            if model_ttf is not None and scaler is not None:
                user_data = {col: [float(request.form.get(col, 0))] for col in feature_columns}
                user_input_df = pd.DataFrame(user_data)
                user_input_df[feature_columns] = scaler.transform(user_input_df[feature_columns])
                user_ttf_pred = model_ttf.predict(user_input_df)

                if user_ttf_pred[0] <= 100:
                    result = 'Pesawat harus diperbaiki!'
                else:
                    result = 'Pesawat masih dalam kondisi baik.'

    return render_template('index.html', rmse=rmse, result=result)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
