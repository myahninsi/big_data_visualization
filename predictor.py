import pandas as pd             # type: ignore
import joblib                   # type: ignore
import matplotlib
matplotlib.use('Agg')           # Select Matplolib Backend
import matplotlib.pyplot as plt
import numpy as np              # type: ignore
import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # type: ignore

def load_model_and_scaler(model_path='pickle/linear_regression_model.pkl', scaler_path='pickle/minmax_scaler.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_prices(model, scaler,new_data):
    #new_data = pd.read_csv('data/test.csv')
    X_new = new_data.drop(columns=['price'])
    y_new = new_data[["price"]]
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
    predictions_rounded = np.round(predictions, 2)
    new_data['predicted_price'] = predictions_rounded
    y_pred=new_data[["predicted_price"]]
    return y_new, y_pred

def evaluate_model(y_new, y_pred):
    mse = mean_squared_error(y_new, y_pred)
    rmse = math.sqrt(mean_squared_error(y_new, y_pred))
    mae = mean_absolute_error(y_new, y_pred)
    r2 = r2_score(y_new, y_pred)
    return mse,rmse,mae,r2

def plot_predictions_vs_actuals(y_new, y_pred):
    #plt.figure(figsize=(10, 5))
    plt.scatter(y_new, y_pred,label="Predictions")
    #plt.plot(y_new,y_pred,color="red",linestyle="--",label="Perfect fit")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.savefig("static/img/actual_vs_predicted.png")
    plt.close()