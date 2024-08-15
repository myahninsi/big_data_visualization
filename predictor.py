import pandas as pd                                                             # type: ignore
import joblib
import pickle                                                                   # type: ignore
import matplotlib
matplotlib.use('Agg')                                                           # Select Matplolib Backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np                                                              # type: ignore
import math
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error   # type: ignore
import scipy.stats as stats

def zipcode_detail():
    zd=pd.read_excel("data/zipcode_detail.xlsx")
    zd_detail=zd.to_dict(orient="records")
    return zd_detail

def determine_model(name):                                                      #retrieve the models and r2 and rmse
    with open('pickle/multiple_linear_regression/mlr.pkl', 'rb') as handle:
        pickle_dict = pickle.load(handle)
    
    if name=="mlr":
        model=pickle_dict["linear_regression"]
        model_r2=pickle_dict["linear_regression_r2"]
        model_rmse=pickle_dict["linear_regression_rmse"]
        return model,model_r2,model_rmse
    elif name=="mlr_lasso":
        model=pickle_dict["lasso_regression"]
        model_r2=pickle_dict["lasso_regression_r2"]
        model_rmse=pickle_dict["lasso_regression_rmse"]
        return model,model_r2,model_rmse
    elif name=="mlr_ridge":
        model=pickle_dict["ridge_regression"]
        model_r2=pickle_dict["ridge_regression_r2"]
        model_rmse=pickle_dict["ridge_regression_rmse"]
        return model,model_r2,model_rmse
    elif name=="mlr_elasticnet":
        model=pickle_dict["elastic_net_regression"]
        model_r2=pickle_dict["elastic_net_regression_r2"]
        model_rmse=pickle_dict["elastic_net_regression_rmse"]
        return model,model_r2,model_rmse
    else:
        pass
    
def predict_prices(model,data):
    with open('pickle/multiple_linear_regression/mlr.pkl', 'rb') as handle:
        pickle_dict = pickle.load(handle)
    
    #model=pickle_dict["linear_regression"]
    scaler=pickle_dict["scaler"]
    features=pickle_dict["features"]

    scaled_data=scaler.transform(data[features])
    y_pred = model.predict(scaled_data)
    return y_pred



    residuals = y_new.values.flatten() - y_pred.values.flatten()
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Normal Q-Q Plot')
    plt.savefig("static/img/QQ_plot.png")
    plt.close()