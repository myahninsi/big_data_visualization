from flask import Flask,render_template
import pandas as pd # type: ignore
from predictor import load_model_and_scaler, predict_prices, evaluate_model,plot_predictions_vs_actuals

app = Flask(__name__)

@app.route('/')
def home():
    test_data = pd.read_csv('data/test.csv') 
    headings=test_data.columns.tolist()
    data=test_data.values.tolist()
    return render_template("index.html",headings=headings,data=data)

@app.route('/visualize')
def dashboard():
    return render_template("dashboard.html")

@app.route('/predict',methods=['POST'])
def predict():
    model, scaler = load_model_and_scaler()                 #loading the linear regression model and scaler
    test_data = pd.read_csv('data/test.csv')                #loading the dataset  
    y_new,y_pred = predict_prices(model, scaler,test_data)  #returning the actual, and predicted dependent variable
    mse,rmse,mae,r2 = evaluate_model(y_new, y_pred)         #return evaluation metrics
    metrics={"mse":mse,"rmse":rmse,"mae":mae,"r2":r2}
    predictions=True
    plot_predictions_vs_actuals(y_new,y_pred)                                        #just indicates that the predictions were performed and allow to display a section in HTML
    return render_template("index.html", headings=test_data.columns.tolist(), data=test_data.values.tolist(), predictions=predictions,metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True) #debug=True: the server will automatically reload if code changes