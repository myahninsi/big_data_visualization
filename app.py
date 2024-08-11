from flask import Flask,render_template,request,redirect,jsonify
import pandas as pd # type: ignore
import os
from predictor import load_model_and_scaler, predict_prices, evaluate_model,plot_predictions_vs_actuals,determine_model,data_files_names

app = Flask(__name__)

@app.route('/')
def home():
    file_names=data_files_names()
    return render_template("index.html",file_names=file_names)

@app.route('/visualize')
def dashboard():
    return render_template("dashboard.html")

@app.route('/data/<filename>')
def get_data(filename):
    file_path=os.path.join("data",filename)
    data=pd.read_csv(file_path)
    headers=data.columns.tolist()
    rows=data.values.tolist()
    return jsonify({'headers':headers,'rows':rows})         #for dynamic update of the table it returns a json to use it with javscript in the HTML

@app.route('/predict',methods=['POST'])
def predict():
    mrg_name = request.form.get('selectedOption')           #obtains the name of multiple linear regression (mrg) to be executed
    fn=request.form.get('selectedOption2')                  #obtains the name of the file (fn) in this case csv
    model_path=determine_model(mrg_name)                    #retrieving the path of the respective model
    model, scaler = load_model_and_scaler(model_path)       #retrieving the multiple linear regression model and scaler
    #test_data = pd.read_csv('data/test.csv')               #loading a dataset  
    data = pd.read_csv('data/{}'.format(fn))
    y_new,y_pred = predict_prices(model, scaler,data)       #returning the actual, and predicted dependent variable
    mse,rmse,mae,r2 = evaluate_model(y_new, y_pred)         #return evaluation metrics
    metrics={"mse":mse,"rmse":rmse,"mae":mae,"r2":r2}
    predictions=True
    plot_predictions_vs_actuals(y_new,y_pred)               #just indicates that the predictions were performed and allow to display a section in HTML
    return render_template("index.html", headings=data.columns.tolist(), data=data.values.tolist(), predictions=predictions,metrics=metrics)

@app.route('/upload',methods=['POST'])
def upload_file():                                          #saves a file (csv)
    file=request.files['file']
    saving_path="data/{}".format(file.filename)
    file.save(saving_path)
    upload_files=True                                       #allow to display in frontend that file was uploaded successfully
    #actual_data = pd.read_csv(saving_path) 
    #headings=actual_data.columns.tolist()
    #data=actual_data.values.tolist()
    file_names=data_files_names()
    return render_template("index.html",upload_files=upload_files,file_names=file_names)

if __name__ == '__main__':
    app.run(debug=True) #debug=True: the server will automatically reload if code changes