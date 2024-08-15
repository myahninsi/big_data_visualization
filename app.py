from flask import Flask,render_template,request,redirect,jsonify
import pandas as pd # type: ignore
import os
from predictor import predict_prices,determine_model,zipcode_detail

app = Flask(__name__)

@app.route('/')                                                 #displays the housing price prediction view
def home():
    zipcodes=zipcode_detail()                                   #loading the zip codes with its labels
    return render_template("model.html",zipcodes=zipcodes)

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
def predict_price():
    try:
        data = {
            'bedrooms': [int(request.form.get('bedrooms', 0))],
            'bathrooms': [int(request.form.get('bathrooms', 0))],
            'sqft_living': [float(request.form.get('sqft_living', 0.0))],
            'sqft_lot': [float(request.form.get('sqft_lot', 0.0))],
            'floors': [int(request.form.get('floors', 0))],
            'waterfront': [int(request.form.get('waterfront', 0))],
            'view': [int(request.form.get('view', 0))],
            'condition': [int(request.form.get('condition', 0))],
            'grade': [int(request.form.get('grade', 0))],
            'sqft_basement': [float(request.form.get('sqft_basement', 0.0))],
            'yr_built': [int(request.form.get('yr_built', 0))],
            'sqft_lot15': [float(request.form.get('sqft_lot15', 0.0))],
            'Renovated': [int(request.form.get('renovated', 0))],
            'zipcode_label': [int(request.form.get('zipcode', 0))] 
        }
        
        df=pd.DataFrame(data)                       #converting the data entered by the user as a pandas data frame      
        mlr_name=request.form.get('regression')     #obtaines the name of the multiple linear regression technique
        model,r2,rmse=determine_model(mlr_name)     #retrieves the model and its r2 and rmse
        r2=round(r2,4)
        rmse=round(rmse,2)
        predicted_price=predict_prices(model,df)                #retrieves the price based on the model selected
        print(predicted_price)
        
        if mlr_name in ("mlr","mlr_ridge"):
            predicted_price=round(predicted_price[0][0],2)
        else:
            predicted_price=round(predicted_price[0],2)
        
        zipcodes=zipcode_detail()

        return render_template("model.html",predictions=True,predicted_price=predicted_price,r2=r2,rmse=rmse,mlr_name=mlr_name,zipcodes=zipcodes)
    except Exception as ex:
        return render_template("model.html",input_error=True)

if __name__ == '__main__':
    app.run(debug=True) #debug=True: the server will automatically reload if code changes