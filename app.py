from flask import Flask,request, jsonify,render_template  
import pickle 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler 



app = Flask(__name__) 

## import ridge model and scaler
ridge_model = pickle.load(open('./models/l2_rig.pkl', 'rb'))
scaler = pickle.load(open('./models/scaler.pkl', 'rb'))

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_FWI():
        # Get Data from form
        Temp = float(request.form['Temp'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Regions = float(request.form['Regions'])

        # Scale the input data
        input_data_scaled = scaler.transform([[Temp, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Regions]])

        #Model Prediction
        prediction = ridge_model.predict(input_data_scaled)[0]
        prediction = np.round(prediction, 2)
        return render_template('index.html', prediction_fwi=prediction)

if __name__ == '__main__':
    app.run(debug=True)