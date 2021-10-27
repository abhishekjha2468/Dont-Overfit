import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import math
#downloading model from my github
#github id --> https://github.com/abhishekjha2468
import requests
# url = 'https://github.com/abhishekjha2468/Dont-Overfit/blob/main/LG.pkl?raw=true'
# with open("LG.pkl",'wb') as output_file: output_file.write(requests.get(url).content)
# url = 'https://github.com/abhishekjha2468/Dont-Overfit/blob/main/XGBoost.pkl?raw=true'
# with open("XGBoost.pkl",'wb') as output_file: output_file.write(requests.get(url).content)
# url = 'https://github.com/abhishekjha2468/Dont-Overfit/blob/main/LGBM.pkl?raw=true'
# with open("LGMB.pkl",'wb') as output_file: output_file.write(requests.get(url).content)
# url = 'https://github.com/abhishekjha2468/Dont-Overfit/blob/main/LASSO.pkl?raw=true'
# with open("LASSO.pkl",'wb') as output_file: output_file.write(requests.get(url).content)
# url = 'https://github.com/abhishekjha2468/Dont-Overfit/blob/main/scaler.pkl?raw=true'
# with open("scaler.pkl",'wb') as output_file: output_file.write(requests.get(url).content)


#loading model from downloaded pickle file
import pickle
with open("LG.pkl", 'rb') as file:  LR_Model = pickle.load(file)
with open("XGBoost.pkl", 'rb') as file:  XGB_Model = pickle.load(file)
with open("LGMB.pkl", 'rb') as file:  LGMB_Model = pickle.load(file)
with open("LASSO.pkl", 'rb') as file:  LASSO_Model = pickle.load(file)
with open("scaler.pkl", 'rb') as file:  scaler = pickle.load(file)


def prediction(query):
  """
  This function take query dataset as an input and return its predicted class probability 
  Note: input dataframe with 300 columns, named 0-299
  """
  df=pd.DataFrame(scaler.transform(query[list(map(str,range(300)))]))
  df["ceil_min"]=list(map(lambda x: math.ceil(min(x)) ,scaler.transform(query[list(map(str,range(300)))])))
  df["mean"]=list(map(lambda x: np.mean(x) ,scaler.transform(query[list(map(str,range(300)))])))
  df["sum"]=list(map(lambda x: np.sum(x) ,scaler.transform(query[list(map(str,range(300)))])))
  df["max_x_mean"]=list(map(lambda x: np.max(x)*np.mean(x) ,scaler.transform(query[list(map(str,range(300)))])))
  df["min_x_sum"]=list(map(lambda x: np.min(x)*np.sum(x) ,scaler.transform(query[list(map(str,range(300)))])))
  lr_pred=LR_Model.predict_proba(df)[:,1]
  xgb_pred=XGB_Model.predict_proba(df)[:,1]
  lgmb_pred=LGMB_Model.predict_proba(df)[:,1]
  lasso_pred=LASSO_Model.predict(df)
  pred=lr_pred*0.1 + xgb_pred*0.1 + lgmb_pred*0.1 + lasso_pred*0.7
  return pred

# url="https://raw.githubusercontent.com/abhishekjha2468/Dont-Overfit/main/index.html"
# with open("index.html",'wb') as output_file: output_file.write(requests.get(url).content)



# import shutil
# try: shutil.rmtree("templates")
# except: pass
# os.mkdir("templates")
# shutil.copy("index.html","templates/index.html")

# !pip install flask_ngrok

from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
app = Flask(__name__)
# run_with_ngrok(app) 

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  file = request.files['file']
  fileName = file.filename
  file.save(secure_filename(file.filename))
  print(fileName)
  df=pd.read_csv(fileName)
  p=prediction(df)
  df=pd.DataFrame()
  df["index"]=list(range(len(p)))
  df["Prediction"]=p
  html = df.to_html()  
  # write html to file
  # text_file = open("output.html", "w")
  # text_file.write(html)
  # text_file.close()
  return html
# app.run()
