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
url = 'https://github.com/abhishekjha2468/Dont-Overfit/blob/main/LG.pkl?raw=true'
with open("LG.pkl",'wb') as output_file: output_file.write(requests.get(url).content)
url = 'https://github.com/abhishekjha2468/Dont-Overfit/blob/main/XGBoost.pkl?raw=true'
with open("XGBoost.pkl",'wb') as output_file: output_file.write(requests.get(url).content)
url = 'https://github.com/abhishekjha2468/Dont-Overfit/blob/main/LGBM.pkl?raw=true'
with open("LGMB.pkl",'wb') as output_file: output_file.write(requests.get(url).content)
url = 'https://github.com/abhishekjha2468/Dont-Overfit/blob/main/LASSO.pkl?raw=true'
with open("LASSO.pkl",'wb') as output_file: output_file.write(requests.get(url).content)
url = 'https://github.com/abhishekjha2468/Dont-Overfit/blob/main/scaler.pkl?raw=true'
with open("scaler.pkl",'wb') as output_file: output_file.write(requests.get(url).content)


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
  index_page="""<!DOCTYPE html>
<html>
<head>
	<title>Don't Overfit II</title>
	<!-- <link rel="stylesheet" type="text/css" href="https://github.com/abhishekjha2468/Dont-Overfit/blob/main/bootstrap.css">
	<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
	<link href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap" rel="stylesheet"> -->

<style type="text/css">
		#co{
	text-align: center;
	padding-top: 2%;
	text-shadow: 1px 2px 3px rgba(4,5,4,0.4),
	0px 8px 13px rgba(7,6,5,0.1),
	0px 18px 23px rgba(5,4,2,0.1);

}
body{
	background: url(https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1052&q=80);
	background-position: center;
	background-size:cover; : 
	font-family: lato;
}	
h4{
	color: white;
	font-weight: 5;
	font-size: 2em;
}
	
html{
	height: 100%;
}
hr{
	width: 400px;
	border-top:1px solid #f8f8f8;
	border-bottom: 1px solid rgba(5,3,1,0.2);	
}
#rr{
	text-align: center;
	padding-right: 0%
	

}
#ee{
	padding-top:0%;
	padding-left:0%;
}
#ww{
	padding-bottom: 5%;
	padding-right: 2%
}
#tt{
	padding-bottom: 3%
	padding-left:30%;
}
#ll{
	padding-bottom: 3%
	padding-left:0%;
}
h1{
	padding-bottom: 0%
	color:green;
	padding-right: 2%
}
.button {
  background-color: #e7e7e7;
  color: black;
  border: 4px solid #000000;
  padding: 20px 40px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 24px;
  margin: 4px 2px;
  opacity: 0.7;
  cursor: pointer;
  border-radius: 12px;
}
.Upload {
  background-color: #FFFFFF;
  color: black;
  border: 10px solid #000000;
  padding: 20px 40px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 24px;
  margin: 4px 2px;
  opacity: 0.6;
  cursor: pointer;
  border-radius: 12px;
}
.topnav {
  overflow: hidden;
  background-color: #333;
}

.topnav a {
  float: left;
  color: #f2f2f2;
  text-align: center;
  padding: 14px 16px;
  text-decoration: none;
  font-size: 17px;
}

.topnav a:hover {
  background-color: #ddd;
  color: black;
}

.topnav a.active {
  background-color: #04AA6D;
  color: white;
}
	</style>
	
</head>
<body>
	<div class="topnav">
	  <a class="active" >Home</a>
	  <a href="https://www.linkedin.com/in/abhishek-jha-0a0971120/">LinkedIN</a>
	  <a href="https://github.com/abhishekjha2468/Dont-Overfit">GitHub</a>
	  <a >Blog</a>
	  <!-- <a href="#about">About</a> -->
	</div>
	<!-- <nav class="navbar navbar-default navbar-fixed-top">
		<div class="container">
		<div class="navbar-header">
			   <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#bs-example-navbar-collapse-1" aria-expanded="false">
        <span class="sr-only">Toggle navigation</span>
        <span class="icon-bar"></span>
        <span class="icon-bar"></span>
        <span class="icon-bar"></span>
      </button>
   
			<a class="navbar-brand">Don't Overfit II </a>
		</div>
		  <div class="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
		<ul class="nav navbar-nav">
			<li class="active"><a href="#">Home</a></li>
			<li><a >ABOUT</a></li>
			
		</ul>
		<ul class="nav navbar-nav navbar-right"> -->
			<!-- <li><a href=""> SINGUP<i class="fa fa-user-plus"></i></a></li>
			<li><a href="">LOGIN<i class="fa fa-user"></i></a></li>
			<li><a href="https://www.linkedin.com/in/abhishek-jha-0a0971120/">LinkedIN</a></li>
			<li><a href="https://github.com/abhishekjha2468/Dont-Overfit">GitHub</a></li>
		</ul>
		</div>
	</nav> -->

			<div class="container">
				<div class="row">
					<div class="col-lr-12">
						<div id="co">
							<h1 style="color: #FFFFFF;"><b>Don't Overfit</b></h1><br></br>	
						<h4 id="ww"> <form method="POST" action="/predict" enctype="multipart/form-data">
						<!-- <label>Target value: </label> -->
  					    <!-- <p id="tt"><textarea name="feature", style="color: #000000;", size="50", rows=1></textarea></p> -->
  					    <!-- <label> Choose the : </label> -->
  					    <div id="ll">
  					    <label>Upload Your Query Dataset: </label></div>
  					    <div id="rr">
      					<p><b><input class="Upload" type="file" name="file" style="color: #000000;" size="10"></b></p></div>
      					<div id="ee">
      					<p><input class="button" type="submit" value="Submit" style="color: #000000;" size="10"></p></h4></div>
					</div>
						<hr>
						<!--<button class=""btn btn-default btn-lg><i class="fa fa-paw"></i>GET STARTED</button>-->	
					</div>
				</div>
			</div>
		</div>


	<script src="https://code.jquery.com/jquery-2.2.4.js"
  integrity="sha256-iT6Q9iMJYuQiMWNd9lDyBUStIq/8PuOW33aOqmvFpqI="
  crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" crossorigin="anonymous"></script>
<script type="text/javascript" scr="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js"></script>
</body>
</html>"""
  return index_page

@app.route('/predict', methods=['GET','POST'])
def predict():
	try:
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
	except e:
		html="ERROR OCCURED ADN THE ERROR IS \n " + str(e)
	# write html to file
	# text_file = open("output.html", "w")
	# text_file.write(html)
	# text_file.close()
	return html
# app.run()
