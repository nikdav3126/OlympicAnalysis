from flask import Flask, jsonify
from sqlalchemy import create_engine
import os
import numpy as np
import json
import pandas as pd
from decimal import Decimal 



# Connect to POSTGRES
##################################

rds_connection_string = "postgresql://sxmbucwgkkmyhj:3794ffbd79519b6577cf9f3a746b6578d8e89f0550b2765241e3ea4c384eb0c2@ec2-44-196-174-238.compute-1.amazonaws.com:5432/d94k0gpgqqtch6"
engine = create_engine(rds_connection_string)
#conn = engine.connect()



# Flask Setup
##################################

app = Flask(__name__)



# Flask Routes
###################################

@app.route("/")
def welcome():
    '''List Available API Routes'''
    return(
        f'<h2>Available Routes: </h2><br>'
        f'<h3>/api/v1.0/noc-regions</h3>'
        f'<h3>/api/v1.0/data</h3>'
        f'<h3>/api/v1.0/countrypredictions</h3>'
 

    )


@app.route("/api/v1.0/noc-regions")
def zip_pops():
    # Create our session (link) from Python to the DB

    data = engine.execute("SELECT * FROM nocregion")
    result = json.dumps([dict(r) for r in data])
    return result

@app.route("/api/v1.0/data")

def poplocs():
    
    data = engine.execute("SELECT latitude, longitude, estimated_population FROM zips")
    result = json.dumps([dict(r) for r in data])
    return result


@app.route("/api/v1.0/countrypredictions")
def pets():
    data = engine.execute('SELECT mastercity.primary_city, mastercity.latitude, mastercity.longitude, dogfriendly.overall_rank FROM mastercity INNER JOIN dogfriendly ON mastercity.primary_city=dogfriendly.city ORDER BY dogfriendly.overall_rank ASC')
    result = json.dumps([dict(r) for r in data])
    return result

@app.route("/api/v1.0/athletes")
def predictionData():
   
   from user_prediction import predict

   prediction = predict(sex,age,height,weight,NOC) 
   # prediction = [(Shooting, Archery, Horse stuff),(running, swimming, sleeping)]

   return redirect('/athletes') #with the data from prediction filled in

if __name__ == '__main__':
    app.run(debug=True)
