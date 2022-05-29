from flask import Flask, redirect, render_template, request, url_for
from flask_pymongo import PyMongo

#set up the flask server
app = Flask(__name__, template_folder='templates')

#set routes for the page
@app.route('/')
def homepage():

    #return the data with a render template
    return render_template("index.html")


@app.route('/country')
def country_page():
    
    #return the data with a render template
    return render_template("country.html")

@app.route('/athlete', methods = ['POST','GET'])
def athlete_page():
    
    #return the data with a render template
    return render_template("athlete.html")

@app.route('/predict')
def predictionPage():
    if request.method == 'POST':
      AAAAA = request.form['NOC']
      return redirect(url_for('/country',BBBB = AAAAA))
    else:
      AAAAA = request.args.get('NOC')
      return redirect(url_for('/athlete',BBBB = AAAAA))
    # import predict function

@app.route('/sports')
def sport_page():

    #return the data with a render template
    return render_template("sport.html")

if __name__=="__main__":
    app.run(debug=True)