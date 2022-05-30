from flask import Flask, redirect, render_template, request, url_for
import json

rds_connection_string = "postgresql://vkvodlagclammk:665830b16c2ac37d2d21b681e0d49f46dfc70833b169c43fb440602fdea73276@ec2-54-211-255-161.compute-1.amazonaws.com:5432/dcuj01hki6lv7t"
engine = create_engine(rds_connection_string)

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

#background process happening without any refreshing
@app.route('/predict', methods = ['GET','POST'])
def background_process_test():

    # POST request
    if request.method == 'POST':
        print(request.get_json())  # parse as JSON

        dicta = request.get_json()

        # print(dicta)

        Age = int(dicta['Age'])
        Sex = dicta['Sex']
        Height = int(dicta['Height'])
        Weight = int(dicta['Weight'])
        nat = str(dicta['NOC'])
        NOC = nat.split('~')[1]

        print(Sex, Age, Height, Weight, NOC)

        # result = '/* potato */'
        from User_Sport_Pred import user_predictor

        result = (user_predictor(Sex, Age, Height, Weight, NOC))
        
        print(result)

        return result

@app.route('/sports')
def sport_page():

    #return the data with a render template
    return render_template("sport.html")

if __name__=="__main__":
    app.run(debug=True)
