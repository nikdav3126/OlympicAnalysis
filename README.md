# OlympicAnalysis


# Synopsis
Olympic Games are the leading international sporting events with more than 200 nations and thousands of athletes participating in a variety of competitions.  The first modern Games started in Athens, Greece in 1896.

Our group’s focus is to use machine learning to predict:
a country’s likelihood of winning medals in a specific sport 
the sports in which a user is most and least likely to succeed 
…in any future Olympics.

# Data Exploration
Our data was trimmed down to focus on summer sport events happening from 1960 to 2016 for 212 NOCs (National Olympic Committee aka country). We tested over 150,000 instances of athletes competing.

Olympics datasets were obtained from kaggle.com:
“Olympic games results vs GDP vs Population”
“120 years of Olympic history: athletes and results”
“Worldbank Data”

# Datasets:
●	https://www.kaggle.com/code/vhlaca/olympic-games-results-vs-gdp-vs-population/data
●	https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results



# Technologies Used:
Python Pandas
HTML/CSS/ Bootstrap
JavaScript Plotly
JavaScript D3.js
PostgreSQL
Heroku
SciKit 
Tensorflow 

# Data Flow
![Screen Shot 2022-05-31 at 6 50 50 PM](https://user-images.githubusercontent.com/94502554/171308216-c6a15006-8527-4b3d-a4a3-5b6f231c6d8f.png)

# Machine Learning
Data .csv files were imported and read in by pandas
Columns renamed
Unnecessary columns dropped
Boolean values (true/false) created for medals (gold,silver, bronze)
Bins created for future one hot encoding

The machine learning algorithm, Logistic Regression, was used to fit, model, and predict:

Each country’s top 5 medal wins by sport
The best and worst sports a user would best suited to perform in based on an athlete’s physique, age, and country of origin

# Deployed interactive website
https://fierce-tundra-81326.herokuapp.com/



