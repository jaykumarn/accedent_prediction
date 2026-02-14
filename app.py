
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

def load_or_create_model():
    model_path = 'test1.pkl'
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError()
        return pickle.load(open(model_path, 'rb'))
    except (ValueError, ModuleNotFoundError, FileNotFoundError):
        print("Model incompatible with current sklearn version. Retraining...")
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        
        df = pd.read_csv('accidents_india.csv')
        df.dropna(inplace=True)
        
        df.Sex_Of_Driver = df.Sex_Of_Driver.fillna(df.Sex_Of_Driver.mean())
        df.Vehicle_Type = df.Vehicle_Type.fillna(df.Vehicle_Type.mean())
        df.Speed_limit = df.Speed_limit.fillna(df.Speed_limit.mean())
        df.Road_Type = df.Road_Type.fillna(df.Road_Type.mean())
        df.Number_of_Pasengers = df.Number_of_Pasengers.fillna(df.Speed_limit.mean())
        
        c = LabelEncoder()
        df['Day'] = c.fit_transform(df['Day_of_Week'])
        df.drop('Day_of_Week', axis=1, inplace=True)
        
        l = LabelEncoder()
        df['Light'] = l.fit_transform(df['Light_Conditions'])
        df.drop('Light_Conditions', axis=1, inplace=True)
        
        s = LabelEncoder()
        df['Severity'] = s.fit_transform(df['Accident_Severity'])
        df.drop('Accident_Severity', axis=1, inplace=True)
        
        x = df.drop(['Pedestrian_Crossing', 'Special_Conditions_at_Site', 'Severity'], axis=1)
        y = df['Severity']
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.86, random_state=42)
        
        reg = DecisionTreeClassifier(criterion='gini')
        reg.fit(x_train, y_train)
        
        pickle.dump(reg, open(model_path, 'wb'))
        print(f"New model saved. Score: {reg.score(x_test, y_test)}")
        return reg

test = load_or_create_model()


@app.route('/')
def hello_world():
    return render_template("t.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=test.predict(final)

    if prediction == 0:
        return render_template('t.html',pred="\t\t\t\t\tProbability of accident severity is : Minor")
    else:
        return render_template('t.html',pred="\t\t\t\t\tProbability of accident severity is : Major")

@app.route('/Map')
def map1():
    return render_template("map.html")    

@app.route('/Graphs')
def graph():
    return render_template("graph.html")

@app.route('/Map1')
def map2():
    return render_template("ur.html")

@app.route('/Map2')
def map3():
    return render_template("bs.html")

@app.route('/Map3')
def map4():
    return render_template("hm.html")

@app.route('/Pie')
def pie():
    return render_template("pie.html")


if __name__=="__main__":
    app.run() 
