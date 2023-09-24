import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

print("--------------------\nHEART DISEASE PREDICTION\n--------------------")
print("loading files....")
hd=pd.read_csv('heart.csv')
hd.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved','exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
y=hd.target.values
hd.drop(["target"],inplace=True,axis=1)
x=hd.values

scaler = StandardScaler().fit(x)

x=scaler.transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =0.25,random_state=0)
print("loading complete....\n\n")
'''
DT=DecisionTreeClassifier()
DT.fit(x_train,y_train)
DT.score(x_test,y_test)
DT.score(x_train,y_train)
DT.score(x,y)

KNN=KNeighborsClassifier()
KNN.fit(x_train,y_train)
KNN.score(x_test,y_test)
KNN.score(x_train,y_train)
KNN.score(x,y)

GN=GaussianNB()
GN.fit(x_train,y_train)
GN.score(x_test,y_test)
GN.score(x_train,y_train)
GN.score(x,y)
'''
RF=RandomForestClassifier(n_estimators=100)
RF.fit(x_train,y_train)
RF.score(x_train,y_train)
RF.score(x_test,y_test)
RF.score(x,y)
l=[]
l.append(int(input("Enter your age : ")))
l.append(int(input("Enter your sex 0 or 1 [mail=1,femail=0] : ")))
l.append(int(input("chest pain experienced \n 0 = None\n 1 = typical angina\n 2 = atypical angina\n 3 = non-anginal pain\n 4 = asymptomatic\nEnter the type number : ")))
l.append(int(input("resting blood pressure : ")))
l.append(int(input("cholesterol measurement in mg/dl : ")))
l.append(int(input("fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false) : ")))
l.append(int(input("Resting electrocardiographic measurement :\n 0 = normal\n 1 = having ST-T wave abnormality\n 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria) \nEnter the number :")))
l.append(int(input("maximum heart rate achieved : ")))
l.append(int(input("Exercise induced angina (1 = yes; 0 = no) : ")))
l.append(float(input("ST depression induced by exercise relative to rest \n('ST' relates to positions on the ECG plot.) (eg: 2.3): ")))
l.append(int(input("slope of the peak exercise ST segment \n 1 = upsloping\n 2 = flat\n 3 = downsloping \nEnter the number : ")))
l.append(int(input("The number of major vessels (0-3) : ")))
l.append(int(input("A blood disorder called thalassemia \n 1 = normal\n 2 = fixed defect\n 3 = reversable defect\nEnter the number : ")))
#[41,0,1,130,204,0,0,172,0,1.4,2,0,2]
x=[l]
x=scaler.transform(x)
pre=RF.predict(x)[0]
if pre==1:
	print("\nYou have a high chance of getting heart disease.\nTAKE CARE OF YOUR SELF")
else:
	print("\nYou have a low chance of getting heart disease.\nHAVE A NICE DAY")
x=input()