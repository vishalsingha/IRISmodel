# import library
import numpy as np 
import pandas as pd 
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# read data
df = pd.read_csv("C:\\Users\\Vishal Singh\\Desktop\\IRIS.csv")

# get label
Y = df['species'].astype('category').cat.codes

# preprocess data
s = StandardScaler()

X = df.drop(['species'] ,axis = 1)
s.fit(X)
pickle.dump(s, open('standardscaler.pkl', 'wb'))
X = s.transform(X)

# split train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


# creating classifier
clf = LogisticRegression()
clf.fit(x_train, y_train)

# print model accuracy
print(clf.score(x_test, y_test))

# dump model
pickle.dump(clf, open('model.pkl', 'wb'))
