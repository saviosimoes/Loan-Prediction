import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv")
train = df.copy()
df2 = pd.read_csv("test.csv")
test = df2.copy()

#Cleaning Data Starts
train['Dependents'] =train['Dependents'].fillna(0)
train['Dependents'] = train['Dependents'].replace(to_replace ="3+", value ="3")
train['Dependents'] = pd.to_numeric(train['Dependents'])
train['Self_Employed'] =train['Self_Employed'].fillna('No')
train['LoanAmount'] =train['LoanAmount'].fillna(np.mean(train['LoanAmount']))
train['Loan_Amount_Term'] =train['Loan_Amount_Term'].fillna(np.mean(train['Loan_Amount_Term']))
train['Credit_History'] =train['Credit_History'].fillna( method ='ffill')

test['Dependents'] =test['Dependents'].fillna(0)
test['Dependents'] = test['Dependents'].replace(to_replace ="3+", value ="3")
test['Dependents'] = pd.to_numeric(test['Dependents'])
test['Self_Employed'] =test['Self_Employed'].fillna('No')
test['LoanAmount'] =test['LoanAmount'].fillna(np.mean(test['LoanAmount']))
test['Loan_Amount_Term'] =test['Loan_Amount_Term'].fillna(np.mean(test['Loan_Amount_Term']))
test['Credit_History'] =test['Credit_History'].fillna( method ='ffill')

print(df2.isnull().sum())
print(test.isnull().sum())

#print(train.columns)
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
train['Dependents'] = lab.fit_transform(train['Dependents'])
train['Education'] = lab.fit_transform(train['Education'])
train['Self_Employed'] = lab.fit_transform(train['Self_Employed'])
train['Property_Area'] = lab.fit_transform(train['Property_Area'])
train['Loan_Status'] = lab.fit_transform(train['Loan_Status'])

test['Dependents'] = lab.fit_transform(test['Dependents'])
test['Education'] = lab.fit_transform(test['Education'])
test['Self_Employed'] = lab.fit_transform(test['Self_Employed'])
test['Property_Area'] = lab.fit_transform(test['Property_Area'])
#Cleaning data ends

corrmatrix = train.corr()

print(train.columns)
X = train.iloc[:, [7,8,10,11]].values
X_test = test.iloc[:,[7,8,10,11]].values
y = train.iloc[:, -1].values

#Logistic Regression gives 80.94% accuracy
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X, y)

y_predlogistic = classifier.predict(X)
y_predtestlogistic = classifier.predict(X_test)

#Decision trees give 96% accuracy
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

y_predtree = regressor.predict(X)
y_predtree = np.around(y_predtree)
y_predtesttree = regressor.predict(X_test)
y_predtesttree = np.around(y_predtesttree)


#RandomForest gives 94% accuracy
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)

y_predrandom = classifier.predict(X)
y_predtestrandom = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
logcm = confusion_matrix(y, y_predlogistic)
print('Confusion matrix for Logistic Regression: ')
print(logcm)

from sklearn.metrics import confusion_matrix
treecm = confusion_matrix(y, y_predtree)
print('Confusion Matrix for Decision Trees')
print(treecm)

from sklearn.metrics import confusion_matrix
randomcm = confusion_matrix(y, y_predrandom)
print('Confusion Matrix for Random Forest')
print(randomcm)







