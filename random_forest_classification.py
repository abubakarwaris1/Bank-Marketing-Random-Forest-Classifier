# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\\Udemy - Machine Learning\\practice\\Bank Marketing-Random Forest Classifier\\bank_marketing_dataset.csv')
X = dataset.iloc[:, 0:16].values
y = dataset.iloc[:, 16].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1]= labelencoder_X.fit_transform(X[:, 1])
X[:, 2]= labelencoder_X.fit_transform(X[:, 2])
X[:, 3]= labelencoder_X.fit_transform(X[:, 3])
X[:, 4]= labelencoder_X.fit_transform(X[:, 4])
X[:, 6]= labelencoder_X.fit_transform(X[:, 6])
X[:, 7]= labelencoder_X.fit_transform(X[:, 7])
X[:, 8]= labelencoder_X.fit_transform(X[:, 8])
X[:, 10]= labelencoder_X.fit_transform(X[:, 10])
X[:, 15]= labelencoder_X.fit_transform(X[:, 15])
onehotencoder = OneHotEncoder(categorical_features = [1,2,3,4,6,7,8,10,15])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)



# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



