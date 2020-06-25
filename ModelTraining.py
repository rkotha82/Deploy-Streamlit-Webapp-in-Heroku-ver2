import pandas as pd
import numpy as np

df = pd.read_csv('BankNote_Authentication.csv')
print(df)

# Independent and Dependent Features
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X.head()
y.head(10)

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Implement Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

# Check Accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
score

# Create a Pickle file using Serialization
import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()

print(classifier.predict([[2, 3, -4, 1]]))



