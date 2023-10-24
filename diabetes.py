import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

data = pd.read_csv('diabetes.csv')
print(data.head())

# Splitting the data into independent and dependent variables
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Splitting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Saving the model
joblib.dump(model, 'diabetes_model.pkl')

# Loading the model
model = joblib.load('diabetes_model.pkl')

# Testing the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

cm = confusion_matrix(y_test, predictions)
print('Confusion Matrix:\n', cm)

report = classification_report(y_test, predictions)
print('Classification Report:\n', report)

