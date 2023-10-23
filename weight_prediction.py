# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load the data
data = pd.read_csv('weight_data.csv')

# If 'gender' column exists in your dataset, uncomment the following line to map 'gender' to numerical values
# data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})

# Split the data into features (X) and target (y)
X = data.drop('weight_category', axis=1)
y = data['weight_category']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier and train it on the training data
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Use the trained model to make predictions on the test data
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Compute the confusion matrix
cm = confusion_matrix(y_test, predictions)
print('Confusion Matrix:\n', cm)

# Print classification report
report = classification_report(y_test, predictions)
print(report)

# Save the model
joblib.dump(model, 'model.pkl')

# Load the model
model = joblib.load('model.pkl')
