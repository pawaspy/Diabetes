# Diabetes Prediction Project

This project uses a Random Forest Classifier to predict whether a person has diabetes or not based on various diagnostic measurements.

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

You need to have the following libraries installed:

- pandas
- scikit-learn
- joblib

You can install them using pip:

`pip install pandas numpy scikit-learn joblib`

### Running the Project

1. Clone the repository to your local machine:

`clone https://github.com/pawaspy/Diabetes-Prediction`

2. Navigate to the project directory:

`cd Diabetes-Prediction`

3. Run the Python script:

python diabetes.py


## Project Structure

The project contains the following files:

- `diabetes_prediction.py`: This is the main script that loads the data, trains the model, makes predictions, and evaluates the model.

- `model.pkl`: This is the trained model saved as a pickle file. You can load this model to make predictions without having to retrain it.

- `diabetes.csv`: This is the dataset used for training. It contains diagnostic measurements for several patients along with an 'Outcome' column indicating whether the patient has diabetes.

## Model Evaluation

The model's performance is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score

The confusion matrix and classification report are also printed to provide a more detailed breakdown of the model's performance.