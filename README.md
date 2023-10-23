# Weight Category Prediction Project

This project uses a Random Forest Classifier to predict a person's weight category based on their height and weight.

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

You need to have the following libraries installed:

- pandas
- numpy
- scikit-learn
- joblib

You can install them using pip:

`pip install pandas numpy scikit-learn joblib`

### Running the Project

1. Clone the repository to your local machine:

`clone https://github.com/pawaspy/Weight-Category-Prediction`

2. Navigate to the project directory:

`cd Weight-Category-Prediction`

3. Run the Python script:

python weight_prediction.py

## Project Structure

The project contains the following files:

- `weight_prediction.py`: This is the main script that loads the data, trains the model, makes predictions, and evaluates the model.

- `model.pkl`: This is the trained model saved as a pickle file. You can load this model to make predictions without having to retrain it.

- `data.csv`: This is the dataset used for training. It contains columns for height, weight, and weight category.

## Model Evaluation

The model's performance is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score

The confusion matrix is also printed to provide a more detailed breakdown of the model's performance.

