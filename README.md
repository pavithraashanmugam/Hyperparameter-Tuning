# Hyperparameter Tuning using GridSearchCV and RandomizedSearchCV

This repository demonstrates hyperparameter tuning techniques for machine learning models using `GridSearchCV` and `RandomizedSearchCV` in Python. It focuses on applying these techniques to a classification problem, using the breast cancer dataset available in the `sklearn.datasets` module.

### Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Steps](#steps)
  - [1. Loading the Dataset](#1-loading-the-dataset)
  - [2. Preprocessing the Data](#2-preprocessing-the-data)
  - [3. Hyperparameter Tuning with GridSearchCV](#3-hyperparameter-tuning-with-gridsearchcv)
  - [4. Hyperparameter Tuning with RandomizedSearchCV](#4-hyperparameter-tuning-with-randomizedsearchcv)
  - [5. Building the Model with Best Parameters](#5-building-the-model-with-best-parameters)
- [Results](#results)


### Overview

Hyperparameter tuning is a crucial step in optimizing machine learning models. `GridSearchCV` and `RandomizedSearchCV` are two commonly used techniques to find the best combination of hyperparameters for a given model.

In this project, we use the breast cancer dataset from scikit-learn to:
- Demonstrate the use of `GridSearchCV` to exhaustively search through a specified hyperparameter grid.
- Demonstrate the use of `RandomizedSearchCV` for a randomized search, which allows sampling from a range of hyperparameters.

The SVC (Support Vector Classification) model is used for classification in this project.

### Dependencies

This project requires the following Python libraries:

- `numpy`
- `pandas`
- `scikit-learn`

You can install the required dependencies by running:

```bash
pip install numpy pandas scikit-learn
```

### Dataset

The dataset used is the breast cancer dataset from `sklearn.datasets`. It consists of 569 samples with 30 features and a binary classification target (malignant or benign).

### Steps

#### 1. Loading the Dataset

```python
import sklearn.datasets
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
```

The dataset contains the following:

- `data`: 30 features describing the characteristics of cell nuclei present in breast cancer biopsies.
- `target`: 0 for malignant and 1 for benign tumors.

#### 2. Preprocessing the Data

The data is then converted into a pandas DataFrame for better understanding and manipulation:

```python
import pandas as pd
df = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
df['label'] = breast_cancer_dataset.target
```

- **Shape**: 569 rows and 31 columns (30 features + target column).
- **Missing Values**: No missing data in the dataset.
- **Target Distribution**: The target variable (`label`) is imbalanced with 212 malignant cases (0) and 357 benign cases (1).

The dataset is split into feature variables (`X`) and target labels (`Y`):

```python
X = df.drop(columns='label', axis=1)
Y = df['label']
```

#### 3. Hyperparameter Tuning with GridSearchCV

`GridSearchCV` is used to exhaustively search through a predefined grid of hyperparameters.

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define the model
model = SVC()

# Define the hyperparameters to tune
parameters = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [1, 5, 10, 20]
}

# Perform Grid Search
classifier = GridSearchCV(model, parameters, cv=5)
classifier.fit(X, Y)

# Get best parameters and accuracy
best_params = classifier.best_params_
highest_accuracy = classifier.best_score_
```

- **Best Parameters**: `{'C': 10, 'kernel': 'linear'}`
- **Highest Accuracy**: `0.9525`

The results of the grid search are stored in `classifier.cv_results_`, which contains the performance scores for each combination of hyperparameters.

#### 4. Hyperparameter Tuning with RandomizedSearchCV

RandomizedSearchCV performs a randomized search over hyperparameter space and is generally more efficient when searching a large space.

```python
from sklearn.model_selection import RandomizedSearchCV

# Define the hyperparameters to sample from
param_dist = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [1, 5, 10, 20, 100]
}

# Perform Randomized Search
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=5)
random_search.fit(X, Y)

# Get best parameters and accuracy
best_params_random = random_search.best_params_
highest_accuracy_random = random_search.best_score_
```

- **Best Parameters (RandomizedSearchCV)**: `{'C': 10, 'kernel': 'linear'}`
- **Highest Accuracy**: `0.9508`

#### 5. Building the Model with Best Parameters

Once the best hyperparameters are found using either `GridSearchCV` or `RandomizedSearchCV`, we can build a final model using those parameters and evaluate its performance on the dataset.

##### Using GridSearchCV Best Parameters:

```python
# Build the model with the best parameters found
best_model_grid = SVC(C=10, kernel='linear')

# Train the model
best_model_grid.fit(X, Y)

# Evaluate the model (e.g., accuracy)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Fit the model to the training data
best_model_grid.fit(X_train, Y_train)

# Predict on the test set
Y_pred = best_model_grid.predict(X_test)

# Calculate accuracy
accuracy_grid = accuracy_score(Y_test, Y_pred)
print(f"Accuracy of the model with GridSearchCV best parameters: {accuracy_grid:.4f}")
```

##### Using RandomizedSearchCV Best Parameters:

```python
# Build the model with the best parameters found from RandomizedSearchCV
best_model_random = SVC(C=10, kernel='linear')

# Train the model
best_model_random.fit(X, Y)

# Evaluate the model (e.g., accuracy)
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Fit the model to the training data
best_model_random.fit(X_train, Y_train)

# Predict on the test set
Y_pred_random = best_model_random.predict(X_test)

# Calculate accuracy
accuracy_random = accuracy_score(Y_test, Y_pred_random)
print(f"Accuracy of the model with RandomizedSearchCV best parameters: {accuracy_random:.4f}")
```

### Results

| Hyperparameter Tuning Method | Best Parameters              | Best Accuracy |
|------------------------------|------------------------------|---------------|
| GridSearchCV                  | {'C': 10, 'kernel': 'linear'} | 0.9525        |
| RandomizedSearchCV            | {'C': 10, 'kernel': 'linear'} | 0.9508        |

Both methods provide similar results for this problem, with `C=10` and `kernel='linear'` providing the best performance.
