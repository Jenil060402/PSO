import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pyswarm import pso


data = pd.read_csv("cleveland\Heart_disease_cleveland_new.csv")


X = data.drop(columns=["target"])
y = data["target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define the objective function
def objective_function(selected_features, X, y):
    selected_indices = np.where(selected_features)[0]
    if len(selected_indices) == 0:
        return 1e6  # Penalize if no features are selected
    X_subset = X[:, selected_indices]
    
    # Train a logistic regression model
    clf = LogisticRegression()
    clf.fit(X_subset, y)
    
    # Predict on the training set
    y_pred = clf.predict(X_subset)
    
    # Calculate accuracy
    return -accuracy_score(y, y_pred)  # Negative accuracy to maximize


# PSO feature selection
def pso_feature_selection(X, y):
    num_features = X.shape[1]

    def objective_function_wrapper(selected_features):
        return objective_function(selected_features, X, y)

    lb = np.zeros(num_features)  # Lower bounds for feature selection (binary vector)
    ub = np.ones(num_features)   # Upper bounds for feature selection (binary vector)

    # Perform PSO
    best_features, _ = pso(objective_function_wrapper, lb, ub, swarmsize=10, maxiter=20)

    best_indices = np.where(best_features)[0]
    return best_indices


best_feature_indices = pso_feature_selection(X_train_scaled, y_train)
print("Selected feature indices:", best_feature_indices)
