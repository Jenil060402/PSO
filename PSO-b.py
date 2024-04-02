import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from pyswarm import pso

# Load the dataset (Cleveland dataset)
data = pd.read_csv("cleveland/Heart_disease_cleveland_new.csv")

# Separate features and target variable
X = data.drop(columns=["target"])
y = data["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the fitness function to be maximized (classification accuracy)
def fitness_function(features, X_train, X_test, y_train, y_test):
    selected_features = np.where(features)[0]
    if len(selected_features) == 0:
        return 0.0  # Return zero fitness if no features are selected
    # Train a classifier using selected features
    clf = LogisticRegression()
    clf.fit(X_train.iloc[:, selected_features], y_train)
    # Evaluate the classifier on the test set
    y_pred = clf.predict(X_test.iloc[:, selected_features])
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with {len(selected_features)} selected features: {accuracy:.4f}")
    return accuracy  # Return accuracy for maximization

# Define the PSO optimization function
def optimize_features(X_train, X_test, y_train, y_test):
    n_features = X_train.shape[1]
    # Define the bounds for each feature (0 for not selected, 1 for selected)
    lb = np.zeros(n_features)
    ub = np.ones(n_features)
    # Perform PSO optimization
    features, _ = pso(fitness_function, lb, ub, args=(X_train, X_test, y_train, y_test), swarmsize=25, maxiter=1000)
    return features

# Run PSO to select the best subset of features
selected_features = optimize_features(X_train, X_test, y_train, y_test)

# Print the selected features
print("Selected features indices:", np.where(selected_features)[0])

# Train and evaluate classifiers with different feature subsets
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear'),
    "kNN": KNeighborsClassifier(n_neighbors=5)
}

for name, clf in classifiers.items():
    clf.fit(X_train.iloc[:, selected_features.astype(bool)], y_train)
    y_pred = clf.predict(X_test.iloc[:, selected_features.astype(bool)])
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy with selected features: {accuracy:.4f}")
