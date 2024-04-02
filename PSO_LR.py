# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from pyswarm import pso
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the dataset
data = pd.read_csv("cleveland/Heart_disease_cleveland_new.csv")

# Separate features and target variable
X = data.drop(columns=["target"])
y = data["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if the trained model file exists
if os.path.exists("trained_classifier_RF.pkl"):
    # Load the trained model from the file
    clf = joblib.load("trained_classifier_RF.pkl")
    print("Trained model loaded from file.")
else:
    # Define the fitness function to be maximized (classification accuracy)
    def fitness_function(features, X_train, X_test, y_train, y_test):
        selected_features = np.where(features)[0]
        if len(selected_features) == 0:
            return 0.0  # Return zero fitness if no features are selected
        # Train a classifier using selected features
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train.iloc[:, selected_features], y_train)
        # Evaluate the classifier on the test set
        y_pred = clf.predict(X_test.iloc[:, selected_features])
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    # Define the PSO optimization function
    def optimize_features(X_train, X_test, y_train, y_test):
        n_features = X_train.shape[1]
        # Define the bounds for each feature (0 for not selected, 1 for selected)
        lb = np.zeros(n_features)
        ub = np.ones(n_features)
        # Perform PSO optimization
        features, _ = pso(fitness_function, lb, ub, args=(X_train, X_test, y_train, y_test), swarmsize=30, maxiter=400)
        return features

    # Run PSO to select the best subset of features
    selected_features = optimize_features(X_train, X_test, y_train, y_test)

    # Print the selected features
    print("Selected features indices:", np.where(selected_features)[0])

    # Train and evaluate classifier with selected features
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train.iloc[:, selected_features.astype(bool)], y_train)

    # Save trained classifier to a file
    joblib.dump(clf, "trained_classifier_RF.pkl")
    print("Trained model saved to file.")

# Use the trained model for predictions
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# Save selected features to a file
if not os.path.exists("selected_features_RF.npy"):
    np.save("selected_features_RF.npy", selected_features)
    print("Selected features saved to file.")
else:
    print("Selected features file already exists.")
