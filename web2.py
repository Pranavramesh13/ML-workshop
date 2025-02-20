import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Example model

# Load dataset (replace with your actual dataset)
X, y = np.random.rand(1000, 8), np.random.randint(0, 2, 1000)

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (replace with your actual model)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

# Save model and accuracy
with open("diabetes_model.sav", "wb") as model_file:
    pickle.dump(model, model_file)

with open("diabetes_accuracy.pkl", "wb") as acc_file:
    pickle.dump(model_accuracy, acc_file)

print(f"Model Accuracy: {model_accuracy:.2f}")  # Just for checking
