import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from preprocess import load_and_preprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load and preprocess the dataset
dataset_path = BASE_DIR / 'skincare_symptoms_disease_products_dataset.csv'
data = load_and_preprocess(dataset_path)

# Select features and target variable

X = data[['skin_type', 'symptom_1', 'symptom_2', 'disease']]
y = data['recommended_ingredient']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

model_path = BASE_DIR / 'skincare_model.pkl'
joblib.dump(model, model_path)
print(f"\nModel trained and saved successfully as '{model_path.name}'.")

preprocessed_path = BASE_DIR / 'preprocessed_data.csv'
data.to_csv(preprocessed_path, index=False)