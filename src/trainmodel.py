import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from preprocess import load_and_preprocess

# Load and preprocess the dataset
data = load_and_preprocess("/home/linder/skincare_predictor/src/skincare_symptoms_disease_products_dataset.csv")

# Select features and target variable
# FIX: added 'disease' as a feature — it improves prediction accuracy
# FIX: corrected column name from 'recommended ingredient' (space) to 'recommended_ingredient' (underscore)
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

# Save the trained model
joblib.dump(model, 'skincare_model.pkl')
print("\nModel trained and saved successfully as 'skincare_model.pkl'.")

# Save the preprocessed data for reference
data.to_csv('preprocessed_data.csv', index=False)