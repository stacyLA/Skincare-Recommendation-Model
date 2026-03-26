import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load and preprocess the dataset
from preprocess import load_and_preprocess
data=load_and_preprocess("skincare_symptoms_disease_products_dataset.csv")
 
 # select features and target variable
x = data[['skin_type', 'symptom_1', 'symptom_2']]
y = data['recommended ingredient']

#split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Save the trained model
joblib.dump(model, 'skincare_model.pkl')

# Save the preprocessed data for future use
data.to_csv('preprocessed_data.csv', index=False)   

print("Model trained and saved successfully.")
