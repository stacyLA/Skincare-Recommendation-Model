import joblib
import numpy as np

def predict_ingredient(skin_type: str, symptom_1: str, symptom_2: str):
   
    # Load the saved model and encoders
    model = joblib.load('skincare_model.pkl')
    encoders = joblib.load('encoders.pkl')

    # --- Validate inputs against known labels ---
    for col, value in [('skin_type', skin_type), ('symptom_1', symptom_1), ('symptom_2', symptom_2)]:
        known_labels = list(encoders[col].classes_)
        if value not in known_labels:
            raise ValueError(f"Unknown value '{value}' for '{col}'. "
                             f"Valid options are: {known_labels}")

    #  Encode user inputs 
    skin_type_enc = encoders['skin_type'].transform([skin_type])[0]
    symptom_1_enc = encoders['symptom_1'].transform([symptom_1])[0]
    symptom_2_enc = encoders['symptom_2'].transform([symptom_2])[0]

    # Estimate disease from symptoms (most common disease for this symptom pair) 
   
    disease_labels = encoders['disease'].classes_
    best_prob = -1
    best_ingredient = None
    best_disease = None

    for disease_str in disease_labels:
        disease_enc = encoders['disease'].transform([disease_str])[0]
        input_features = np.array([[skin_type_enc, symptom_1_enc, symptom_2_enc, disease_enc]])
        proba = model.predict_proba(input_features)[0]
        max_prob = proba.max()
        if max_prob > best_prob:
            best_prob = max_prob
            best_ingredient = model.predict(input_features)[0]
            best_disease = disease_str

    # Decode the predicted ingredient back to its original string label
    predicted_ingredient = encoders['recommended_ingredient'].inverse_transform([best_ingredient])[0]

    return {
        "skin_type": skin_type,
        "symptom_1": symptom_1,
        "symptom_2": symptom_2,
        "estimated_disease": best_disease,
        "recommended_ingredient": predicted_ingredient
    }


# example usage
if __name__ == "__main__":
    result = predict_ingredient(
        skin_type="dry",
        symptom_1="red patches",
        symptom_2="irritation"
    )
    print("\n--- Skincare Recommendation ---")
    for key, value in result.items():
        print(f"  {key}: {value}")