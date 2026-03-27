import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_preprocess(path):
    # Load the dataset using the passed-in path (was hardcoded with a broken string before)
    data = pd.read_csv(path)

    # Store label encoders so they can be reused during prediction
    encoders = {}

    # Encode all categorical columns
    categorical_cols = ['skin_type', 'symptom_1', 'symptom_2', 'disease',
                        'recommended_ingredient', 'recommended_product_type']

    for col in categorical_cols:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder  # Save each encoder for later use in predict.py

    # Save encoders to disk so predict.py can reuse the same label mappings
    joblib.dump(encoders, 'encoders.pkl')

    return data