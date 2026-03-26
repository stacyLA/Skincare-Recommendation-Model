import pandas as pd
from sklearn.preprocessing import LabelEncoder
def load_and_preprocess(path):
    # Load the dataset
  data = pd.read_csv("filedata/skincare_symptoms_disease_products_dataset.csv_path")
  #converting text to numbers
  encoder=LabelEncoder()
  data['skin_type']=encoder.fit_transform(data['skin_type'])
  data['symptom_1']=encoder.fit_transform(data['symptom_1'])
  data['symptom_2']=encoder.fit_transform(data['symptom_2'])
  data['disease']=encoder.fit_transform(data['disease'])
  data['recommended_ingredient']=encoder.fit_transform(data['recommended_ingredient']) 
  return data
    
