import pandas as pd
import joblib
import os

from data_cleaner import clean_text  # your cleaning function
from train_model import train_model  # fallback if model missing

# Paths to model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL = os.path.join(BASE_DIR, '../models/final_model.pkl')
LABEL_ENCODER_PKL = os.path.join(BASE_DIR, '../models/final_label_encoder.pkl')

# Load model & encoder; train if missing
if os.path.exists(MODEL_PKL) and os.path.exists(LABEL_ENCODER_PKL):
    model = joblib.load(MODEL_PKL)
    label_encoder = joblib.load(LABEL_ENCODER_PKL)
else:
    print("Model or label encoder not found. Training new model...")
    model, label_encoder = train_model()

# prediction func - smaller joblib code of trained model

def predict_category(description, withdrawal = 0, deposit = 0, confidence_threshold = 0.4):
    cleaned = clean_text(description)

    input_df = pd.DataFrame([{
        "Description_clean" : cleaned,
        "Withdrawal Amount": withdrawal,
        "Deposit Amount": deposit
    }])

# predict probabilities
    prob = model.predict_proba(input_df)[0]
    pred_encoded = prob.argmax()
    confidence = prob[pred_encoded]

    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    if confidence < confidence_threshold:
        pred_label = "Uncertain"

    all_class_probs = {
            label_encoder.inverse_transform([i])[0]: round(float(p), 3)
            for i, p in enumerate(prob)
        }

    return {
        'description': description,
        'used_description' : cleaned,
        'predicted_category' : pred_label,
        'confidence': round(float(confidence),2),
        'all_class_probabilities': all_class_probs
    }   

# testing

if __name__ == '__main__':

    test_description = "UPI-SAGAR-BDO #2345"
    test_withdrawal = 2214
    test_deposit = 0
    result = predict_category(test_description, test_withdrawal, test_deposit)
    print("\nPrediction Output:")
    print(f"Description: {result['description']}")
    print(f"Used Description: {result['used_description']}")
    print(f"Predicted Category: {result['predicted_category']}")
    print(f"Confidence: {result['confidence']}")
    print("All Class Probabilities:")

    for cat, prob in result['all_class_probabilities'].items():
        print(f"{cat} : {prob}")
