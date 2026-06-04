import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

from data_cleaner import clean_data

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_CSV = os.path.join(BASE_DIR, '../data/processed/cleaned_data.csv')
MODEL_PKL = os.path.join(BASE_DIR, '../models/final_model.pkl')
LABEL_ENCODER_PKL = os.path.join(BASE_DIR, '../models/final_label_encoder.pkl')

def train_model():

    if not os.path.exists(CLEANED_CSV) or os.path.getsize(CLEANED_CSV) == 0:
        print("Cleaned file missing or empty. Running data cleaning...")
        df = clean_data()
    else:
        df = pd.read_csv(CLEANED_CSV)

# spilt
    X = df[["Description_clean", "Withdrawal Amount", "Deposit Amount"]]
    y = df['Category Name']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)

# label encoding - gives labels to each category(food-0, entertainment-1)
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

# column transformer
    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(), 'Description_clean'),
        ('num', StandardScaler(), ['Withdrawal Amount', 'Deposit Amount'])
    ])

# pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter = 2000, class_weight = 'balanced'))
    ])

# training
    model.fit(X_train, y_train_enc) #undergoes pipeline

    print("Training Accuracy:", model.score(X_train, y_train_enc))
    print("Testing Accuracy:", model.score(X_test, y_test_enc))           #correct preictions/total predictions

    os.makedirs(os.path.dirname(MODEL_PKL), exist_ok = True)

# save model
    joblib.dump(model, MODEL_PKL, compress = 3)
    joblib.dump(label_encoder, LABEL_ENCODER_PKL, compress = 3)

    print(f'Model saved to models/')
    return model, LABEL_ENCODER_PKL

if __name__ == "__main__":
   train_model()

