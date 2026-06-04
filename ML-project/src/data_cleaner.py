import pandas as pd 
import re 
import os

# root directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(BASE_DIR, '../data/raw/raw_transaction_data.csv')
CLEANED_CSV = os.path.join(BASE_DIR, '../data/processed/cleaned_data.csv')

def clean_text(text):
    text = str(text).upper()
    text = re.sub(r'-', ' ', text)            # replace hyphen with space
    text = re.sub(r'\d+', '', text)        #remove numbers               pattern,replacement
    text = re.sub(r'[^A-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)   #remove special characters
    return text.strip()

def clean_data():
    df = pd.read_csv(RAW_CSV)
    df['Withdrawal Amount'] = df['Withdrawal Amount'].fillna(0)
    df['Deposit Amount'] = df['Deposit Amount'].fillna(0)
    df['Description_clean'] = df['Description'].apply(clean_text)

    print("\nCategory Counts: ")
    print(df['Category Name'].value_counts())

    os.makedirs(os.path.dirname(CLEANED_CSV), exist_ok = True)
    df.to_csv(CLEANED_CSV, index = False)

    print("Cleaned data saved to data/processed/")
    return df

if __name__ == "__main__":
    clean_data()
