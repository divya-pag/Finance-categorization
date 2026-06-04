# Financial Transaction Category Prediction

## Overview

Developed an end-to-end Machine Learning system to automatically classify financial transactions into predefined categories using transaction descriptions and transaction amounts. The solution helps automate transaction categorization and improve financial data organization.

## Dataset

* 200,000+ financial transaction records
* Features:

  * Transaction Description
  * Withdrawal Amount
  * Deposit Amount
* Target:

  * Category Prediction

## Project Workflow

1. Data Cleaning and Preprocessing
2. Feature Engineering
3. Text Vectorization using TF-IDF
4. Model Training and Evaluation
5. Category Prediction
6. Flask-based Deployment

## Model

* TF-IDF Vectorization for text feature extraction
* Logistic Regression for multi-class classification
* Label Encoding for category mapping

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Flask
* HTML

## Project Structure

```text
Financial-Category-Prediction/
│
├── data/
├── models/
├── src/
│   ├── data_cleaner.py
│   ├── train_model.py
│   └── predict_category.py
│
├── templates/
│   └── index.html
│
├── app.py
├── README.md
└── requirements.txt
```

## Key Highlights

* Processed and analyzed over 200,000 transaction records.
* Built a complete machine learning pipeline from preprocessing to deployment.
* Implemented text classification using TF-IDF and Logistic Regression.
* Developed a web interface for real-time transaction category prediction using Flask.

## Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open the application in your browser and enter transaction details to predict the transaction category.
