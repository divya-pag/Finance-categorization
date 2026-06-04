import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from predict_category import predict_category
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        description = request.form.get('description')
        withdrawal = float(request.form.get('withdrawal') or 0)
        deposit = float(request.form.get('deposit') or 0)      
        pred = predict_category(description, withdrawal, deposit)
        
        result = {
            'given_description': pred['description'],
            'cleaned_description': pred['used_description'],
            'predicted_category': pred['predicted_category']
        }

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug = True)
