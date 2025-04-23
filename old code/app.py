from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load the model
#notebook_path = Path().resolve()
#model_path = notebook_path.parent /"model"/'churn_model.pkl'
#print(f"Looking for model at: {model_path}")



# Load test data for analytics
#test_data_path = notebook_path.parent /"data"/'test.csv'
test_df = pd.read_csv(r'data\test.csv')
model = joblib.load(r'model\churn_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Create DataFrame with the same structure as training data
        input_data = {
            'gender_Male': [1 if data['gender'] == 'Male' else 0],
            'MultipleLines_No phone service': [1 if data['multipleLines'] == 'No phone service' else 0],
            'MultipleLines_Yes': [1 if data['multipleLines'] == 'Yes' else 0],
            'InternetService_Fiber optic': [1 if data['internetService'] == 'Fiber optic' else 0],
            'InternetService_No': [1 if data['internetService'] == 'No' else 0],
            'Contract_One year': [1 if data['contract'] == 'One year' else 0],
            'Contract_Two year': [1 if data['contract'] == 'Two year' else 0],
            'PaperlessBilling_Yes': [1 if data['paperlessBilling'] == 'Yes' else 0],
            'PaymentMethod_Credit card (automatic)': [1 if data['paymentMethod'] == 'Credit card' else 0],
            'PaymentMethod_Electronic check': [1 if data['paymentMethod'] == 'Electronic check' else 0],
            'PaymentMethod_Mailed check': [1 if data['paymentMethod'] == 'Mailed check' else 0],
            'hasFamily_Yes': [1 if data['hasFamily'] == 'Yes' else 0],
            'SeniorCitizen': [1 if data['seniorCitizen'] else 0],
            'AvgMonthlyCost': [float(data['monthlyCost'])]
        }
        
        # Create DataFrame
        df = pd.DataFrame(input_data)
        
        # Ensure all columns are present (fill missing with 0)
        expected_columns = [
            'gender_Male', 'MultipleLines_No phone service', 'MultipleLines_Yes',
            'InternetService_Fiber optic', 'InternetService_No', 'Contract_One year',
            'Contract_Two year', 'PaperlessBilling_Yes', 
            'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
            'PaymentMethod_Mailed check', 'hasFamily_Yes', 'SeniorCitizen', 'AvgMonthlyCost'
        ]
        
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training data
        df = df[expected_columns]
        
        # Make prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0][1]  # Probability of churn (Yes)
        
        return jsonify({
            'prediction': 'Yes' if prediction[0] == 'Yes' else 'No',
            'probability': float(probability),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


@app.route('/analytics')
def analytics():
    X_test = test_df.drop(['Churn'], axis=1) if 'Churn' in test_df.columns else test_df
    
    # Make predictions on test data
    y_true = test_df['Churn'] if 'Churn' in test_df.columns else None
    y_pred = model.predict(X_test)
    
    # Generate confusion matrix image
    cm_image = None
    if y_true is not None:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        cm_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    
    # Generate classification report
    report = None
    if y_true is not None:
        report = classification_report(y_true, y_pred, output_dict=True)
    
    # Get potential churners
    # Adding probability scores
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability of churn
    
    # Create churners DataFrame with customer IDs and probabilities
    churners_df = test_df.copy()
    churners_df['Churn_Probability'] = probabilities
    churners_df['Predicted_Churn'] = y_pred
    
    # Filter potential churners (probability > 0.5 or as predicted by the model)
    potential_churners = churners_df[churners_df['Predicted_Churn'] == 'Yes'].sort_values(
        by='Churn_Probability', ascending=False
    ).head(100)  # Limit to top 100 highest probability
    
    # Convert to dict for template rendering
    potential_churners_dict = potential_churners.to_dict('records')
    
    return render_template(
        'analytics.html',
        confusion_matrix_img=cm_image,
        classification_report=report,
        potential_churners=potential_churners_dict
    )

if __name__ == '__main__':
    app.run(debug=True)