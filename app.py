import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
import joblib
import pandas as pd
import numpy as np
import random
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import base64
import io

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Load the model
model = joblib.load(r'model\churn_model.pkl')
test_df = pd.read_csv(r'data\test.csv')

# Prepare test data predictions for analytics
X_test = test_df.drop(['Churn'], axis=1) if 'Churn' in test_df.columns else test_df
y_true = test_df['Churn'] if 'Churn' in test_df.columns else None
y_pred = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]  # Probability of churn

# Function to generate fake names
def generate_fake_names(n):
    first_names = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", 
        "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", 
        "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa", 
        "Matthew", "Margaret", "Anthony", "Betty", "Mark", "Sandra", "Donald", "Ashley", 
        "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle", 
        "Kenneth", "Carol", "Kevin", "Amanda", "Brian", "Dorothy", "George", "Melissa", 
        "Edward", "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Sharon", 
        "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Helen", 
        "Nicholas", "Amy", "Eric", "Shirley", "Jonathan", "Angela", "Stephen", "Anna", 
        "Larry", "Ruth", "Justin", "Brenda", "Scott", "Pamela", "Brandon", "Nicole"
    ]
    
    last_names = [
        "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", 
        "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", 
        "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez", "Lewis", "Lee", 
        "Walker", "Hall", "Allen", "Young", "Hernandez", "King", "Wright", "Lopez", 
        "Hill", "Scott", "Green", "Adams", "Baker", "Gonzalez", "Nelson", "Carter", 
        "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell", "Parker", "Evans", 
        "Edwards", "Collins", "Stewart", "Sanchez", "Morris", "Rogers", "Reed", "Cook", 
        "Morgan", "Bell", "Murphy", "Bailey", "Rivera", "Cooper", "Richardson", "Cox", 
        "Howard", "Ward", "Torres", "Peterson", "Gray", "Ramirez", "James", "Watson", 
        "Brooks", "Kelly", "Sanders", "Price", "Bennett", "Wood", "Barnes", "Ross"
    ]
    
    names = []
    for _ in range(n):
        first = random.choice(first_names)
        last = random.choice(last_names)
        names.append(f"{first} {last}")
    
    return names

# Prepare churners dataframe with fake names
churners_df = test_df.copy()
churners_df['Churn_Probability'] = probabilities
churners_df['Predicted_Churn'] = y_pred

# Generate fake names and add to dataframe
if 'CustomerID' not in churners_df.columns:
    churners_df['CustomerID'] = range(1, len(churners_df) + 1)
customer_names = generate_fake_names(len(churners_df))
churners_df['CustomerName'] = customer_names

# App layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Index page layout
index_layout = html.Div([
    html.Div([
        html.H1("Customer Churn Prediction", className="app-header"),
        html.P("Predict customer churn based on their profile information")
    ], style={'textAlign': 'center', 'margin-bottom': '30px'}),
    
    html.Div([
        html.Div([
            html.Div([
                html.Label("Gender:"),
                dcc.Dropdown(
                    id='gender',
                    options=[
                        {'label': 'Male', 'value': 'Male'},
                        {'label': 'Female', 'value': 'Female'}
                    ],
                    value='Male'
                )
            ], className="input-group"),
            
            html.Div([
                html.Label("Senior Citizen:"),
                dcc.RadioItems(
                    id='senior-citizen',
                    options=[
                        {'label': 'Yes', 'value': 1},
                        {'label': 'No', 'value': 0}
                    ],
                    value=0,
                    inline=True
                )
            ], className="input-group"),
            
            html.Div([
                html.Label("Has Family:"),
                dcc.RadioItems(
                    id='has-family',
                    options=[
                        {'label': 'Yes', 'value': 'Yes'},
                        {'label': 'No', 'value': 'No'}
                    ],
                    value='No',
                    inline=True
                )
            ], className="input-group"),
            
            html.Div([
                html.Label("Multiple Lines:"),
                dcc.Dropdown(
                    id='multiple-lines',
                    options=[
                        {'label': 'Yes', 'value': 'Yes'},
                        {'label': 'No', 'value': 'No'},
                        {'label': 'No phone service', 'value': 'No phone service'}
                    ],
                    value='No'
                )
            ], className="input-group"),
            
            html.Div([
                html.Label("Internet Service:"),
                dcc.Dropdown(
                    id='internet-service',
                    options=[
                        {'label': 'DSL', 'value': 'DSL'},
                        {'label': 'Fiber optic', 'value': 'Fiber optic'},
                        {'label': 'No', 'value': 'No'}
                    ],
                    value='DSL'
                )
            ], className="input-group"),
            
            html.Div([
                html.Label("Contract:"),
                dcc.Dropdown(
                    id='contract',
                    options=[
                        {'label': 'Month-to-month', 'value': 'Month-to-month'},
                        {'label': 'One year', 'value': 'One year'},
                        {'label': 'Two year', 'value': 'Two year'}
                    ],
                    value='Month-to-month'
                )
            ], className="input-group"),
            
            html.Div([
                html.Label("Paperless Billing:"),
                dcc.RadioItems(
                    id='paperless-billing',
                    options=[
                        {'label': 'Yes', 'value': 'Yes'},
                        {'label': 'No', 'value': 'No'}
                    ],
                    value='No',
                    inline=True
                )
            ], className="input-group"),
            
            html.Div([
                html.Label("Payment Method:"),
                dcc.Dropdown(
                    id='payment-method',
                    options=[
                        {'label': 'Electronic check', 'value': 'Electronic check'},
                        {'label': 'Mailed check', 'value': 'Mailed check'},
                        {'label': 'Bank transfer', 'value': 'Bank transfer'},
                        {'label': 'Credit card', 'value': 'Credit card'}
                    ],
                    value='Mailed check'
                )
            ], className="input-group"),
            
            html.Div([
                html.Label("Monthly Cost ($):"),
                dcc.Input(
                    id='monthly-cost',
                    type='number',
                    min=0,
                    max=250,
                    step=0.01,
                    value=70.0
                )
            ], className="input-group"),
            
            html.Button('Predict', id='predict-button', n_clicks=0, 
                      style={'marginTop': '20px', 'width': '100%', 'backgroundColor': '#4CAF50', 'color': 'white'})
        ], className="form-container", style={'width': '40%', 'margin': '0 auto'}),
        
        html.Div([
            html.Div(id='prediction-output', className="prediction-result")
        ], style={'textAlign': 'center', 'marginTop': '30px'}),
        
        html.Div([
            html.A("View Analytics", href="/analytics", className="analytics-link")
        ], style={'textAlign': 'center', 'marginTop': '40px'})
    ])
])

# Analytics page layout
analytics_layout = html.Div([
    html.H1("Customer Churn Analytics", className="app-header", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.H3("Confusion Matrix"),
            dcc.Graph(id='confusion-matrix')
        ], className="analytics-card", style={'width': '48%'}),
        
        html.Div([
            html.H3("ROC Curve"),
            dcc.Graph(id='roc-curve')
        ], className="analytics-card", style={'width': '48%'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'}),
    
    html.Div([
        html.H3("Classification Metrics"),
        html.Div(id='classification-report-output')
    ], className="analytics-card", style={'marginBottom': '30px'}),
    
    html.Div([
        html.H3("Feature Importance", style={'textAlign': 'center'}),
        dcc.Graph(id='feature-importance')
    ], className="analytics-card"),
    
    html.Div([
        html.H3("Top Potential Churners", style={'textAlign': 'center'}),
        html.Div([
            dash_table.DataTable(
                id='churners-table',
                columns=[
                    {"name": "Customer Name", "id": "CustomerName"},
                    {"name": "Churn Probability", "id": "Churn_Probability", "type": "numeric", "format": {"specifier": ".2%"}},
                    {"name": "Predicted Churn", "id": "Predicted_Churn"}
                ],
                data=churners_df[churners_df['Predicted_Churn'] == 'Yes'].sort_values(
                    by='Churn_Probability', ascending=False
                ).head(100).to_dict('records'),
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                page_size=10,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                style_table={'overflowX': 'auto'}
            )
        ])
    ], className="analytics-card"),
    
    html.Div([
        html.A("Back to Prediction", href="/", className="back-link")
    ], style={'textAlign': 'center', 'marginTop': '40px'})
])

# Callback to render the appropriate page layout
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/analytics':
        return analytics_layout
    else:
        return index_layout

# Callback for prediction
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [
        State('gender', 'value'),
        State('senior-citizen', 'value'),
        State('has-family', 'value'),
        State('multiple-lines', 'value'),
        State('internet-service', 'value'),
        State('contract', 'value'),
        State('paperless-billing', 'value'),
        State('payment-method', 'value'),
        State('monthly-cost', 'value')
    ]
)
def predict(n_clicks, gender, senior_citizen, has_family, multiple_lines, 
            internet_service, contract, paperless_billing, payment_method, monthly_cost):
    if n_clicks > 0:
        try:
            # Create input data dictionary
            input_data = {
                'gender_Male': [1 if gender == 'Male' else 0],
                'MultipleLines_No phone service': [1 if multiple_lines == 'No phone service' else 0],
                'MultipleLines_Yes': [1 if multiple_lines == 'Yes' else 0],
                'InternetService_Fiber optic': [1 if internet_service == 'Fiber optic' else 0],
                'InternetService_No': [1 if internet_service == 'No' else 0],
                'Contract_One year': [1 if contract == 'One year' else 0],
                'Contract_Two year': [1 if contract == 'Two year' else 0],
                'PaperlessBilling_Yes': [1 if paperless_billing == 'Yes' else 0],
                'PaymentMethod_Credit card (automatic)': [1 if payment_method == 'Credit card' else 0],
                'PaymentMethod_Electronic check': [1 if payment_method == 'Electronic check' else 0],
                'PaymentMethod_Mailed check': [1 if payment_method == 'Mailed check' else 0],
                'hasFamily_Yes': [1 if has_family == 'Yes' else 0],
                'SeniorCitizen': [senior_citizen],
                'AvgMonthlyCost': [float(monthly_cost)]
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
            
            result_color = '#FF6B6B' if prediction[0] == 'Yes' else '#4CAF50'
            probability_percentage = round(probability * 100, 2)
            
            return html.Div([
                html.H3(f"Prediction: {prediction[0]}", style={'color': result_color}),
                html.H4(f"Churn Probability: {probability_percentage}%"),
                html.Div([
                    dcc.Graph(
                        figure=go.Figure(data=[
                            go.Indicator(
                                mode="gauge+number",
                                value=probability_percentage,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Churn Risk"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': result_color},
                                    'steps': [
                                        {'range': [0, 30], 'color': 'rgba(0, 250, 0, 0.3)'},
                                        {'range': [30, 70], 'color': 'rgba(255, 255, 0, 0.3)'},
                                        {'range': [70, 100], 'color': 'rgba(250, 0, 0, 0.3)'}
                                    ]
                                }
                            )
                        ]),
                        config={'displayModeBar': False},
                        style={'height': '300px'}
                    )
                ])
            ])
            
        except Exception as e:
            return html.Div([
                html.H3("Error in prediction", style={'color': 'red'}),
                html.P(str(e))
            ])
    
    return html.Div()

# Callback for analytics page - confusion matrix
@app.callback(
    Output('confusion-matrix', 'figure'),
    [Input('url', 'pathname')]
)
def update_confusion_matrix(pathname):
    if pathname == '/analytics' and y_true is not None:
        cm = confusion_matrix(y_true, y_pred)
        
        # Create annotated heatmap
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['No Churn', 'Churn'],
            y=['No Churn', 'Churn'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=500,
            height=500
        )
        
        return fig
    
    return go.Figure()

# Callback for ROC curve
@app.callback(
    Output('roc-curve', 'figure'),
    [Input('url', 'pathname')]
)
def update_roc_curve(pathname):
    if pathname == '/analytics' and y_true is not None:
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(
            y_true.map({'Yes': 1, 'No': 0}), 
            probabilities
        )
        roc_auc = auc(fpr, tpr)
        
        # Create ROC curve figure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='darkorange', width=2)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Chance',
                line=dict(color='navy', width=2, dash='dash')
            )
        )
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=500, height=500,
            legend=dict(x=0.05, y=1.05, orientation='h')
        )
        
        return fig
    
    return go.Figure()

# Callback for classification report
@app.callback(
    Output('classification-report-output', 'children'),
    [Input('url', 'pathname')]
)
def update_classification_report(pathname):
    if pathname == '/analytics' and y_true is not None:
        report = classification_report(y_true, y_pred, output_dict=True)
        
        metrics = []
        for label in ['No', 'Yes']:
            if label in report:
                metrics.append(html.Div([
                    html.H4(f"Class: {label}"),
                    html.Ul([
                        html.Li(f"Precision: {report[label]['precision']:.2f}"),
                        html.Li(f"Recall: {report[label]['recall']:.2f}"),
                        html.Li(f"F1-Score: {report[label]['f1-score']:.2f}"),
                        html.Li(f"Support: {report[label]['support']}")
                    ])
                ]))
        
        metrics.append(html.Div([
            html.H4("Overall"),
            html.Ul([
                html.Li(f"Accuracy: {report['accuracy']:.2f}"),
                html.Li(f"Macro Avg F1: {report['macro avg']['f1-score']:.2f}"),
                html.Li(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.2f}")
            ])
        ]))
        
        return html.Div(metrics)
    
    return html.Div()

# Callback for feature importance
@app.callback(
    Output('feature-importance', 'figure'),
    [Input('url', 'pathname')]
)
def update_feature_importance(pathname):
    if pathname == '/analytics':
        # Get feature importance from the model if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # If not a tree-based model, create dummy importance values
            importances = np.ones(len(X_test.columns)) / len(X_test.columns)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Feature Importance',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600
        )
        
        return fig
    
    return go.Figure()

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Customer Churn Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f7fa;
                color: #333;
            }
            .app-header {
                color: #2c3e50;
                margin-bottom: 30px;
            }
            .form-container {
                background-color: white;
                padding: 25px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }
            .input-group {
                margin-bottom: 15px;
            }
            .input-group label {
                display: block;
                margin-bottom: 5px;
                font-weight: 500;
            }
            .prediction-result {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                display: inline-block;
                min-width: 300px;
            }
            .analytics-card {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .analytics-link, .back-link {
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 4px;
                transition: background-color 0.3s;
            }
            .analytics-link:hover, .back-link:hover {
                background-color: #2980b9;
            }
            button {
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)