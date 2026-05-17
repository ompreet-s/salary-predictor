from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load('salary_predictor.pkl')

CATEGORICAL = ['job_title', 'education_level', 'industry', 'company_size', 'location', 'remote_work']
ALL_NUMERIC = ['experience_years', 'skills_count', 'certifications',
               'exp_x_skills', 'total_credential_score', 'is_premium_role',
               'is_remote', 'is_big_company', 'is_premium_location', 'is_higher_edu']

def engineer_features(df):
    df = df.copy()
    df['exp_x_skills']           = df['experience_years'] * df['skills_count']
    df['total_credential_score'] = df['experience_years'] + df['skills_count'] + df['certifications']
    df['is_premium_role']        = df['job_title'].isin(['AI Engineer','Data Scientist','ML Engineer','DevOps Engineer']).astype(int)
    df['is_remote']              = (df['remote_work'] == 'Yes').astype(int)
    df['is_big_company']         = df['company_size'].isin(['Large','Enterprise']).astype(int)
    df['is_premium_location']    = df['location'].isin(['USA','UK','Canada','Australia']).astype(int)
    df['is_higher_edu']          = df['education_level'].isin(['Master','PhD']).astype(int)
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'job_title':        request.form['job_title'],
            'experience_years': int(request.form['experience_years']),
            'education_level':  request.form['education_level'],
            'skills_count':     int(request.form['skills_count']),
            'industry':         request.form['industry'],
            'company_size':     request.form['company_size'],
            'location':         request.form['location'],
            'remote_work':      request.form['remote_work'],
            'certifications':   int(request.form['certifications']),
        }
        df = pd.DataFrame([data])
        df = engineer_features(df)
        prediction = model.predict(df[CATEGORICAL + ALL_NUMERIC])[0]
        result = f'Predicted Salary: ${prediction:,.0f} per year'
    except Exception as e:
        result = f'Error: {str(e)}'

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)