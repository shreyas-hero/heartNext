from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

app = Flask(__name__)
CORS(app)

# Global variables for model and scaler
model = None
scaler = None

def create_synthetic_data():
    """Create synthetic heart disease dataset for training"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features
    age = np.random.normal(50, 15, n_samples)
    sex = np.random.choice([0, 1], n_samples)  # 0: female, 1: male
    cp = np.random.choice([0, 1, 2, 3], n_samples)  # chest pain type
    trestbps = np.random.normal(130, 20, n_samples)  # resting blood pressure
    chol = np.random.normal(240, 50, n_samples)  # cholesterol
    fbs = np.random.choice([0, 1], n_samples)  # fasting blood sugar
    restecg = np.random.choice([0, 1, 2], n_samples)  # resting ECG
    thalach = np.random.normal(150, 25, n_samples)  # max heart rate
    exang = np.random.choice([0, 1], n_samples)  # exercise induced angina
    oldpeak = np.random.exponential(1, n_samples)  # ST depression
    slope = np.random.choice([0, 1, 2], n_samples)  # slope of peak exercise
    ca = np.random.choice([0, 1, 2, 3], n_samples)  # number of major vessels
    thal = np.random.choice([0, 1, 2, 3], n_samples)  # thalassemia
    
    # Create realistic correlations for target variable
    risk_score = (
        (age > 55) * 0.3 +
        (sex == 1) * 0.2 +
        (cp == 0) * 0.4 +
        (trestbps > 140) * 0.2 +
        (chol > 240) * 0.2 +
        (fbs == 1) * 0.1 +
        (thalach < 120) * 0.3 +
        (exang == 1) * 0.3 +
        (oldpeak > 2) * 0.2 +
        (ca > 0) * 0.2 +
        (thal > 1) * 0.2
    )
    
    # Add some randomness and create binary target
    risk_score += np.random.normal(0, 0.2, n_samples)
    target = (risk_score > 0.8).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'target': target
    })
    
    return data

def train_model():
    """Train the heart disease prediction model"""
    global model, scaler
    
    # Create synthetic data
    data = create_synthetic_data()
    
    # Separate features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained with accuracy: {accuracy:.2f}")
    
    # Save model and scaler
    joblib.dump(model, 'heart_disease_model.pkl')
    joblib.dump(scaler, 'heart_disease_scaler.pkl')
    
    return model, scaler

def load_model():
    """Load trained model and scaler"""
    global model, scaler
    
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('heart_disease_scaler.pkl')
        print("Model and scaler loaded successfully")
    except FileNotFoundError:
        print("Model files not found. Training new model...")
        model, scaler = train_model()
    
    return model, scaler

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        "message": "Heart Disease Prediction API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Predict heart disease risk",
            "/retrain": "POST - Retrain the model",
            "/model_info": "GET - Get model information"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict heart disease risk"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Required features
        required_features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        # Check if all required features are present
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({
                "error": f"Missing required features: {missing_features}"
            }), 400
        
        # Create feature array
        features = np.array([[
            data['age'], data['sex'], data['cp'], data['trestbps'],
            data['chol'], data['fbs'], data['restecg'], data['thalach'],
            data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Prepare response
        result = {
            "prediction": int(prediction),
            "risk_level": "High Risk" if prediction == 1 else "Low Risk",
            "probability": {
                "low_risk": float(probability[0]),
                "high_risk": float(probability[1])
            },
            "confidence": float(max(probability))
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        feature_descriptions = {
            'age': 'Age in years',
            'sex': 'Sex (0: female, 1: male)',
            'cp': 'Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)',
            'trestbps': 'Resting blood pressure (mm Hg)',
            'chol': 'Serum cholesterol (mg/dl)',
            'fbs': 'Fasting blood sugar > 120 mg/dl (0: false, 1: true)',
            'restecg': 'Resting ECG results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (0: no, 1: yes)',
            'oldpeak': 'ST depression induced by exercise relative to rest',
            'slope': 'Slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)',
            'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
            'thal': 'Thalassemia (0: normal, 1: fixed defect, 2: reversible defect, 3: not described)'
        }
        
        return jsonify({
            "model_type": "Random Forest Classifier",
            "features": feature_names,
            "feature_descriptions": feature_descriptions,
            "target": "Heart disease presence (0: no disease, 1: disease present)"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model"""
    try:
        global model, scaler
        model, scaler = train_model()
        return jsonify({"message": "Model retrained successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })

if __name__ == '__main__':
    # Load or train model on startup
    load_model()
    
    print("Starting Heart Disease Prediction API...")
    print("Available endpoints:")
    print("- GET  /: API information")
    print("- POST /predict: Predict heart disease risk")
    print("- GET  /model_info: Get model information")
    print("- POST /retrain: Retrain the model")
    print("- GET  /health: Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)