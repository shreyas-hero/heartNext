import requests
import json

# API endpoint
BASE_URL = "http://localhost:5000"

def test_prediction():
    """Test the prediction endpoint"""
    # Sample patient data
    patient_data = {
        "age": 63,
        "sex": 1,  # male
        "cp": 1,   # atypical angina
        "trestbps": 145,  # resting blood pressure
        "chol": 233,      # cholesterol
        "fbs": 1,         # fasting blood sugar > 120
        "restecg": 2,     # resting ECG
        "thalach": 150,   # max heart rate
        "exang": 0,       # exercise induced angina
        "oldpeak": 2.3,   # ST depression
        "slope": 3,       # slope of peak exercise ST segment
        "ca": 0,          # number of major vessels
        "thal": 6         # thalassemia
    }
    
    try:
        # Make prediction request
        response = requests.post(f"{BASE_URL}/predict", json=patient_data)
        
        if response.status_code == 200:
            result = response.json()
            print("Prediction Result:")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Probabilities: Low Risk = {result['probability']['low_risk']:.2f}, High Risk = {result['probability']['high_risk']:.2f}")
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the Flask server is running.")
    except Exception as e:
        print(f"Error: {e}")

def test_model_info():
    """Test the model info endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        
        if response.status_code == 200:
            info = response.json()
            print("\nModel Information:")
            print(f"Model Type: {info['model_type']}")
            print("\nFeature Descriptions:")
            for feature, description in info['feature_descriptions'].items():
                print(f"- {feature}: {description}")
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the Flask server is running.")
    except Exception as e:
        print(f"Error: {e}")

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            health = response.json()
            print(f"\nHealth Check: {health['status']}")
            print(f"Model Loaded: {health['model_loaded']}")
            print(f"Scaler Loaded: {health['scaler_loaded']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the Flask server is running.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing Heart Disease Prediction API")
    print("=" * 50)
    
    # Test health check
    test_health_check()
    
    # Test model info
    test_model_info()
    
    # Test prediction
    test_prediction()
    
    print("\nTesting completed!")