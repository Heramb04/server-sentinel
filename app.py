from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
MODEL_FILE = "server_failure_model.pkl"
model = None

def load_model():
    global model
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        print(f"Model loaded from {MODEL_FILE}")
    else:
        print(f"CRITICAL ERROR: {MODEL_FILE} not found!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model: return jsonify({'error': 'Model not loaded'}), 500
    try:
        data = request.get_json()

        # --- 1. EXACT MAPPING TO TRAINING DATA ---
        # The model was trained on these 9 specific features in this exact order.
        # We must provide all of them.

        # Feature 1: cpu_percent (From Slider 1)
        cpu = float(data.get('cpu_percent', 10))

        # Feature 2: ram_percent (From NEW SLIDER!)
        ram = float(data.get('ram_percent', 30))

        # Feature 3: cpu_temp (From Slider 4)
        cpu_temp = float(data.get('cpu_temp', 50))

        # Feature 4: gpu_temp (Inferred from CPU temp for simplicity, or add a slider)
        # During gaming, GPU temp was usually ~15C cooler than CPU temp in your data.
        gpu_temp = cpu_temp - 15

        # Feature 5 & 6: Network/Disk (Background noise, set to low values)
        net_recv = 1024.0
        disk_write = 0.0

        # Feature 7: cpu_rolling_avg (From Slider 2 - CRITICAL NEW FEATURE)
        # If user didn't use the new slider yet, default to current 'cpu'
        cpu_rolling = float(data.get('cpu_rolling_avg', cpu))

        # Feature 8: ram_rolling_avg (Simplified to equal current RAM)
        ram_rolling = ram

        # Feature 9: cpu_temp_change (From Slider 5)
        cpu_temp_change = float(data.get('cpu_temp_change', 0.0))

        # --- 2. BUILD THE DATAFRAME ---
        # Must match training columns EXACTLY:
        # ['cpu_percent', 'ram_percent', 'cpu_temp', 'gpu_temp',
        #  'net_recv_bytes', 'disk_write_bytes',
        #  'cpu_rolling_avg', 'ram_rolling_avg', 'cpu_temp_change']
        input_df = pd.DataFrame([{
            'cpu_percent': cpu,
            'ram_percent': ram,
            'cpu_temp': cpu_temp,
            'gpu_temp': gpu_temp,
            'net_recv_bytes': net_recv,
            'disk_write_bytes': disk_write,
            'cpu_rolling_avg': cpu_rolling,
            'ram_rolling_avg': ram_rolling,
            'cpu_temp_change': cpu_temp_change
        }])

        # --- 3. PREDICT ---
        prediction = model.predict(input_df)[0]
        # Get probability of class 1 (CRITICAL)
        probability_critical = model.predict_proba(input_df)[0][1]

        return jsonify({
            'status': 'CRITICAL' if prediction == 1 else 'NORMAL',
            'probability': round(probability_critical * 100, 1)
        })
    except Exception as e: return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
