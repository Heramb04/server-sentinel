import gradio as gr
import joblib
import pandas as pd
import psutil
import numpy as np
import os

# --- 1. Load the Model ---
MODEL_FILE = "server_failure_model.pkl"
model = None

if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        print(f"Model loaded with joblib.")
    except Exception:
        import pickle
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded with pickle.")

# --- 2. Helper: Get Live System Data ---
def get_live_metrics():
    # CPU
    cpu = psutil.cpu_percent(interval=0.1)
    
    # RAM
    ram = psutil.virtual_memory().percent
    
    # TEMP (Try multiple common sensors for Linux/Bazzite)
    temp = 50.0 # Fallback
    try:
        temps = psutil.sensors_temperatures()
        if 'k10temp' in temps: # AMD Ryzen (Likely for your Victus)
            temp = temps['k10temp'][0].current
        elif 'amdgpu' in temps: # AMD GPU fallback
            temp = temps['amdgpu'][0].current
        elif 'coretemp' in temps: # Intel fallback
            temp = temps['coretemp'][0].current
        elif 'acpitz' in temps: # Generic ACPI fallback
            temp = temps['acpitz'][0].current
    except:
        pass # Keep fallback if sensors fail/permission denied

    # Rolling Averages (Simulated for live demo)
    # Since we can't easily store state in a simple function call without a database,
    # we'll assume "Sustained" is roughly equal to "Current" for the live snapshot.
    cpu_avg = cpu 
    
    return cpu, cpu_avg, ram, temp, 0.0 # Change rate 0 for snapshot

# --- 3. The Prediction Function ---
def predict(mode, s_cpu, s_cpu_avg, s_ram, s_temp, s_change):
    # Logic: If Live Mode is ON, overwrite inputs with real data
    if mode == "Live System Monitor":
        cpu, cpu_avg, ram, temp, change = get_live_metrics()
    else:
        # Use slider values directly
        cpu, cpu_avg, ram, temp, change = s_cpu, s_cpu_avg, s_ram, s_temp, s_change

    if model is None:
        return "Model Missing", "0%", cpu, cpu_avg, ram, temp, change

    # Prepare Data exactly as trained
    input_df = pd.DataFrame([{
        'cpu_percent': float(cpu),
        'ram_percent': float(ram),
        'cpu_temp': float(temp),
        'gpu_temp': float(temp) - 15.0, # Heuristic
        'net_recv_bytes': 1024.0,
        'disk_write_bytes': 0.0,
        'cpu_rolling_avg': float(cpu_avg),
        'ram_rolling_avg': float(ram),
        'cpu_temp_change': float(change)
    }])

    # Predict
    pred_class = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0][1]
    
    status = "CRITICAL FAILURE IMMINENT" if pred_class == 1 else "SYSTEM NORMAL"
    probability = f"{pred_prob * 100:.1f}%"
    
    # Return Status, Prob, AND the updated slider values (to animate them)
    return status, probability, cpu, cpu_avg, ram, temp, change

# --- 4. The Gradio UI Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖥️ Server Health Sentinel AI (Hybrid Edition)")
    gr.Markdown("### AIOps Failure Prediction System (PoC)")
    
    # The Mode Toggle
    mode_switch = gr.Radio(["Simulation Mode", "Live System Monitor"], 
                           value="Simulation Mode", 
                           label="Operation Mode")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎛️ Telemetry Inputs")
            
            s_cpu = gr.Slider(0, 100, value=10, label="Current CPU Load (%)")
            s_cpu_avg = gr.Slider(0, 100, value=10, label="Sustained CPU Load (Last 1 min) (%)")
            s_ram = gr.Slider(0, 100, value=30, label="RAM Usage (%)")
            s_temp = gr.Slider(30, 100, value=50, label="Current Temperature (°C)")
            s_change = gr.Slider(-2, 5, value=0, step=0.5, label="Temp Change Rate (°C/sec)")
            
            btn = gr.Button("Run Analysis", variant="primary")

        with gr.Column():
            gr.Markdown("### 🧠 AI Diagnosis")
            out_status = gr.Textbox(label="Status")
            out_prob = gr.Textbox(label="Failure Probability")
            
            gr.Markdown("""
            **Architecture:** Random Forest Classifier
            **Trained on:** 10,000+ Real-world Linux Telemetry Points
            **Live Mode:** Uses `psutil` to fetch real-time hardware stats.
            """)

    # Connect the button
    # Note: We output back to the sliders (s_cpu, etc.) to update them visually in Live Mode!
    btn.click(fn=predict, 
              inputs=[mode_switch, s_cpu, s_cpu_avg, s_ram, s_temp, s_change], 
              outputs=[out_status, out_prob, s_cpu, s_cpu_avg, s_ram, s_temp, s_change])

# Launch
demo.launch()