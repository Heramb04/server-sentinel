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
    except Exception:
        import pickle
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)

# --- 2. Helper: Get Live System Data ---
def get_live_metrics():
    cpu = psutil.cpu_percent(interval=None) # Non-blocking for timer
    ram = psutil.virtual_memory().percent
    
    # Sensor fallback logic
    temp = 50.0 
    try:
        temps = psutil.sensors_temperatures()
        if 'k10temp' in temps: temp = temps['k10temp'][0].current
        elif 'amdgpu' in temps: temp = temps['amdgpu'][0].current
        elif 'coretemp' in temps: temp = temps['coretemp'][0].current
        elif 'acpitz' in temps: temp = temps['acpitz'][0].current
    except:
        pass

    return cpu, cpu, ram, temp, 0.0 # cpu_avg = cpu for snapshot

# --- 3. UI Logic Functions ---

def toggle_ui_mode(mode):
    """
    Hides/Shows panels AND enables/disables the timer based on mode.
    """
    if mode == "Live System Monitor":
        # Show Live Panel, Hide Sim Panel, TURN ON TIMER (active=True)
        return {
            sim_panel: gr.update(visible=False), 
            live_panel: gr.update(visible=True),
            timer: gr.update(active=True)
        }
    else:
        # Show Sim Panel, Hide Live Panel, TURN OFF TIMER (active=False)
        return {
            sim_panel: gr.update(visible=True), 
            live_panel: gr.update(visible=False),
            timer: gr.update(active=False)
        }

def predict_logic(mode, s_cpu, s_cpu_avg, s_ram, s_temp, s_change):
    """
    Performs prediction and returns values to populate the correct panel.
    """
    # 1. Get Data Source
    if mode == "Live System Monitor":
        cpu, cpu_avg, ram, temp, change = get_live_metrics()
    else:
        cpu, cpu_avg, ram, temp, change = s_cpu, s_cpu_avg, s_ram, s_temp, s_change

    # 2. Check Model
    status_msg = "Model Missing"
    prob_str = "0%"
    
    if model:
        # Prepare Data
        input_df = pd.DataFrame([{
            'cpu_percent': float(cpu),
            'ram_percent': float(ram),
            'cpu_temp': float(temp),
            'gpu_temp': float(temp) - 15.0,
            'net_recv_bytes': 1024.0,
            'disk_write_bytes': 0.0,
            'cpu_rolling_avg': float(cpu_avg),
            'ram_rolling_avg': float(ram),
            'cpu_temp_change': float(change)
        }])
        
        # Predict Probabilities directly
        try:
            prob_val = model.predict_proba(input_df)[0][1]
            
            # LOGIC UPDATE: Use tiers instead of simple 50% threshold
            if prob_val >= 0.80:
                status_msg = "CRITICAL FAILURE IMMINENT"
            elif prob_val >= 0.50:
                status_msg = "WARNING: ELEVATED RISK"
            else:
                status_msg = "SYSTEM NORMAL"
                
            prob_str = f"{prob_val * 100:.1f}%"
        except Exception as e:
            status_msg = f"Error: {str(e)}"

    # 3. Return everything 
    return status_msg, prob_str, cpu, cpu_avg, ram, temp, change

# --- 4. The Gradio Blocks UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖥️ Server Health Sentinel (Hybrid Edition)")
    
    # Top Control: Mode Switch
    mode_switch = gr.Radio(
        ["Simulation Mode", "Live System Monitor"], 
        value="Simulation Mode", 
        label="Operation Mode",
        info="Select 'Live' to read local hardware sensors."
    )
    
    # Timer component (Initially inactive)
    timer = gr.Timer(2.0, active=False)
    
    # --- PANEL A: SIMULATION (Sliders) ---
    with gr.Group(visible=True) as sim_panel:
        gr.Markdown("### 🎛️ Manual Simulation Controls")
        with gr.Row():
            s_cpu = gr.Slider(0, 100, value=10, label="Current CPU Load (%)")
            s_cpu_avg = gr.Slider(0, 100, value=10, label="Sustained CPU Load (%)")
            s_ram = gr.Slider(0, 100, value=30, label="RAM Usage (%)")
        with gr.Row():
            s_temp = gr.Slider(30, 100, value=50, label="Current Temperature (°C)")
            s_change = gr.Slider(-2, 5, value=0, step=0.5, label="Temp Change Rate")

    # --- PANEL B: LIVE MONITOR (Read-Only Displays) ---
    with gr.Group(visible=False) as live_panel:
        gr.Markdown("### 📡 Live Sensor Readings (Localhost)")
        with gr.Row():
            l_cpu = gr.Number(label="Live CPU Load", precision=1)
            l_cpu_avg = gr.Number(label="Live Sustained CPU", precision=1)
            l_ram = gr.Number(label="Live RAM", precision=1)
        with gr.Row():
            l_temp = gr.Number(label="Live Temp (°C)", precision=1)
            l_change = gr.Number(label="Temp Change", value=0, precision=1)

    # --- OUTPUTS ---
    gr.Markdown("### 🧠 AI Diagnosis")
    with gr.Row():
        out_status = gr.Textbox(label="Status")
        out_prob = gr.Textbox(label="Failure Probability")

    btn = gr.Button("Analyze System Status", variant="primary")

    # --- EVENTS ---
    
    # 1. Mode Switch Logic: Toggles panels AND Timer
    mode_switch.change(
        fn=toggle_ui_mode, 
        inputs=mode_switch, 
        outputs=[sim_panel, live_panel, timer]
    )

    # 2. Manual Button Click
    btn.click(
        fn=predict_logic, 
        inputs=[mode_switch, s_cpu, s_cpu_avg, s_ram, s_temp, s_change],
        outputs=[out_status, out_prob, l_cpu, l_cpu_avg, l_ram, l_temp, l_change]
    )
    
    # 3. Timer Tick (Auto-Update)
    # Triggers the exact same logic as the button, automatically
    timer.tick(
        fn=predict_logic,
        inputs=[mode_switch, s_cpu, s_cpu_avg, s_ram, s_temp, s_change],
        outputs=[out_status, out_prob, l_cpu, l_cpu_avg, l_ram, l_temp, l_change]
    )

demo.launch()