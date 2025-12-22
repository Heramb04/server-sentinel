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
    cpu = psutil.cpu_percent(interval=None) 
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

    return cpu, cpu, ram, temp, 0.0

# --- 3. UI Logic Functions ---

def toggle_ui_mode(mode):
    """
    Switching modes resets the interface.
    """
    if mode == "Live System Monitor":
        return {
            sim_panel: gr.update(visible=False), 
            live_panel: gr.update(visible=True),
            btn: gr.update(value="Start Live Monitoring", variant="primary"),
            timer: gr.update(active=False) # Stop timer when switching modes
        }
    else:
        return {
            sim_panel: gr.update(visible=True), 
            live_panel: gr.update(visible=False),
            btn: gr.update(value="Analyze Simulation", variant="primary"),
            timer: gr.update(active=False)
        }

def handle_button_click(mode, is_running, s_cpu, s_cpu_avg, s_ram, s_temp, s_change):
    """
    Handles the Start/Stop logic for Live mode, or single-shot for Sim mode.
    """
    if mode == "Simulation Mode":
        # Just run once
        return predict_logic(mode, s_cpu, s_cpu_avg, s_ram, s_temp, s_change) + (False, gr.update(value="Analyze Simulation"))
    
    else:
        # Live Mode: Toggle Start/Stop
        if not is_running:
            # User clicked Start -> Turn ON
            return predict_logic(mode, s_cpu, s_cpu_avg, s_ram, s_temp, s_change) + (True, gr.update(value="Stop Live Monitoring", variant="stop"))
        else:
            # User clicked Stop -> Turn OFF
            # We return existing values but stop the timer
            # We need to return *something* for all outputs, so we re-return current slider vals
            # status, prob, cpu, cpu_avg, ram, temp, change, is_running, btn_update, timer_update
            return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), False, gr.update(value="Start Live Monitoring", variant="primary"), gr.update(active=False))

def predict_logic_wrapper(mode, is_running, s_cpu, s_cpu_avg, s_ram, s_temp, s_change):
    """
    Wrapper to handle the logic flow including timer updates.
    """
    # If we are in Live Mode and Running, we turn the timer ON
    # If Sim mode, timer OFF
    results = handle_button_click(mode, is_running, s_cpu, s_cpu_avg, s_ram, s_temp, s_change)
    
    # results format from handle_button_click:
    # (status, prob, cpu, cpu_avg, ram, temp, change, NEW_is_running, btn_update, timer_update_OPTIONAL)
    
    # We need to explicitly handle the timer output based on the new 'is_running' state
    # The handle_button_click logic is a bit complex to map directly to outputs 
    # because 'predict_logic' returns 7 items.
    
    # Let's simplify: 
    # This wrapper is called by the BUTTON.
    
    if mode == "Simulation Mode":
        # Run logic once. Timer is always False. State is False.
        status, prob, c, ca, r, t, ch = predict_logic(mode, s_cpu, s_cpu_avg, s_ram, s_temp, s_change)
        return status, prob, c, ca, r, t, ch, False, gr.update(value="Analyze Simulation"), gr.update(active=False)
    
    else: # Live Mode
        if is_running:
             # Clicked Stop.
             return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), False, gr.update(value="Start Live Monitoring", variant="primary"), gr.update(active=False)
        else:
             # Clicked Start. Run immediate prediction, set State True, Timer Active.
             status, prob, c, ca, r, t, ch = predict_logic(mode, s_cpu, s_cpu_avg, s_ram, s_temp, s_change)
             return status, prob, c, ca, r, t, ch, True, gr.update(value="Stop Live Monitoring", variant="stop"), gr.update(active=True)

def timer_tick(mode, is_running):
    """
    Called by timer. Only runs if Live Mode AND is_running.
    """
    if mode == "Live System Monitor" and is_running:
        status, prob, c, ca, r, t, ch = predict_logic(mode, 0,0,0,0,0) # Inputs ignored in live mode
        return status, prob, c, ca, r, t, ch
    else:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()


def predict_logic(mode, s_cpu, s_cpu_avg, s_ram, s_temp, s_change):
    """
    Core prediction logic. Returns values for the 7 UI output fields.
    """
    # 1. Get Data
    if mode == "Live System Monitor":
        cpu, cpu_avg, ram, temp, change = get_live_metrics()
    else:
        cpu, cpu_avg, ram, temp, change = s_cpu, s_cpu_avg, s_ram, s_temp, s_change

    # 2. Check Model
    status_html = "<h2 style='color:grey'>Model Missing</h2>"
    prob_str = "0%"
    
    if model:
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
        
        try:
            prob_val = model.predict_proba(input_df)[0][1]
            
            # --- STYLING LOGIC ---
            if prob_val >= 0.80:
                color = "#ff0000" # Red
                text = "CRITICAL FAILURE IMMINENT"
                icon = "🔥"
            elif prob_val >= 0.50:
                color = "#ffaa00" # Orange
                text = "WARNING: ELEVATED RISK"
                icon = "⚠️"
            else:
                color = "#00cc00" # Green
                text = "SYSTEM NORMAL"
                icon = "✅"
                
            # Bold, expressive HTML output
            status_html = f"""
            <div style='text-align: center; padding: 10px; border: 2px solid {color}; border-radius: 10px; background-color: {color}20;'>
                <h1 style='color: {color}; margin: 0; font-size: 24px;'>{icon} {text}</h1>
            </div>
            """
            prob_str = f"{prob_val * 100:.1f}%"
            
        except Exception as e:
            status_html = f"<div style='color:red'>Error: {str(e)}</div>"

    return status_html, prob_str, cpu, cpu_avg, ram, temp, change

# --- 4. The Gradio Blocks UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖥️ Server Health Sentinel (Hybrid Edition)")
    
    # State variable to track if Live Monitoring is ON/OFF
    is_running = gr.State(False)
    
    mode_switch = gr.Radio(
        ["Simulation Mode", "Live System Monitor"], 
        value="Simulation Mode", 
        label="Operation Mode",
        info="Select 'Live' to read local hardware sensors."
    )
    
    timer = gr.Timer(2.0, active=False)
    
    with gr.Group(visible=True) as sim_panel:
        gr.Markdown("### 🎛️ Manual Simulation Controls")
        with gr.Row():
            s_cpu = gr.Slider(0, 100, value=10, label="Current CPU Load (%)")
            s_cpu_avg = gr.Slider(0, 100, value=10, label="Sustained CPU Load (%)")
            s_ram = gr.Slider(0, 100, value=30, label="RAM Usage (%)")
        with gr.Row():
            s_temp = gr.Slider(30, 100, value=50, label="Current Temperature (°C)")
            s_change = gr.Slider(-2, 5, value=0, step=0.5, label="Temp Change Rate")

    with gr.Group(visible=False) as live_panel:
        gr.Markdown("### 📡 Live Sensor Readings (Localhost)")
        with gr.Row():
            l_cpu = gr.Number(label="Live CPU Load", precision=1)
            l_cpu_avg = gr.Number(label="Live Sustained CPU", precision=1)
            l_ram = gr.Number(label="Live RAM", precision=1)
        with gr.Row():
            l_temp = gr.Number(label="Live Temp (°C)", precision=1)
            l_change = gr.Number(label="Temp Change", value=0, precision=1)

    gr.Markdown("### 🧠 AI Diagnosis")
    with gr.Row():
        # Changed to HTML component for styling
        out_status = gr.HTML(label="Status")
        out_prob = gr.Textbox(label="Failure Probability")

    btn = gr.Button("Analyze Simulation", variant="primary")

    # --- EVENTS ---
    
    # 1. Mode Switch: Reset everything
    mode_switch.change(
        fn=toggle_ui_mode, 
        inputs=mode_switch, 
        outputs=[sim_panel, live_panel, btn, timer]
    )

    # 2. Button Click: Handles Start/Stop logic
    btn.click(
        fn=predict_logic_wrapper, 
        inputs=[mode_switch, is_running, s_cpu, s_cpu_avg, s_ram, s_temp, s_change],
        outputs=[out_status, out_prob, l_cpu, l_cpu_avg, l_ram, l_temp, l_change, is_running, btn, timer]
    )
    
    # 3. Timer Tick: Auto-update ONLY if running
    timer.tick(
        fn=timer_tick,
        inputs=[mode_switch, is_running],
        outputs=[out_status, out_prob, l_cpu, l_cpu_avg, l_ram, l_temp, l_change]
    )

demo.launch()