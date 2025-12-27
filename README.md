### Server Health Sentinel AI Predicting Thermal Runaway Before It Happens

This project is a Proof-of-Concept (PoC) for AIOps (Artificial Intelligence for IT Operations). It demonstrates how machine learning can move beyond simple "threshold-based" monitoring to predictive failure analysis.

## Try the Demo: Adjust the sliders in the Live Telemetry Simulation panel to see how the model reacts to different stress scenarios.

Scenario A (Idle): Low CPU, Low Temp → System Normal

Scenario B (Gaming/Load): High Sustained CPU, High Temp → CRITICAL FAILURE IMMINENT

Scenario C (Cool Down): Low Current CPU but High Sustained Load + High Temp → CRITICAL (Predicting residual heat)

## Prerequisites & Setup To run this project locally with full Live Monitoring capabilities:

1. System RequirementsOS: Linux (Recommended), macOS, or Windows.Note: Live hardware sensors (psutil) work best on Linux/Bazzite.Python: Version 3.9 or higher.

2. Installation: 
Clone the repository and install the required Python libraries: git clone https://github.com/Heramb04/server-sentinel.git

cd server-sentinel

pip install -r requirements.txt

4. How to Run/Launch the application: 

python app.py

The app will open in your browser at http://127.0.0.1:7860.

Select "Live System Monitor" to see your real-time hardware health.


## The Model: 
Random Forest Classifier. Unlike simple if/else logic, this system uses a Random Forest Classifier (an ensemble of 100 decision trees) to weigh multiple factors simultaneously.It was trained on a custom dataset of 10,000+ telemetry points collected from my own system under various real-world conditions:Idle/Web Browsing (Baseline), Compilation/Workloads (CPU Spikes), Gaming (Sustained CPU+GPU Thermal Stress)

### Feature Engineering:
The model doesn't just look at current stats. It relies on engineered trend features to understand context:

Rolling Averages: A 1-minute sustained load is more dangerous than a 1-second spike.

Thermal Inertia: Combining current temp with recent load history to predict "heat soak".

Rate of Change: How fast is the temperature climbing?

## Performance:
AUC Score: ~0.99 (Highly accurate on test set)False Positive Rate: <0.5%False Negative Rate: <1.0% 

## Tech Stack: 
Training: Scikit-Learn, Pandas, Psutil

Deployment: Gradio, Hugging Face Spaces

Hardware Target: x86_64 Linux Systems
