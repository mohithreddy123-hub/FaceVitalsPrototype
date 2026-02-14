# 🫀 Face Vitals Prototype  
Camera-Based Heart Rate & Respiratory Rate Estimation

---

## 📌 Project Description

Face Vitals Prototype is a non-contact vital sign monitoring system that estimates:

- **Heart Rate (BPM)**
- **Respiratory Rate (Breaths per Minute)**

using only a standard webcam.

The system detects subtle color changes in the forehead caused by blood circulation (remote Photoplethysmography - rPPG). These temporal pixel variations are processed using signal processing techniques and machine learning to estimate vital signs.

---

## 🚀 Features

- Face detection using OpenCV Haar Cascade
- Forehead region extraction
- RGB signal collection from webcam frames
- POS (Plane-Orthogonal-to-Skin) signal processing
- Bandpass filtering for heart and respiratory bands
- FFT-based frequency estimation
- Sliding window analysis
- Random Forest ML-based heart rate prediction
- Streamlit web interface
- Signal visualization using Matplotlib

---

## 🧠 How It Works

1. Webcam captures video frames.
2. Face is detected using Haar Cascade.
3. Forehead region is extracted.
4. Average RGB values are collected over time.
5. POS algorithm enhances pulse-related signal.
6. Bandpass filter isolates:
   - Heart rate band (0.8–2.5 Hz)
   - Respiratory band (0.1–0.5 Hz)
7. FFT identifies dominant frequency.
8. Frequency is converted to BPM.
9. Optional ML model refines heart rate prediction.

---

## 📂 Project Structure
FaceVitals/
│
├── app.py # OpenCV real-time implementation
├── streamlit_app.py # Streamlit web application
├── train_model.py # ML training script
├── models/
│ └── heart_rate_model.pkl # Trained ML model
├── haarcascade_frontalface_default.xml
└── README.md

---

## 🛠 Requirements

Install required libraries:

```bash
pip install opencv-python numpy scipy matplotlib scikit-learn streamlit joblib
▶ How to Run
OpenCV Version
python app.py


Stay still during 20-second recording.

Streamlit Version
streamlit run streamlit_app.py


Click Start Measurement and remain steady.

📊 Output

The system displays:

FFT Heart Rate (BPM)

ML Heart Rate (if model available)

Respiratory Rate (Breaths/min)

POS signal graph

Filtered heart & respiratory signals

⚠ Limitations

Sensitive to motion

Sensitive to lighting conditions

Not medical-grade accuracy

Requires stable webcam positioning

🚀 Future Improvements

Deep learning based rPPG

Better motion compensation

Real-time mobile application

Cloud-based health monitoring dashboard

👨‍💻 Author

Mohith Reddy
B.Tech Major Project
Computer Vision & Biomedical Signal Processing
