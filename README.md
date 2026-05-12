# 🫀 FaceVitals – Camera-Based Heart & Respiratory Rate Monitor

## 📌 Overview

**FaceVitals** is an advanced **non-contact health monitoring system** that estimates:

* ❤️ **Heart Rate (HR)** – Beats Per Minute (BPM)
* 🌬️ **Respiratory Rate (RR)** – Breaths Per Minute
* 

using a standard camera by analyzing **temporal pixel variations in facial skin**.

> 💡 This system leverages **Computer Vision, Signal Processing, and Machine Learning** to monitor vital signs without any physical sensors.

---

## 🚀 Features

* 📷 Real-time face detection & forehead ROI extraction
* 🎨 RGB signal processing from facial skin
* 🧠 rPPG (Remote Photoplethysmography) for pulse extraction
* ⚙️ POS algorithm for signal enhancement & noise reduction
* 📈 FFT-based frequency estimation
* 🤖 Machine Learning-based HR refinement
* 📊 Dataset evaluation (Accuracy, MAE, R² Score)
* 📄 PDF report generation
* 🖥️ Interactive Streamlit UI
* 💊 Basic health insights based on results

---

## 🧠 Concepts Used

* **rPPG** – Extracts physiological signals from subtle skin color changes
* **POS Algorithm** – Reduces noise and motion artifacts
* **FFT** – Converts signal from time domain to frequency domain
* **Bandpass Filtering**

  * ❤️ HR: 0.8 – 2.5 Hz
  * 🌬️ RR: 0.1 – 0.5 Hz

---

## 🔄 Workflow

1. Capture video from webcam
2. Detect face and extract forehead region (ROI)
3. Collect RGB signals over time
4. Apply POS algorithm for signal extraction
5. Filter signals using bandpass filters
6. Apply FFT to detect dominant frequency
7. Convert frequency → HR & RR
8. Improve HR prediction using Machine Learning model

---

## ⚙️ Installation

### 🔹 Step 1: (Optional but Recommended) Create Virtual Environment

```bash id="venv1"
python -m venv venv
```

---

### 🔹 Step 2: Activate Virtual Environment

#### 🪟 Windows:

```bash id="venv2"
venv\Scripts\activate
```

#### 🐧 Linux / Mac:

```bash id="venv3"
source venv/bin/activate
```

---

### 🔹 Step 3: Install Dependencies

```bash id="venv4"
pip install streamlit opencv-python numpy scipy matplotlib scikit-learn joblib reportlab
```

---

### 📦 Libraries Used

* **streamlit** → User interface
* **opencv-python (cv2)** → Face detection & video capture
* **numpy** → Numerical operations
* **scipy** → Signal processing
* **matplotlib** → Visualization
* **scikit-learn** → Machine learning
* **joblib** → Model saving/loading
* **reportlab** → PDF generation

---

### 📁 Required Folders

* `models/` → Contains trained ML model files
* `Dataset/` → Used for evaluation

---

## ▶️ Usage

```bash
streamlit run app.py
```

---

# 🚀 Results & Outputs

## ❤️ Live Measurement Output

Heart Rate (HR - Heart Rate BPM):-79.50

Respiratory Rate (RR - Respiratory Rate):-22.71

Confidence Level: High

Signal Quality: Good Signal ✅

💊 Health Suggestion

Heart rate is normal → Keep maintaining a healthy lifestyle

Breathing rate is high → Take deep breaths and try to relax



## 📈 Temporal Signal Analysis

<img width="1384" height="746" alt="1632a7556ca0f0cda23d80e33d85bf55f0ea4b7ad53efff6f74fe357" src="https://github.com/user-attachments/assets/fd7709ac-ddcc-44cb-b550-65165e044194" />


---

## 📊 Dataset Results

<img width="880" height="490" alt="Screenshot 2026-03-29 215651" src="https://github.com/user-attachments/assets/3ca89a62-04c9-4592-8d2c-6ca6e514fea4" />

---

## 📉 Ground Truth vs Predicted

<img width="1104" height="869" alt="77c861f2b92a76c2d945eea77ed3b36a71b8553a50036562b8c2c0f9" src="https://github.com/user-attachments/assets/6e176116-0cfd-413c-b825-4abb5d62a00b" />


---

## 📈 Overall Performance

* ✅ **Average Accuracy:** 77.96
* 📉 **MAE:** 22.97 BPM
* 📈 **R² Score:** -2.294

---

## 🤖 Machine Learning Model

* Model: **Gradient Boosting Regressor**

* Features:

  * Mean
  * Standard Deviation
  * Dominant Frequency
  * Max / Min
  * Median
  * Percentiles
  * Variance

---

## ⚠️ Limitations

* Sensitive to lighting conditions
* Affected by facial movement
* Requires stable positioning
* Not a medical-grade device

---

## 🔮 Future Improvements

* Deep learning-based rPPG (CNN / LSTM)
* Mobile app integration
* Improved motion robustness
* Multi-face tracking

---

## 👨‍💻 Author

**K. Mohith Reddy**
Final Year – Computer Science & Engineering

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
