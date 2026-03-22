# 🫀 FaceVitals – Camera-Based Heart & Respiratory Rate Monitor

## 📌 Overview

**FaceVitals** is a non-contact health monitoring system that estimates:

* ❤️ **Heart Rate (HR)** – Beats Per Minute (BPM)
* 🌬️ **Respiratory Rate (RR)** – Breaths Per Minute

using a standard camera by analyzing **temporal pixel variations** in facial skin.

---

## 🚀 Features

* 📷 Real-time face detection & forehead ROI extraction
* 🎨 RGB signal processing
* 🧠 rPPG (Remote Photoplethysmography)
* ⚙️ POS algorithm for signal enhancement
* 📈 FFT-based frequency estimation
* 🤖 Machine Learning-based HR refinement
* 📊 Dataset evaluation (Accuracy, MAE, R²)
* 📄 PDF report generation
* 🖥️ Streamlit UI

---

## 🧠 Concepts Used

* **rPPG** – Extracts pulse from skin color changes
* **POS Algorithm** – Reduces noise and motion artifacts
* **FFT** – Converts signal to frequency domain
* **Bandpass Filtering**

  * HR: 0.8 – 2.5 Hz
  * RR: 0.1 – 0.5 Hz

---

## 🔄 Workflow

1. Capture video from webcam
2. Detect face and extract forehead region
3. Collect RGB signals over time
4. Apply POS algorithm
5. Filter signals using bandpass
6. Apply FFT to detect dominant frequency
7. Convert frequency → HR & RR
8. Improve HR using ML model

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

* **streamlit** → UI
* **opencv-python (cv2)** → Face detection & video capture
* **numpy** → Numerical operations
* **scipy** → Signal processing
* **matplotlib** → Visualization
* **scikit-learn** → Machine learning
* **joblib** → Model saving/loading
* **reportlab** → PDF generation

---

### 📁 Required Folders

* `models/` → contains trained model files
* `Dataset/` → used for evaluation

---


---

## ▶️ Usage

```bash
streamlit run app.py
```

---

# 🚀 Results & Outputs

## ❤️ Live Measurement Output

* **Heart Rate (HR):** 56.09 BPM
* **Respiratory Rate (RR):** 10.20 breaths/min
* **Confidence Level:** High
* **Signal Quality:** Good Signal ✅

---

## 📈 Temporal Signal Analysis


<img width="1384" height="746" alt="ab9b0548d58fee24eb5e373f14b04acd100dff45e0c88c017eea74a8" src="https://github.com/user-attachments/assets/42197d72-f9ae-45dd-b0d5-cb3e54a4c339" />

---

## 📊 Dataset Results
<img width="1330" height="401" alt="Screenshot 2026-03-22 164852" src="https://github.com/user-attachments/assets/409af278-a0c9-4194-a8aa-6433ef2c21db" />




---

## 📉 Ground Truth vs Predicted

<img width="1104" height="869" alt="46171056b268c6fb0cad88d72cfa78448e57e6d79ad87c8d7ae74590" src="https://github.com/user-attachments/assets/baa57deb-6440-41fb-8e08-471f333e3274" />


---

## 📈 Overall Performance

* ✅ **Average Accuracy:** 94.66%
* 📉 **MAE:** 5.84 BPM
* 📈 **R² Score:** -0.642

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

* Sensitive to lighting
* Affected by movement
* Requires stable positioning
* Not a medical-grade device

---

## 🔮 Future Improvements

* Deep learning-based rPPG
* Mobile app integration
* Motion robustness
* Multi-face tracking

---

## 👨‍💻 Author

**K. Mohith Reddy**
Final Year Computer science & engineering

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---
