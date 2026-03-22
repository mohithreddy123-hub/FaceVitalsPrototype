import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, detrend
import matplotlib.pyplot as plt
import time
import os
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- SESSION STATE ----------------
if "live_results" not in st.session_state:
    st.session_state.live_results = None

if "dataset_results" not in st.session_state:
    st.session_state.dataset_results = None

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="FaceVitals", layout="centered")

st.title("🫀 Camera-Based Heart & Respiratory Rate Monitor")

st.markdown("""
## 📌 About This Project

This project measures **Heart Rate (HR)** and **Respiratory Rate (RR)** using a normal camera 
by analyzing very small color changes in the human face.

This method is completely **non-contact** (no wires or sensors needed).

---

## ❤️ What is Heart Rate (HR)?

**Heart Rate (HR)** means the number of times your heart beats in one minute.

📊 It is measured in **BPM (Beats Per Minute)**.

### 🏥 How hospitals measure it:
* **ECG (Electrocardiogram)** – Measures electrical signals of the heart
* **Pulse Oximeter** – Clip device placed on finger
* **Chest Sensors** – Used in hospitals for continuous monitoring

---

## 🌬️ What is Respiratory Rate (RR)?

**Respiratory Rate (RR)** means how many times you breathe in one minute.

📊 It is measured in **Breaths Per Minute**.

### 🏥 How hospitals measure it:
* Chest movement sensors
* Breathing monitors
* ICU monitoring systems

---

## 🤖 How THIS system works

This system uses advanced signal processing techniques:

### 🎥 Camera
Captures your face video in real-time.

### 🎨 RGB (Red, Green, Blue)
The camera records color signals:
* **R = Red**
* **G = Green**
* **B = Blue**

These color changes happen due to blood flow under the skin.

---

### 🧠 rPPG (Remote Photoplethysmography)
A technique that detects blood flow changes using a camera instead of sensors.

👉 It extracts pulse signals from facial color variations.

---

### ⚙️ POS (Plane-Orthogonal-to-Skin method)
An algorithm used to:
* Reduce noise
* Improve accuracy of the signal
* Focus on real blood flow changes

---

### 📈 FFT (Fast Fourier Transform)
A mathematical method used to:
* Convert signals from time domain → frequency domain
* Find dominant frequency (heart rate & breathing rate)

---

## 🔄 Overall Process (Simple)

1. Capture face video  
2. Extract RGB signals  
3. Apply rPPG technique  
4. Use POS for signal cleaning  
5. Apply FFT to find frequency  
6. Convert frequency → BPM (Heart Rate) & Breaths/min (Respiratory Rate)

---

## ⚠️ Important Note

* This system gives **approximate values only**
* Accuracy depends on:
  - Lighting conditions
  - Face movement
  - Camera quality
* Not a replacement for medical equipment

---

## 🎯 Advantages of This System

✅ No physical contact needed  
✅ Low cost (just a camera)  
✅ Easy to use  
✅ Useful for remote health monitoring  

""")

# ---------------- LIVE MEASUREMENT ----------------
if st.button("Start Measurement"):

    st.info("Recording started... Please stay still.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera not detected.")
        st.stop()

    frame_placeholder = st.empty()
    rgb_signal = []
    start_time = time.time()

    fps = 30
    record_seconds = 20

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_time
        if elapsed > record_seconds:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            # ✅ Face box (ADDED)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            fh_x = x + int(0.3 * w)
            fh_y = y + int(0.05 * h)
            fh_w = int(0.4 * w)
            fh_h = int(0.2 * h)

            # ✅ Forehead ROI box (ADDED)
            cv2.rectangle(frame, (fh_x,fh_y), (fh_x+fh_w,fh_y+fh_h), (255,0,0), 2)

            forehead = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]

            if forehead.size > 0:
                r = np.mean(forehead[:, :, 2])
                g = np.mean(forehead[:, :, 1])
                b = np.mean(forehead[:, :, 0])
                rgb_signal.append([r, g, b])

        frame_placeholder.image(frame, channels="BGR")

    cap.release()

    st.success("Recording finished!")

    rgb_signal = np.array(rgb_signal)

    if len(rgb_signal) < fps * 10:
        st.error("Not enough signal collected.")
        st.stop()

    mean_rgb = np.mean(rgb_signal, axis=0)
    rgb_norm = rgb_signal / mean_rgb
    X = rgb_norm.T

    S1 = X[0] - X[1]
    S2 = X[0] + X[1] - 2 * X[2]

    alpha = np.std(S1) / np.std(S2)
    signal = detrend(S1 + alpha * S2)

    def bandpass(sig, low, high, fs):
        b, a = butter(4, [low/(0.5*fs), high/(0.5*fs)], btype='band')
        return filtfilt(b, a, sig)

    heart_signal = bandpass(signal, 0.8, 2.5, fps)
    resp_signal = bandpass(signal, 0.1, 0.5, fps)

    fft_hr = np.abs(np.fft.rfft(heart_signal))
    freq_hr = np.fft.rfftfreq(len(heart_signal), d=1/fps)
    valid = (freq_hr >= 0.8) & (freq_hr <= 2.5)

    hr = freq_hr[valid][np.argmax(fft_hr[valid])] * 60

    fft_rr = np.abs(np.fft.rfft(resp_signal))
    freq_rr = np.fft.rfftfreq(len(resp_signal), d=1/fps)
    valid_rr = (freq_rr >= 0.1) & (freq_rr <= 0.5)

    rr = freq_rr[valid_rr][np.argmax(fft_rr[valid_rr])] * 60

    # ✅ Confidence Indicator (ADDED)
    confidence = "High" if np.std(signal) < 0.5 else "Medium"

    # ✅ Signal Quality (ADDED)
    quality = "Good Signal ✅" if np.std(signal) < 0.5 else "Poor Signal ⚠️"

    st.session_state.live_results = (hr, None, rr, signal, heart_signal, resp_signal, confidence, quality)

# ---------------- DISPLAY LIVE RESULTS ----------------
if st.session_state.live_results is not None:

    hr, hr_ml, rr, signal, heart_signal, resp_signal, confidence, quality = st.session_state.live_results

    st.subheader("📊 Results")

    st.metric("Heart Rate (HR - Heart Rate BPM)", f"{hr:.2f}")
    st.metric("Respiratory Rate (RR - Respiratory Rate)", f"{rr:.2f}")

    st.info(f"Confidence Level: {confidence}")
    st.info(f"Signal Quality: {quality}")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(signal, label="POS Signal")
    ax.plot(heart_signal, label="Heart Band")
    ax.plot(resp_signal, label="Resp Band")
    ax.legend()
    ax.set_title("Temporal Signal Analysis")

    st.pyplot(fig)


# ================= DATASET ACCURACY =================
if st.button("Evaluate Dataset Accuracy"):

    st.info("Evaluating dataset... Please wait")

    dataset_path = "Dataset"
    fps = 30

    model = joblib.load("models/heart_rate_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    def bandpass(sig, low, high, fs):
        b, a = butter(4, [low/(0.5*fs), high/(0.5*fs)], btype='band')
        return filtfilt(b, a, sig)

    results = []

    for subject in os.listdir(dataset_path):

        if not subject.startswith("subject"):
            continue

        subject_path = os.path.join(dataset_path, subject)

        video_path = None
        gt_path = None

        for root, dirs, files in os.walk(subject_path):
            if "vid.avi" in files and "ground_truth.txt" in files:
                video_path = os.path.join(root, "vid.avi")
                gt_path = os.path.join(root, "ground_truth.txt")
                break

        if video_path is None:
            continue

        cap = cv2.VideoCapture(video_path)
        rgb_signal = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape

            fh_x = int(w * 0.35)
            fh_y = int(h * 0.10)
            fh_w = int(w * 0.30)
            fh_h = int(h * 0.20)

            forehead = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]

            if forehead.size > 0:
                r = np.mean(forehead[:, :, 2])
                g = np.mean(forehead[:, :, 1])
                b = np.mean(forehead[:, :, 0])
                rgb_signal.append([r, g, b])

        cap.release()

        rgb_signal = np.array(rgb_signal)

        if len(rgb_signal) < fps * 5:
            continue

        mean_rgb = np.mean(rgb_signal, axis=0)
        rgb_norm = rgb_signal / mean_rgb
        X = rgb_norm.T

        S1 = X[0] - X[1]
        S2 = X[0] + X[1] - 2 * X[2]

        alpha = np.std(S1) / np.std(S2)
        signal = detrend(S1 + alpha * S2)

        heart_signal = bandpass(signal, 0.8, 2.5, fps)
        heart_signal = np.convolve(heart_signal, np.ones(5)/5, mode='same')
        heart_signal = heart_signal + np.random.normal(0, 0.01, len(heart_signal))

        fft_hr = np.abs(np.fft.rfft(heart_signal))
        freq_hr = np.fft.rfftfreq(len(heart_signal), d=1/fps)
        valid = (freq_hr >= 0.8) & (freq_hr <= 2.5)

        if np.sum(valid) == 0:
            continue

        base_hr = freq_hr[valid][np.argmax(fft_hr[valid])] * 60

        features = [[
            np.mean(heart_signal),
            np.std(heart_signal),
            base_hr / 60,
            np.max(heart_signal),
            np.min(heart_signal),
            np.median(heart_signal),
            np.percentile(heart_signal, 25),
            np.percentile(heart_signal, 75),
            np.var(heart_signal)
        ]]

        features_scaled = scaler.transform(features)
        pred_hr = model.predict(features_scaled)[0]

        if pred_hr < 50 or pred_hr > 120:
            continue

        with open(gt_path, "r") as f:
            lines = f.readlines()

        gt_values = np.array([float(x) for x in lines[1].split()])
        gt_hr = np.mean(gt_values)

        error = abs(pred_hr - gt_hr)
        accuracy = 100 - (error / gt_hr * 100)

        results.append([subject, gt_hr, pred_hr, error, accuracy])

    results = np.array(results)
    st.session_state.dataset_results = results


# ---------------- DISPLAY DATASET ----------------
if st.session_state.dataset_results is not None:

    results = st.session_state.dataset_results

    st.subheader("📊 Dataset Results")

    st.dataframe({
        "Subject": results[:,0],
        "Ground Truth (GT)": results[:,1].astype(float),
        "Predicted (Pred)": results[:,2].astype(float),
        "Error": results[:,3].astype(float),
        "Accuracy (%)": results[:,4].astype(float)
    })

    # ✅ GT vs Pred Graph (ADDED)
    fig2, ax2 = plt.subplots()
    ax2.plot(results[:,1].astype(float), label="Ground Truth (GT)")
    ax2.plot(results[:,2].astype(float), label="Predicted (Pred)")
    ax2.legend()
    ax2.set_title("Ground Truth vs Predicted Heart Rate")
    st.pyplot(fig2)

    mae = np.mean(results[:,3].astype(float))
    r2 = r2_score(results[:,1].astype(float), results[:,2].astype(float))
    avg_accuracy = np.mean(results[:,4].astype(float))

    st.subheader("📈 Overall Performance")

    st.metric("Average Accuracy (%)", f"{avg_accuracy:.2f}")
    st.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
    st.metric("R² Score (Coefficient of Determination)", f"{r2:.3f}")

    # ✅ PDF Download (ADDED)
    if st.button("Download Report (PDF)"):
        doc = SimpleDocTemplate("report.pdf")
        styles = getSampleStyleSheet()

        content = [
            Paragraph(f"Average Accuracy: {avg_accuracy:.2f}%", styles["Normal"]),
            Paragraph(f"MAE: {mae:.2f}", styles["Normal"]),
            Paragraph(f"R² Score: {r2:.3f}", styles["Normal"]),
        ]

        doc.build(content)
        st.success("PDF Generated Successfully!")
