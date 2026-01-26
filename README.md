# Face Vitals Prototype (Heart Rate + Respiratory Rate)

This is a mini prototype project that detects a face using OpenCV Haar Cascade and extracts a forehead region to estimate:
- **Heart Rate (BPM)**
- **Respiratory Rate (Breaths per minute)**

It uses webcam video and calculates pixel intensity variation (green channel) for signal processing.

---

## 📌 Features
- Face detection using Haar Cascade
- Forehead region detection
- Signal extraction using green channel mean
- Bandpass filtering for:
  - Heart rate (0.8 Hz – 2.5 Hz)
  - Respiratory rate (0.1 Hz – 0.5 Hz)
- FFT-based estimation
- Signal plotting using Matplotlib

---

## 📂 Project Files
- `face_detection.py` → Main Python file
- `haarcascade_frontalface_default.xml` → Face detection model file

---

## 🛠 Requirements
Install dependencies using:

```bash
pip install opencv-python numpy scipy matplotlib
