import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, detrend
import matplotlib.pyplot as plt
import time

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="FaceVitals", layout="centered")

st.title("🫀 Camera-Based Heart & Respiratory Rate Monitor")

st.markdown("""
### What is Heart Rate?
Heart rate is the number of times your heart beats per minute.
Doctors usually measure it using a stethoscope or pulse oximeter.

### What is Respiratory Rate?
Respiratory rate is the number of breaths per minute.
Doctors observe chest movement or use medical monitors.

### How does this system work?
This system uses a camera to detect tiny color changes in the forehead
caused by blood flow. These temporal pixel variations are processed
to estimate Heart Rate and Respiratory Rate.
""")

# ---------------- SETTINGS ----------------
fps = 30
record_seconds = 20
window_seconds = 10
window_size = fps * window_seconds
step_size = fps * 1

# ---------------- START BUTTON ----------------
if st.button("Start Measurement"):

    st.info("Recording started... Please stay still.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    rgb_signal = []
    start_time = time.time()

    frame_placeholder = st.empty()

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

            fh_x = x + int(0.3 * w)
            fh_y = y + int(0.05 * h)
            fh_w = int(0.4 * w)
            fh_h = int(0.2 * h)

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

    # ---------------- POS ----------------
    mean_rgb = np.mean(rgb_signal, axis=0)
    rgb_norm = rgb_signal / mean_rgb
    X = rgb_norm.T

    S1 = X[0] - X[1]
    S2 = X[0] + X[1] - 2 * X[2]

    alpha = np.std(S1) / np.std(S2)
    pos_signal = S1 + alpha * S2
    signal = detrend(pos_signal)

    # ---------------- FILTER ----------------
    def bandpass(sig, low, high, fs, order=4):
        nyq = 0.5 * fs
        b, a = butter(order, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, sig)

    # ---------------- SLIDING WINDOW ----------------
    heart_rates = []
    resp_rates = []

    for start in range(0, len(signal) - window_size, step_size):

        segment = signal[start:start + window_size]

        heart_seg = bandpass(segment, 0.8, 2.5, fps)
        fft_hr = np.abs(np.fft.rfft(heart_seg))
        freq_hr = np.fft.rfftfreq(len(heart_seg), d=1/fps)
        valid_hr = (freq_hr >= 0.8) & (freq_hr <= 2.5)

        if np.sum(valid_hr) > 0:
            hr = freq_hr[valid_hr][np.argmax(fft_hr[valid_hr])] * 60
            heart_rates.append(hr)

        resp_seg = bandpass(segment, 0.1, 0.5, fps)
        fft_rr = np.abs(np.fft.rfft(resp_seg))
        freq_rr = np.fft.rfftfreq(len(resp_seg), d=1/fps)
        valid_rr = (freq_rr >= 0.1) & (freq_rr <= 0.5)

        if np.sum(valid_rr) > 0:
            rr = freq_rr[valid_rr][np.argmax(fft_rr[valid_rr])] * 60
            resp_rates.append(rr)

    heart_rate = np.mean(heart_rates)
    resp_rate = np.mean(resp_rates)

    # ---------------- RESULTS ----------------
    st.subheader("📊 Results")

    st.metric("Heart Rate (BPM)", f"{heart_rate:.2f}")
    st.metric("Respiratory Rate (Breaths/min)", f"{resp_rate:.2f}")

    # ---------------- PLOT ----------------
    heart_signal = bandpass(signal, 0.8, 2.5, fps)
    resp_signal = bandpass(signal, 0.1, 0.5, fps)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(signal, label="POS Signal")
    ax.plot(heart_signal, label="Heart Band")
    ax.plot(resp_signal, label="Resp Band")
    ax.legend()
    ax.set_title("Temporal Signal Analysis")

    st.pyplot(fig)
