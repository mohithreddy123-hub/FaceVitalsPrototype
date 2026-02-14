import cv2
import numpy as np
from scipy.signal import butter, filtfilt, detrend
import matplotlib.pyplot as plt
import joblib
import time

# ---------------- SETTINGS ----------------
fps = 30
record_seconds = 20
window_seconds = 10
window_size = fps * window_seconds
step_size = fps * 1

# ---------------- LOAD ML MODEL ----------------
try:
    model = joblib.load("models/heart_rate_model.pkl")
    print("ML model loaded successfully.")
except:
    print("ML model not found! Running only FFT estimation.")
    model = None

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected!")
    exit()

rgb_signal = []
prev_gray = None
start_time = time.time()

print("Recording started... Stay still!")

# ---------------- RECORD RGB SIGNAL ----------------
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

        # Draw face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Forehead ROI
        fh_x = x + int(0.3 * w)
        fh_y = y + int(0.05 * h)
        fh_w = int(0.4 * w)
        fh_h = int(0.2 * h)

        forehead = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]

        cv2.rectangle(frame, (fh_x, fh_y),
                      (fh_x+fh_w, fh_y+fh_h), (255, 0, 0), 2)

        if forehead.size > 0:
            r = np.mean(forehead[:, :, 2])
            g = np.mean(forehead[:, :, 1])
            b = np.mean(forehead[:, :, 0])
            rgb_signal.append([r, g, b])

    prev_gray = gray.copy()

    cv2.putText(frame, f"Recording: {int(elapsed)} sec",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    cv2.imshow("FaceVitals - Motion Compensated", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Recording finished.")

rgb_signal = np.array(rgb_signal)

if len(rgb_signal) < fps * 10:
    print("Not enough signal collected.")
    exit()

# ---------------- POS ALGORITHM ----------------
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

heart_rate_fft = np.mean(heart_rates)
resp_rate = np.mean(resp_rates)

# ---------------- ML PREDICTION ----------------
heart_rate_ml = None

if model is not None and len(heart_rates) > 0:
    dominant_freq = heart_rate_fft / 60

    features = [[
        np.mean(signal),
        np.std(signal),
        dominant_freq,
        np.max(heart_rates)
    ]]

    heart_rate_ml = model.predict(features)[0]

# ---------------- RESULTS ----------------
print("\n================ RESULTS ================")
print(f"FFT Heart Rate  : {heart_rate_fft:.2f} BPM")

if heart_rate_ml is not None:
    print(f"ML Heart Rate   : {heart_rate_ml:.2f} BPM")

print(f"Respiratory Rate: {resp_rate:.2f} BPM")
print("=========================================")

# ---------------- PLOT ----------------
heart_signal = bandpass(signal, 0.8, 2.5, fps)
resp_signal = bandpass(signal, 0.1, 0.5, fps)

plt.figure(figsize=(10,5))
plt.plot(signal, label="POS Signal")
plt.plot(heart_signal, label="Heart Band")
plt.plot(resp_signal, label="Resp Band")
plt.legend()
plt.title("POS + Motion Compensation + Sliding Window")
plt.show()
