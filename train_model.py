import cv2
import numpy as np
import os
from scipy.signal import butter, filtfilt, detrend
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ---------------- SETTINGS ----------------
dataset_path = "Dataset"
fps = 30

if not os.path.exists(dataset_path):
    print("Dataset folder not found!")
    exit()

subjects = [s for s in os.listdir(dataset_path) if s.startswith("subject")]

X = []
y = []

# ---------------- BANDPASS FILTER ----------------
def bandpass(sig, low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

# ---------------- PROCESS EACH SUBJECT ----------------
for subject in subjects:

    subject_path = os.path.join(dataset_path, subject)
    print("\nProcessing:", subject)

    video_path = None
    gt_path = None

    # Auto search inside folder
    for root, dirs, files in os.walk(subject_path):
        if "vid.avi" in files and "ground_truth.txt" in files:
            video_path = os.path.join(root, "vid.avi")
            gt_path = os.path.join(root, "ground_truth.txt")
            break

    if video_path is None:
        print("Video not found for", subject)
        continue

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open video for", subject)
        continue

    rgb_signal = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w, _ = frame.shape

        # Forehead ROI
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

    print("Frames:", frame_count)
    print("RGB length:", len(rgb_signal))

    rgb_signal = np.array(rgb_signal)

    if len(rgb_signal) < fps * 5:
        print("Not enough signal")
        continue

    # ---------------- POS ALGORITHM ----------------
    mean_rgb = np.mean(rgb_signal, axis=0)
    rgb_norm = rgb_signal / mean_rgb

    X_rgb = rgb_norm.T

    S1 = X_rgb[0] - X_rgb[1]
    S2 = X_rgb[0] + X_rgb[1] - 2 * X_rgb[2]

    alpha = np.std(S1) / np.std(S2)
    pos_signal = S1 + alpha * S2

    signal = detrend(pos_signal)

    # ---------------- HEART BAND ----------------
    heart_signal = bandpass(signal, 0.8, 2.5, fps)

    fft_hr = np.abs(np.fft.rfft(heart_signal))
    freq_hr = np.fft.rfftfreq(len(heart_signal), d=1/fps)
    valid = (freq_hr >= 0.8) & (freq_hr <= 2.5)

    if np.sum(valid) == 0:
        continue

    dominant_freq = freq_hr[valid][np.argmax(fft_hr[valid])]

    features = [
        np.mean(heart_signal),
        np.std(heart_signal),
        dominant_freq,
        np.max(fft_hr[valid])
    ]

    # ---------------- GROUND TRUTH ----------------
    with open(gt_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        continue

    hr_values = np.array([float(x) for x in lines[1].split()])
    gt_hr = np.mean(hr_values)

    X.append(features)
    y.append(gt_hr)

# ---------------- TRAIN MODEL ----------------
if len(X) == 0:
    print("No valid data found!")
    exit()

X = np.array(X)
y = np.array(y)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -------- Accuracy --------
predictions = model.predict(X)
mae = mean_absolute_error(y, predictions)
r2 = r2_score(y, predictions)

print("\nModel trained successfully!")
print("Subjects used:", len(X))
print("MAE:", round(mae, 2), "BPM")
print("R2:", round(r2, 3))

if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(model, "models/heart_rate_model.pkl")
print("Model saved.")
