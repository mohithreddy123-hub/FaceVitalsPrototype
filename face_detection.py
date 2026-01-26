import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
signal = []
fps = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # ---- FACE ----
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # ---- FOREHEAD ----
        fh_x = x + int(0.3 * w)
        fh_y = y + int(0.05 * h)
        fh_w = int(0.4 * w)
        fh_h = int(0.2 * h)

        forehead = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]
        cv2.rectangle(frame, (fh_x, fh_y),
                      (fh_x+fh_w, fh_y+fh_h), (255, 0, 0), 2)

        if forehead.size > 0:
            green_mean = np.mean(forehead[:, :, 1])
            signal.append(green_mean)

        # ---- LEFT CHEEK ----
        lc_x = x + int(0.1 * w)
        lc_y = y + int(0.45 * h)
        lc_w = int(0.3 * w)
        lc_h = int(0.25 * h)

        cv2.rectangle(frame, (lc_x, lc_y),
                      (lc_x+lc_w, lc_y+lc_h), (0, 0, 255), 2)

        # ---- RIGHT CHEEK ----
        rc_x = x + int(0.6 * w)
        rc_y = y + int(0.45 * h)
        rc_w = int(0.3 * w)
        rc_h = int(0.25 * h)

        cv2.rectangle(frame, (rc_x, rc_y),
                      (rc_x+rc_w, rc_y+rc_h), (0, 0, 255), 2)

    cv2.imshow("Face, Forehead & Cheeks Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------- SIGNAL PROCESSING ----------------
signal = np.array(signal)

def bandpass(sig, low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

# Heart Rate
heart_signal = bandpass(signal, 0.8, 2.5, fps)
fft_hr = np.abs(np.fft.rfft(heart_signal))
freq_hr = np.fft.rfftfreq(len(heart_signal), d=1/fps)
valid_hr = (freq_hr >= 0.8) & (freq_hr <= 2.5)
heart_rate = freq_hr[valid_hr][np.argmax(fft_hr[valid_hr])] * 60

# Respiratory Rate
resp_signal = bandpass(signal, 0.1, 0.5, fps)
fft_rr = np.abs(np.fft.rfft(resp_signal))
freq_rr = np.fft.rfftfreq(len(resp_signal), d=1/fps)
valid_rr = (freq_rr >= 0.1) & (freq_rr <= 0.5)
resp_rate = freq_rr[valid_rr][np.argmax(fft_rr[valid_rr])] * 60

print(f"\nEstimated Heart Rate      : {heart_rate:.2f} BPM")
print(f"Estimated Respiratory Rate: {resp_rate:.2f} BPM")

plt.figure()
plt.plot(signal, label="Raw Signal")
plt.plot(heart_signal, label="Heart Signal")
plt.plot(resp_signal, label="Respiratory Signal")
plt.legend()
plt.title("Temporal Pixel Variation Signals")
plt.show()
