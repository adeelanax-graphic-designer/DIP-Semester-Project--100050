import cv2
import numpy as np
import os

# -------------------------------
# Create result folders
# -------------------------------
os.makedirs("Results/Violation", exist_ok=True)
os.makedirs("Results/No_Violation", exist_ok=True)

violation_log = open("Results/Violation/violation_log.txt", "w")
no_violation_log = open("Results/No_Violation/no_violation_log.txt", "w")

# -------------------------------
# Video paths
# -------------------------------
videos = [
    "Dataset/british_highway_traffic.mp4",
    "Dataset/dhaka_traffic.mp4",
    "Dataset/road_traffic.mp4",
    "Dataset/traffic_detection.mp4",
    "Dataset/Traffic.mp4"
]

STOP_LINE_Y = 300
video_count = 1

for video_path in videos:

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        continue

    print(f"[INFO] Processing: {video_path}")

    cap = cv2.VideoCapture(video_path)
    bg = cv2.createBackgroundSubtractorMOG2()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame = cv2.resize(frame, (800, 500))

        # -------------------------------
        # RED SIGNAL DETECTION (FIXED)
        # -------------------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2

        red_signal = cv2.countNonZero(red_mask) > 800

        if red_signal:
            cv2.putText(frame, "RED SIGNAL", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 3)

        # -------------------------------
        # VEHICLE DETECTION
        # -------------------------------
        fgmask = bg.apply(frame)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(frame, (0, STOP_LINE_Y),
                 (frame.shape[1], STOP_LINE_Y), (255, 0, 0), 2)

        violation_found = False

        for cnt in contours:
            if cv2.contourArea(cnt) > 2000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if (y + h > STOP_LINE_Y) and red_signal:
                    violation_found = True
                    cv2.putText(frame, "TRAFFIC VIOLATION!",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 3)

        # -------------------------------
        # SAVE LOGS
        # -------------------------------
        if violation_found:
            violation_log.write(
                f"Video {video_count} | Frame {frame_id} | Violation Detected\n")
        else:
            no_violation_log.write(
                f"Video {video_count} | Frame {frame_id} | No Violation\n")

        cv2.imshow("Smart Traffic Violation Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    video_count += 1

violation_log.close()
no_violation_log.close()
cv2.destroyAllWindows()
