import sys
import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QProgressBar
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage

# Initialize MediaPipe Pose and webcam
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# GUI class
class PostureApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize the GUI layout
        self.initUI()

        # Variables for posture correction
        self.is_calibrated = False
        self.calibration_frames = 0
        self.calibration_shoulder_angles = []
        self.calibration_neck_angles = []
        self.last_alert_time = 0
        self.alert_cooldown = 10  # Cooldown between alerts
        self.sound_file = r'C:\Users\Param\Desktop\folder\alert_sound.wav'
        self.cap = None  # Webcam capture

        # Timer for updating the webcam feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def initUI(self):
        # Main window setup
        self.setWindowTitle('BTI_IT sem-V Posture Corrector App')
        self.setGeometry(200, 200, 800, 600)
        self.setStyleSheet("background-color: #2C2C2C; color: white;")

        # Start and Stop buttons
        self.start_btn = QPushButton('Start Posture Detection', self)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.start_btn.clicked.connect(self.start_detection)

        self.stop_btn = QPushButton('Stop Posture Detection', self)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #e53935;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_detection)

        # Webcam display area
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("border: 2px solid white;")

        # Calibration progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #555555;
                border: 1px solid #000000;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #00FF7F;
                width: 10px;
            }
        """)

        # Status label
        self.status_label = QLabel('Status: Not Started', self)
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")

        # Layout configuration
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        self.setLayout(layout)

    def start_detection(self):
        self.cap = cv2.VideoCapture(0)  # Open webcam
        self.status_label.setText("Status: Calibrating...")
        self.timer.start(20)

    def stop_detection(self):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.status_label.setText("Status: Stopped")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Pose Detection: Extract key body landmarks
            left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
            right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                              int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
            left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                        int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))

            # Angle Calculation
            shoulder_angle = self.calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
            neck_angle = self.calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))

            # Calibration
            if not self.is_calibrated and self.calibration_frames < 30:
                self.calibration_shoulder_angles.append(shoulder_angle)
                self.calibration_neck_angles.append(neck_angle)
                self.calibration_frames += 1
                self.progress_bar.setValue(int((self.calibration_frames / 30) * 100))
            elif not self.is_calibrated:
                self.shoulder_threshold = np.mean(self.calibration_shoulder_angles) - 10
                self.neck_threshold = np.mean(self.calibration_neck_angles) - 10
                self.is_calibrated = True
                self.status_label.setText("Status: Detection Running")

            # Posture Feedback
            current_time = time.time()
            if self.is_calibrated:
                if shoulder_angle < self.shoulder_threshold or neck_angle < self.neck_threshold:
                    self.status_label.setText("Status: Poor Posture")
                    if current_time - self.last_alert_time > self.alert_cooldown:
                        if os.path.exists(self.sound_file):
                            playsound(self.sound_file)
                        self.last_alert_time = current_time
                else:
                    self.status_label.setText("Status: Good Posture")

            # Draw skeleton and angles on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert image for displaying in the GUI
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        converted_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(converted_image)
        self.image_label.setPixmap(pixmap)

    def calculate_angle(self, point1, point2, point3):
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        if magnitude1 == 0 or magnitude2 == 0:
            return 0

        cosine_angle = dot_product / (magnitude1 * magnitude2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))

        return angle

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = PostureApp()
    ex.show()
    sys.exit(app.exec_())
