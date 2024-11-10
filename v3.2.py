import sys
import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
    QVBoxLayout, QHBoxLayout, QProgressBar, QFrame, QStatusBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon

class StyleSheet:
    DARK_THEME = """
    QWidget {
        background-color: #1e1e2e;
        color: #cdd6f4;
        font-family: 'Segoe UI', Arial;
    }
    QPushButton {
        background-color: #89b4fa;
        color: #1e1e2e;
        border: none;
        border-radius: 8px;
        padding: 12px;
        font-weight: bold;
        font-size: 14px;
        min-width: 120px;
    }
    QPushButton:hover {
        background-color: #b4befe;
    }
    QPushButton:pressed {
        background-color: #74c7ec;
    }
    QPushButton:disabled {
        background-color: #6c7086;
        color: #313244;
    }
    QProgressBar {
        border: 2px solid #89b4fa;
        border-radius: 5px;
        text-align: center;
        background-color: #313244;
    }
    QProgressBar::chunk {
        background-color: #89b4fa;
    }
    QLabel {
        font-size: 14px;
    }
    QStatusBar {
        background-color: #313244;
        color: #cdd6f4;
    }
    """

class PostureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize state variables
        self.is_calibrated = False
        self.calibration_frames = 0
        self.calibration_shoulder_angles = []
        self.calibration_neck_angles = []
        self.last_alert_time = 0
        self.alert_cooldown = 10
        self.sound_file = 'alert_sound.wav'
        self.cap = None
        self.detection_active = False
        
        self.initUI()
        
        # Setup webcam timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def initUI(self):
        self.setWindowTitle('Smart Posture Assistant')
        self.setStyleSheet(StyleSheet.DARK_THEME)
        self.setMinimumSize(1000, 700)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        
        # Title
        title = QLabel('Smart Posture Assistant')
        title.setStyleSheet('font-size: 24px; font-weight: bold; padding: 10px;')
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Content layout
        content_layout = QHBoxLayout()
        
        # Left panel (webcam feed)
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_panel)
        
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("border: 2px solid #89b4fa; border-radius: 10px;")
        left_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        
        content_layout.addWidget(left_panel)

        # Right panel (controls)
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)

        # Status section
        status_group = QFrame()
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel('Status: Not Started')
        self.status_label.setStyleSheet('font-size: 16px; font-weight: bold;')
        status_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(15)
        status_layout.addWidget(self.progress_bar)
        
        right_layout.addWidget(status_group)

        # Controls section
        controls_group = QFrame()
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setSpacing(10)

        self.start_btn = QPushButton('Start Detection')
        self.start_btn.clicked.connect(self.start_detection)
        controls_layout.addWidget(self.start_btn)

        self.recalibrate_btn = QPushButton('Recalibrate Posture')
        self.recalibrate_btn.clicked.connect(self.start_calibration)
        self.recalibrate_btn.setEnabled(False)
        controls_layout.addWidget(self.recalibrate_btn)

        self.stop_btn = QPushButton('Stop Detection')
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)

        right_layout.addWidget(controls_group)
        
        # Stats section
        stats_group = QFrame()
        stats_layout = QVBoxLayout(stats_group)
        
        self.shoulder_angle_label = QLabel('Shoulder Angle: --°')
        self.neck_angle_label = QLabel('Neck Angle: --°')
        stats_layout.addWidget(self.shoulder_angle_label)
        stats_layout.addWidget(self.neck_angle_label)
        
        right_layout.addWidget(stats_group)
        
        right_layout.addStretch()
        content_layout.addWidget(right_panel)
        
        # Add content layout to main layout
        main_layout.addLayout(content_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.showMessage('Ready')
        main_layout.addWidget(self.status_bar)
        
        self.setLayout(main_layout)

    def start_detection(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.setText("Error: Could not open camera")
            return
            
        self.detection_active = True
        self.start_calibration()
        self.timer.start(20)
        self.update_button_states()
        
    def start_calibration(self):
        self.is_calibrated = False
        self.calibration_frames = 0
        self.calibration_shoulder_angles = []
        self.calibration_neck_angles = []
        self.status_label.setText("Status: Calibrating...")
        self.progress_bar.setValue(0)
        
    def stop_detection(self):
        self.detection_active = False
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.status_label.setText("Status: Stopped")
        self.update_button_states()
        
    def update_button_states(self):
        self.start_btn.setEnabled(not self.detection_active)
        self.stop_btn.setEnabled(self.detection_active)
        self.recalibrate_btn.setEnabled(self.detection_active and self.is_calibrated)

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

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Flip frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract landmarks
            left_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                           int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
            right_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                            int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
            left_ear = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                       int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))

            # Calculate angles
            shoulder_angle = self.calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
            neck_angle = self.calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))

            # Update angle labels
            self.shoulder_angle_label.setText(f'Shoulder Angle: {shoulder_angle:.1f}°')
            self.neck_angle_label.setText(f'Neck Angle: {neck_angle:.1f}°')

            # Calibration process
            if not self.is_calibrated:
                if self.calibration_frames < 30:
                    self.calibration_shoulder_angles.append(shoulder_angle)
                    self.calibration_neck_angles.append(neck_angle)
                    self.calibration_frames += 1
                    self.progress_bar.setValue(int((self.calibration_frames / 30) * 100))
                else:
                    self.shoulder_threshold = np.mean(self.calibration_shoulder_angles) - 10
                    self.neck_threshold = np.mean(self.calibration_neck_angles) - 10
                    self.is_calibrated = True
                    self.status_label.setText("Status: Detection Running")
                    self.recalibrate_btn.setEnabled(True)
                    self.status_bar.showMessage('Calibration complete')

            # Posture checking
            if self.is_calibrated:
                current_time = time.time()
                if shoulder_angle < self.shoulder_threshold or neck_angle < self.neck_threshold:
                    self.status_label.setText("Status: Poor Posture")
                    self.status_label.setStyleSheet("color: #f38ba8; font-size: 16px; font-weight: bold;")
                    if current_time - self.last_alert_time > self.alert_cooldown:
                        if os.path.exists(self.sound_file):
                            playsound(self.sound_file)
                        self.last_alert_time = current_time
                else:
                    self.status_label.setText("Status: Good Posture")
                    self.status_label.setStyleSheet("color: #a6e3a1; font-size: 16px; font-weight: bold;")

            # Draw skeleton
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # Draw angle indicators
            cv2.putText(frame, f"Shoulder: {shoulder_angle:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Neck: {neck_angle:.1f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convert frame to Qt format and display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better looking widgets
    window = PostureApp()
    window.show()
    sys.exit(app.exec_())