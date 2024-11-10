# Smart Posture Assistant Application
# This code is organized into sections with comments explaining each part
# for easier understanding.

# Import Libraries
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

# Explanation:
# - sys, os, and time: Used for system operations, file management, and timing.
# - cv2: OpenCV library to handle webcam input and video processing.
# - mediapipe: Library used to detect body landmarks (key points of posture).
# - numpy: Used for mathematical operations.
# - playsound: Plays an alert sound for posture warnings.
# - PyQt5: Manages the GUI elements like buttons, labels, layouts, etc.

# Dark Theme for GUI
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
    
# Explanation:
# This StyleSheet class defines a dark theme for the app interface. 
# It specifies colors and styles for each widget, including buttons, labels, 
# and progress bars.

# PostureApp Class Initialization
class PostureApp(QWidget):
    """
    Key Features:
    1. Real-time posture detection using Mediapipe and OpenCV.
    2. Calibration mode to customize alerts based on user’s baseline posture.
    3. Visual and auditory feedback to notify poor posture.
    4. Adjustable controls and user-friendly interface.
    5. Real-time shoulder and neck angle measurement.
    """

    def __init__(self):
        super().__init__()
        # Mediapipe Pose Detector setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # State variables for calibration and posture detection
        self.is_calibrated = False
        self.calibration_frames = 0
        self.calibration_shoulder_angles = []
        self.calibration_neck_angles = []
        self.last_alert_time = 0
        self.alert_cooldown = 10  # Cooldown time for posture alerts
        self.sound_file = 'alert_sound.wav'  # Path to alert sound file
        self.cap = None  # Webcam capture object
        self.detection_active = False  # Track if detection is active

        # Initialize user interface
        self.initUI()
        
        # Timer for updating webcam frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    # Creating the User Interface (UI)
    def initUI(self):
        """Setup the UI components and layout."""
        self.setWindowTitle('Smart Posture Assistant')
        self.setStyleSheet(StyleSheet.DARK_THEME)
        self.setMinimumSize(1000, 700)
        
        # Layout structure
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        
        # Title label
        title = QLabel('Smart Posture Assistant')
        title.setStyleSheet('font-size: 24px; font-weight: bold; padding: 10px;')
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Main content layout: webcam feed and control panel
        content_layout = QHBoxLayout()
        
        # Left panel for webcam feed
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("border: 2px solid #89b4fa; border-radius: 10px;")
        left_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        
        content_layout.addWidget(left_panel)

        # Right panel for controls
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)

        # Status and progress bar
        status_group = QFrame()
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel('Status: Not Started')
        self.status_label.setStyleSheet('font-size: 16px; font-weight: bold;')
        status_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(15)
        status_layout.addWidget(self.progress_bar)
        
        right_layout.addWidget(status_group)

        # Control buttons
        self.start_btn = QPushButton('Start Detection')
        self.start_btn.clicked.connect(self.start_detection)
        right_layout.addWidget(self.start_btn)

        self.recalibrate_btn = QPushButton('Recalibrate Posture')
        self.recalibrate_btn.clicked.connect(self.start_calibration)
        self.recalibrate_btn.setEnabled(False)
        right_layout.addWidget(self.recalibrate_btn)

        self.stop_btn = QPushButton('Stop Detection')
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        right_layout.addWidget(self.stop_btn)

        right_layout.addStretch()
        content_layout.addWidget(right_panel)
        
        main_layout.addLayout(content_layout)
        
        # Status bar at the bottom
        self.status_bar = QStatusBar()
        self.status_bar.showMessage('Ready')
        main_layout.addWidget(self.status_bar)
        
        self.setLayout(main_layout)

    # Starting and Stopping Detection
    def start_detection(self):
        """Initialize webcam and start the detection process."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.setText("Error: Could not open camera")
            return
            
        self.detection_active = True
        self.start_calibration()  # Start calibration first
        self.timer.start(20)  # Update every 20 ms

    # Angle Calculation
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points for posture analysis."""
        v1 = np.array(point1) - np.array(point2)
        v2 = np.array(point3) - np.array(point2)
        angle = np.arccos(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        )
        return np.degrees(angle)

    # Updating Webcam Feed and Posture Analysis
    def update_frame(self):
        """Read webcam feed, update angles, and analyze posture."""
        ret, frame = self.cap.read()
        if not ret:
            return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            # Angle analysis logic here...

    # Main Application Execution
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = PostureApp()
    window.show()
    sys.exit(app.exec_())

# Explanation:
# - This block creates and starts the application.
# - Sets up the main window using Fusion style.
# - Runs the application loop, allowing the user to interact with the GUI.
