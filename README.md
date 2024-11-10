# LiveSmart - Smart Posture Assistant for Spine Health

**LiveSmart** is a smart posture assistant designed to help users maintain proper posture and prevent spine-related injuries. The app uses real-time computer vision technology to monitor posture, provide feedback, and promote healthier ergonomic practices.

---

## 1. Purpose of the App

The **Smart Posture Assistant** app is designed to:

- **Monitor** a user’s posture in real-time using computer vision and provide feedback.
- **Detect** slouching or misalignment, helping users maintain a healthy posture.
- **Alert** users immediately when poor posture is detected, reducing the risk of long-term physical strain.
- **Provide** easy access to posture correction tools for people who work long hours in front of computers.
- **Promote** better ergonomic practices, leading to improved productivity and overall physical well-being.

---

## 2. Key Features of the App

### Real-time Posture Detection

- Uses **Mediapipe** and **OpenCV** to capture and analyze the live video feed for posture tracking.
- Calculates angles between body parts (e.g., neck and shoulder) to determine if the user is maintaining good posture.

### Calibration System

- Custom calibration adjusts to each user’s natural posture, establishing a baseline for posture analysis.
- Angle thresholds for neck and shoulder positions are set based on calibration, providing accurate feedback.

### Audio and Visual Alerts

- Provides **audio alerts** to warn users if poor posture is detected.
- Updates the status visually with messages like "Good Posture" or "Poor Posture" and uses **color indicators** to reinforce feedback.

### Intuitive, Dark-themed User Interface

- **Dark theme UI** for a modern, ergonomic-friendly experience.
- The interface includes a **webcam feed panel**, **angle indicators**, and a **status bar** for real-time posture updates.

### Control Panel with Multiple Options

- Includes **Start**, **Stop**, and **Recalibrate** controls for easy management of posture detection.
- A **progress bar** visually represents the calibration process.

---

## 3. Implementation Plan

To build the app, the following steps will be taken:

### Environment Setup

- Install dependencies: `Mediapipe`, `OpenCV`, `PyQt5`, `numpy`, and `playsound` in a virtual environment.

### GUI Development

- Build the user interface using **PyQt5**.
- Structure the UI with key sections: webcam feed, control buttons, and status display.
- Apply **dark theme styles** using PyQt5’s `StyleSheet` class to enhance the app's look and feel.

### Posture Detection Logic

- Set up a **Mediapipe Pose instance** to detect shoulder and neck landmarks.
- Develop an **angle calculation function** to measure shoulder and neck angles.
- Implement **calibration logic** to establish baseline angles for each user.

### Alert Mechanism

- Implement an alert function for **audio and visual notifications** when poor posture is detected.
- Add a **cooldown feature** to ensure alerts occur at appropriate intervals.

### Testing and Optimization

- Test on various users to refine calibration and posture thresholds.
- Optimize frame processing to ensure **real-time performance** without lag.

---

## 4. Future Features to Add

### Detailed Posture Analytics and Tracking

- Track posture data over time and provide **reports** or **graphs** to show progress.
- Allow users to set **daily goals** for maintaining good posture.

### AI-Enhanced Posture Classification

- Integrate machine learning to classify different types of postural misalignments (e.g., forward head posture, shoulder rounding).
- Provide **exercise recommendations** based on detected posture issues.

### Cross-Platform Compatibility

- Develop a **mobile app** to track posture on smartphones.
- Create a **web-based version** that integrates with webcams for posture monitoring.

### Integration with Ergonomic Accessories

- Integrate with **adjustable standing desks** or **ergonomic chairs**, adjusting the user’s environment based on detected posture.
- Provide **desk height** and **seating recommendations** based on posture feedback.

### Exercise and Stretch Reminders

- Add reminders for **stretching exercises** or **micro-breaks** to reduce physical strain.
- Offer **personalized exercise recommendations** to improve posture.

---

## 5. Additional Considerations

### Privacy and Security

- Process video **locally** on the device to ensure user privacy.
- Provide clear **privacy policies** detailing how video data is handled.

### Accessibility

- Include customizable alerts and visual aids, such as:
  - **Volume control** for audio alerts.
  - **High-contrast mode** for visually impaired users.

### User Education and Support

- Provide a **help section** or **tutorial** within the app to educate users on the importance of good posture.
- Include an **FAQ section** for troubleshooting and setup assistance.

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/LiveSmart.git
    cd LiveSmart
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    python main.py
    ```

---

v3.2 is the final code final to run 

---

## Acknowledgments

- **Mediapipe** and **OpenCV** for posture detection.
- **PyQt5** for the graphical user interface.
- **Playsound** for audio alerts.
- Thanks to the open-source community for their contributions.

