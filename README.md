# Robot Gesture Control 

# Robot Gesture Control

A Python-based robotics control system with **PyQt5** and **PySimpleGUI** GUIs.  
The system supports:

- Voice control (via Vosk / Whisper)
- Pose detection & gesture mapping (Mediapipe)
- Manual overrides (GUI buttons)
- Smooth command sending to Arduino
- Real-time camera feedback

---

## Features

- PyQt5 GUI: Smooth, professional interface with live camera feed
- PySimpleGUI GUI: Lightweight, easy-to-modify interface
- Real-time pose analysis with Mediapipe Holistic
- Smooth motion commands sent to Arduino
- Logging and state management for reproducibility

---

## Installation

```bash
# Clone repo
git clone <YOUR_REPO_URL>
cd robot-gesture-control

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
