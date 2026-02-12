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



# **Components Needed:**

| Component                     | Quantity | Notes                                     |
|-------------------------------|---------|-------------------------------------------|
| Arduino Mega 2560              | 1       | Large number of PWM pins                  |
| High-Torque Digital Servos     | ~11     | RDS3115 or similar                        |
| Windows PC/Laptop              | 1       | The "brain" of the robot                  |
| USB Webcam                     | 1       | UVC-compliant or integrated               |
| Microphone                     | 1       | USB or integrated                          |
| 5V 10A Power Supply             | 1       | Must be rated for 10A or more             |
| Arduino Mega Sensor Shield V5   | 1       | Safe power distribution                    |
| Electrolytic Capacitor (1000-4700µF, 10V+) | 1 | Prevents voltage spikes during servo moves|
| USB A-to-B Cable               | 1       | Connect PC to Arduino    



# **Assembly:**

1. Mount Sensor Shield on Arduino Mega.
2. With power OFF, connect 5V & GND from the 10A supply to the blue screw terminal.
3. Install capacitor (negative stripe to GND).
4. Plug servos into 3-pin headers on Sensor Shield according to Arduino sketch pinout.
5. Connect Arduino to PC with USB cable.


On Arduino 

# Arduino IDE

Open DancingRobot_Arduino.ino.

Select Arduino Mega 2560 and correct COM port.

Upload the sketch.

Running the System
Arduino ("Body")

Connect Arduino to PC.

Ensure servo power supply is ON.

Upload DancingRobot_Arduino.ino.

Python ("Brain")

Activate virtual environment:

venv\Scripts\activate


# Run GUI:

python scripts/Gui_Dancing_Robot.py



Expected Results
Step	Expected Outcome
Hardware Setup	Servos, sensors, capacitor installed, power stable
Arduino Upload	Arduino responds, servos idle at neutral
Python GUI Launch	Camera feed visible, manual override buttons responsive
Gesture Control	Left/Right arms, head tilt, eyes respond
Voice Control	Commands detected and executed by servos
Diagnostics	Each test script prints [SUCCESS] or confirms functionality