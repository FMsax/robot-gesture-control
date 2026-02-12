import time
import threading
import queue
import json
import sys
from datetime import datetime
from typing import Optional, Iterable

import cv2
import numpy as np
import PySimpleGUI as sg
import serial
import pyaudio
import mediapipe as mp


# CONFIGURATION

ARDUINO_COM_PORT = "COM11"
ARDUINO_BAUD_RATE = 115200
VOSK_MODEL_PATH = r"C:\Users\DELL\desktop\LIGHER_GPU_MODEL\vosk-model-small-en-us-0.15"

SMOOTH_SEND_INTERVAL = 0.5
POSE_COOLDOWN = 3.0

# LOGGER (queue so GUI thread can safely read logs)
log_q = queue.Queue()

def log_local(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    log_q.put(f"[{ts}] {msg}")


# ARDUINO MANAGEMENT

_arduino: Optional[serial.Serial] = None
_last_cmd = ""
_last_time = 0.0

def init_arduino(port: str = ARDUINO_COM_PORT, baud: int = ARDUINO_BAUD_RATE) -> Optional[serial.Serial]:
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2)  # allow Arduino to reset
        log_local(f"Arduino connected on {port}")
        return ser
    except Exception as e:
        log_local(f"Arduino connect failed: {e}")
        return None

def smooth_send(cmd: str) -> None:
    """Send command to Arduino but avoid flooding identical commands."""
    global _last_cmd, _last_time, _arduino
    now = time.time()
    try:
        if cmd != _last_cmd or (now - _last_time) > SMOOTH_SEND_INTERVAL:
            msg = f"{cmd};"
            if _arduino and _arduino.is_open:
                _arduino.write(msg.encode())
            log_local(f"[CMD] {msg}")
            _last_cmd = cmd
            _last_time = now
    except Exception as e:
        log_local(f"Serial error while sending '{cmd}': {e}")


# SPEECH RECOGNITION THREAD

speech_q: "queue.Queue[str]" = queue.Queue()

# Centralized phrases (easy to extend)
LEFT_ARM_DOWN_PHRASES = {
    # exact phrase matches
    "left arm down",
    "drop left arm",
    "lower left arm",
    "lower the left arm",
    "bring left arm down",
    # fuzzy phrases that could happen
    "left under",
    "left undone",
    "left i'm",
    "lives under",
    "lives on under",
    "left on now",
}

VOSK_WORDS = [
    "left arm up", "left arm down", "right arm up", "right arm down",
    "head left", "head right", "head up", "head down",
    "eyes left", "eyes center", "eyes right",
    "open", "close", "shut", "center"
] + list(LEFT_ARM_DOWN_PHRASES)


def speech_thread_fn() -> None:
    """Thread that listens to microphone and pushes recognized text to speech_q.

    Uses faster-whisper (Whisper). Added robust mic-rate auto-detection, extra logging
    at thread start, raw-sample logging for debugging, and guaranteed exit logging.
    """
    log_local("Speech thread started — initializing Whisper and audio")

    
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        log_local(f"Whisper import failed: {e}")
        log_local("Speech thread exited unexpectedly (import failure).")
        return

    # Create PyAudio and detect the default input sample rate
    try:
        p = pyaudio.PyAudio()
    except Exception as e:
        log_local(f"PyAudio initialization failed: {e}")
        log_local("Speech thread exited unexpectedly (pyaudio init failure).")
        return

    # Auto-detect microphone
    try:
        info = p.get_default_input_device_info()
        RATE = int(info.get("defaultSampleRate", 16000))
        log_local(f"Microphone default rate detected: {RATE}")
    except Exception as e:
        log_local(f"Failed to detect default input device rate: {e}. Falling back to 16000 Hz.")
        RATE = 16000

    CHUNK = 4096  # frames per read

    # Create Whisper model (catch errors early)
    model = None
    try:
        model = WhisperModel("small", device="cpu")
        log_local("Whisper model loaded successfully.")
    except Exception as e:
        log_local(f"Whisper model load failed: {e}")
        log_local("Speech thread exited unexpectedly (model load failure).")
        return

    # Open recording stream
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)
        stream.start_stream()
    except Exception as e:
        log_local(f"Microphone unavailable or unsupported format: {e}")
        log_local("Speech thread exited unexpectedly (mic open failure).")
        return

    log_local("Speech thread running (Whisper).")
    frame_count = 0
    max_level = 0
    silence_frames = 0

    # Buffering for short-window transcriptions to emulate partials & final results.
    audio_buffer = bytearray()
    TRANSCRIBE_INTERVAL_SEC = 0.8  # how often to run a transcription on buffered audio
    transcribe_min_samples = int(RATE * TRANSCRIBE_INTERVAL_SEC)  # in samples
    last_transcribe_time = time.time()

    # Helper: convert raw int16 bytes -> float32 normalized array (Whisper expects floats -1..1)
    def bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if arr.size == 0:
            return np.array([], dtype=np.float32)
        arr = arr / 32768.0
        return arr

    def transcribe_buffer(np_audio: np.ndarray):
        """Transcribe float32 numpy audio via faster-whisper.

        Returns (text, word_info) where word_info is best-effort list of dicts with word/start/end.
        """
        text_out = ""
        word_info = None
        try:
            # attempt word-level timestamps if available; model.transcribe accepts raw float array
            try:
                segments, info = model.transcribe(np_audio, language="en", word_timestamps=True, vad_filter=False)
            except TypeError:
                # older faster-whisper versions may not accept word_timestamps kwarg
                segments, info = model.transcribe(np_audio, language="en", vad_filter=False)

            for seg in segments:
                if hasattr(seg, "text"):
                    text_out += seg.text.strip() + " "
                if hasattr(seg, "words"):
                    if word_info is None:
                        word_info = []
                    for w in seg.words:
                        word_info.append({
                            "word": getattr(w, "word", None),
                            "start": getattr(w, "start", None),
                            "end": getattr(w, "end", None)
                        })
        except Exception as ex:
            log_local(f"Whisper transcribe error: {ex}")
        return text_out.strip(), word_info

    try:
        # Main loop: read mic, silence detection, run transcription on buffered chunks
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, np.int16)
                audio_level = int(audio_data.max()) if audio_data.size else 0
                max_level = max(max_level, audio_level)

                # For debugging: log a short snapshot of raw samples occasionally
                if frame_count % 10 == 0:
                    raw_snippet = audio_data[:50].tolist() if audio_data.size else []
                    log_local(f"[MIC] Level: {audio_level} | Max: {max_level} | Silence: {silence_frames} | RAW_SNIPPET: {raw_snippet}")
                    max_level = 0

                # Detect silence (adjust threshold if needed)
                if audio_level < 80:
                    silence_frames += 1
                else:
                    silence_frames = 0

                frame_count += 1

                # Append audio bytes to buffer for periodic transcription (only if not totally silent)
                audio_buffer.extend(data)

                now = time.time()
                # If we've collected enough audio or the interval passed -> attempt transcription
                if (len(audio_buffer) >= transcribe_min_samples * 2) or (now - last_transcribe_time >= TRANSCRIBE_INTERVAL_SEC):
                    if silence_frames < 15:
                        np_audio = bytes_to_float32(bytes(audio_buffer))
                        if np_audio.size:
                            text, words = transcribe_buffer(np_audio)
                            if text and len(text) > 1:
                                # send to queue (same behavior as original)
                                speech_q.put(text)
                                conf_label = "high" if audio_level > 1000 else "unknown"
                                log_local(f"[SPEECH] {text} (confidence: {conf_label})")
                                if words:
                                    try:
                                        log_local(f"[WORD_TIMESTAMPS] {words}")
                                    except Exception:
                                        pass
                                silence_frames = 0
                            else:
                                # No final text — try to log an interim partial if the audio level suggests it
                                if audio_level > 1000:
                                    try:
                                        partial_text = text
                                        if partial_text:
                                            log_local(f"[PARTIAL] {partial_text} (level: {audio_level})")
                                    except Exception:
                                        pass
                        # reset the buffer after processing
                        audio_buffer = bytearray()
                    else:
                        # Long silence -> clear buffer to avoid repeated low-value transcribes
                        audio_buffer = bytearray()
                    last_transcribe_time = now

            except Exception as e:
                log_local(f"Speech thread loop error: {e}")
                time.sleep(0.1)
                continue

    except Exception as e:
        log_local(f"Speech thread top-level error: {e}")

    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        try:
            p.terminate()
        except Exception:
            pass
        log_local("Speech thread exited.")

# MEDIAPIPE HOLISTIC / DRAWING

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def init_holistic():
    return mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        refine_face_landmarks=True
    )

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    refine_face_landmarks=True
)

def draw_selected(frame: np.ndarray, results) -> np.ndarray:
    """Draw pose lines and face contours on the frame."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 120, 0), thickness=2)
        )

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 0, 150), thickness=1)
        )

    return frame


# GUI SETUP, I made it beautiful

sg.theme("DarkBlue3")

manual_buttons = [
    [sg.Button("L_ARM_UP", key="L_ARM_UP"), sg.Button("L_ARM_MID", key="L_ARM_MID"), sg.Button("L_ARM_DOWN", key="L_ARM_DOWN")],
    [sg.Button("R_ARM_UP", key="R_ARM_UP"), sg.Button("R_ARM_MID", key="R_ARM_MID"), sg.Button("R_ARM_DOWN", key="R_ARM_DOWN")],
    [sg.Button("HEAD_LEFT", key="HEAD_LEFT"), sg.Button("HEAD_MID", key="HEAD_MID"), sg.Button("HEAD_RIGHT", key="HEAD_RIGHT")],
    [sg.Button("HEAD_UP", key="HEAD_UP"), sg.Button("HEAD_DOWN", key="HEAD_DOWN")],
    [sg.Button("EYES_LEFT", key="EYES_LEFT"), sg.Button("EYES_CENTER", key="EYES_CENTER"), sg.Button("EYES_RIGHT", key="EYES_RIGHT")],
    [sg.Button("EYELIDS_OPEN", key="EYELIDS_OPEN"), sg.Button("EYELIDS_CLOSED", key="EYELIDS_CLOSED")]
]

layout = [
    [
        sg.Column([
            [sg.Image("", key="-IMAGE-")],
            [sg.Text("Log:")],
            [sg.Multiline("", size=(80, 12), key="-LOG-", disabled=True, autoscroll=True)]
        ]),
        sg.Column([
            [sg.Text("Speech:"), sg.Text("", key="-SPEECH-", size=(30, 1))],
            [sg.Button("Start Voice Control", key="START_VOICE")],
            [sg.Frame("Manual Overrides", manual_buttons)],
            [sg.Button("Reset"), sg.Button("Pause Camera"), sg.Button("Resume Camera"), sg.Button("Exit")]
        ])
    ]
]

window = sg.Window("PAUL IS DANCING", layout, finalize=True)

# CAMERA

def open_camera(index: int = 0) -> Optional[cv2.VideoCapture]:
    try:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            log_local("Camera failed to open.")
            return None
        return cap
    except Exception as e:
        log_local(f"Camera open error: {e}")
        return None

# COMMAND PROCESSING (speech -> robot)


def is_phrase_in_text(text: str, phrases: Iterable[str]) -> bool:
    """Return True if any phrase in phrases appears in text."""
    for p in phrases:
        if p in text:
            return True
    return False

def process_speech(low: str) -> None:
    """Map recognized text to robot commands."""
    # LEFT ARM
    if "left arm" in low and "up" in low:
        smooth_send("L_ARM_UP")
    elif (("left arm" in low and "down" in low) or is_phrase_in_text(low, LEFT_ARM_DOWN_PHRASES)):
        smooth_send("L_ARM_DOWN")

    # RIGHT ARM
    if "right arm" in low and "up" in low:
        smooth_send("R_ARM_UP")
    elif "right arm" in low and "down" in low:
        smooth_send("R_ARM_DOWN")

    # HEAD
    if "head" in low and "left" in low:
        smooth_send("HEAD_LEFT")
    if "head" in low and "right" in low:
        smooth_send("HEAD_RIGHT")
    if "head" in low and "up" in low:
        smooth_send("HEAD_UP")
    if "head" in low and "down" in low:
        smooth_send("HEAD_DOWN")

    # EYES
    if "eyes" in low and "left" in low:
        smooth_send("EYES_LEFT")
    if "eyes" in low and "center" in low:
        smooth_send("EYES_CENTER")
    if "eyes" in low and "right" in low:
        smooth_send("EYES_RIGHT")

    # EYELIDS
    if "open" in low:
        smooth_send("EYELIDS_OPEN")
    if "close" in low or "shut" in low:
        smooth_send("EYELIDS_CLOSED")


# RESET & INITIALIZATION
def reset_robot() -> None:
    for c in ["L_ARM_MID", "R_ARM_MID", "EYES_CENTER", "EYELIDS_CLOSED", "HEAD_MID"]:
        smooth_send(c)
    log_local("Robot reset.")


# MAIN

def main() -> None:
    global _arduino
    _arduino = init_arduino()
    reset_robot()

    cap = None  # Start with no camera
    camera_active = False  # Start as False, don't auto-start
    voice_active = False  # Start as False, don't auto-start
    _last_pose_time = 0.0
    speech_thread = None  # Track the speech thread

    log_local("System ready.")

    try:
        while True:
            event, values = window.read(timeout=30)

            # flush logs to GUI
            while not log_q.empty():
                window["-LOG-"].print(log_q.get())

            if event in (sg.WIN_CLOSED, "Exit"):
                log_local("User closed program.")
                break

            # Speech -> process
            if voice_active and not speech_q.empty():
                text = speech_q.get()
                window["-SPEECH-"].update(text)
                low = text.lower()
                process_speech(low)

            # Manual Buttons
            valid_commands = [
                "L_ARM_UP", "L_ARM_MID", "L_ARM_DOWN",
                "R_ARM_UP", "R_ARM_MID", "R_ARM_DOWN",
                "HEAD_LEFT", "HEAD_MID", "HEAD_RIGHT", "HEAD_UP", "HEAD_DOWN",
                "EYES_LEFT", "EYES_CENTER", "EYES_RIGHT", "EYELIDS_OPEN", "EYELIDS_CLOSED"
            ]
            if event in valid_commands:
                smooth_send(event)
                log_local(f"Manual button → {event}")

            # Voice Toggle - START SPEECH THREAD HERE
            if event == "START_VOICE":
                voice_active = not voice_active
                if voice_active:
                    log_local("Voice control activated. Starting speech thread...")
                    speech_thread = threading.Thread(target=speech_thread_fn, daemon=True)
                    speech_thread.start()
                    window["START_VOICE"].update("Stop Voice Control")
                else:
                    log_local("Voice control deactivated.")
                    window["START_VOICE"].update("Start Voice Control")

            # Camera Pause/Resume
            if event == "Pause Camera":
                if cap:
                    cap.release()
                cap = None
                camera_active = False
                log_local("Camera paused.")

            if event == "Resume Camera":
                if not camera_active:
                    cap = open_camera()
                    camera_active = True if cap else False
                    log_local("Camera resumed.")

            # Reset robot
            if event == "Reset":
                reset_robot()
                log_local("Robot reset via GUI.")

            # Camera frame processing
            if camera_active and cap:
                
                ret, frame = cap.read()
                if not ret:
                    log_local("Frame read error.")
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                now = time.time()

                # POSE & FACE LOGIC with cooldown
                if (results.pose_landmarks or results.face_landmarks) and (now - _last_pose_time >= POSE_COOLDOWN):
                    if results.pose_landmarks:
                        pose = results.pose_landmarks.landmark
                        ls = pose[mp_holistic.PoseLandmark.LEFT_SHOULDER].y
                        lw = pose[mp_holistic.PoseLandmark.LEFT_WRIST].y
                        rs = pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y
                        rw = pose[mp_holistic.PoseLandmark.RIGHT_WRIST].y

                        if lw < ls - 0.05:
                            smooth_send("L_ARM_UP")
                        elif lw > ls + 0.05:
                            smooth_send("L_ARM_DOWN")
                        else:
                            smooth_send("L_ARM_MID")

                        if rw < rs - 0.05:
                            smooth_send("R_ARM_UP")
                        elif rw > rs + 0.05:
                            smooth_send("R_ARM_DOWN")
                        else:
                            smooth_send("R_ARM_MID")

                    if results.face_landmarks:
                        face = results.face_landmarks.landmark
                        Lopen = face[145].y - face[159].y
                        Ropen = face[374].y - face[386].y
                        if Lopen < 0.02 or Ropen < 0.02:
                            smooth_send("EYELIDS_CLOSED")
                        else:
                            smooth_send("EYELIDS_OPEN")

                        nose = face[1].y
                        eye_mid = (face[33].y + face[263].y) / 2
                        if nose < eye_mid - 0.02:
                            smooth_send("HEAD_UP")
                        elif nose > eye_mid + 0.02:
                            smooth_send("HEAD_DOWN")
                        else:
                            smooth_send("HEAD_MID")

                        nose_x = face[1].x
                        eye_mid_x = (face[33].x + face[263].x) / 2
                        if nose_x < eye_mid_x - 0.02:
                            smooth_send("HEAD_LEFT")
                        elif nose_x > eye_mid_x + 0.02:
                            smooth_send("HEAD_RIGHT")
                        else:
                            smooth_send("HEAD_MID")

                    _last_pose_time = now

                # draw and show
                frame = draw_selected(frame, results)
                imgbytes = cv2.imencode(".png", frame)[1].tobytes()
                window["-IMAGE-"].update(data=imgbytes)
                
    except Exception as e:
        log_local(f"MAIN ERROR: {e}")

    finally:
        # cleanup
        try:
            if cap:
                cap.release()
        except Exception:
            pass
        try:
            if _arduino and _arduino.is_open:
                _arduino.close()
        except Exception:
            pass
        try:
            holistic.close()
        except Exception:
            pass
        try:
            window.close()
        except Exception:
            pass

        log_local("Program closed.")
        # flush logs to stdout
        while not log_q.empty():
            print(log_q.get())
        # avoid raising SystemExit in some environments
        sys.exit(0)

if __name__ == "__main__":
    main()
