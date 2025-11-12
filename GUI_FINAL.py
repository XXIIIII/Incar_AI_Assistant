"""
Incar AI Assistant - Main GUI Application

This application provides an interactive AI assistant for in-car use with features including:
- Voice recognition and text-to-speech
- Personalized travel scheduling using LangChain and LLM
- Smart camera integration with Moment-DETR for landmark detection
- PyQt5-based graphical user interface
"""

# Standard library imports
import os
import sys
import time
import threading
import multiprocessing

# Third-party imports for GUI
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap

# Third-party imports for computer vision and media
import cv2
import sounddevice as sd
from scipy.io.wavfile import write

# Third-party imports for speech processing
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
from pydub import AudioSegment
from playsound import playsound

# Third-party imports for configuration
from dotenv import load_dotenv, find_dotenv

# Local imports
from Tutor.tutor import Tutor
from Camera.VideoRec import Video_Recorder
from run_on_video.run import slide_video_and_catch_hd

# Constants
SCALE_RATIO = 0.58  # Video height scaling ratio
AUDIO_FREQ = 44100  # Audio recording frequency
RECORD_DURATION = 5  # Recording duration in seconds
RESPOND_WAV_PATH = 'respond.wav'
SCHEDULE_MD_PATH = 'schedule.md'
TEMP_AUDIO_PATH = 'tmp.wav'
RECORD_INPUT_PATH = './record.wav'
VIDEO_OUTPUT_PATH = 'recoded2.mp4'

# GUI Constants
WINDOW_WIDTH = 1800
WINDOW_HEIGHT = 1200
BUTTON_WIDTH = 300
BUTTON_HEIGHT = 200
PROGRESS_BAR_HEIGHT = 60

# Button styling
BUTTON_STYLE = '''
    QPushButton{
        background-color: #fff;
        border: 2px solid #000;
        border-radius: 5px;
        font-weight: bold;
    }
    QPushButton:hover{
        background-color: #fa0;
        border: 2px solid #000;
        border-radius: 5px;
    }
'''

PROGRESS_BAR_STYLE = '''
    QProgressBar {
        border: 2px solid #000;
        border-radius: 5px;
        text-align:center;
        height: 20px;
        width:200px;
    }
    QProgressBar::chunk {
        background: #09c;
        width:1px;
    }
'''


class IncarAIAssistant(QtWidgets.QMainWindow):
    """
    Main window class for the Incar AI Assistant application.
    
    Handles the GUI interface, voice recording, speech recognition,
    AI response generation, and camera functionality.
    """
    
    def __init__(self):
        """Initialize the AI Assistant with all necessary components."""
        super(IncarAIAssistant, self).__init__()
        
        # Initialize communication variables
        self.asr_receiver = ['']  # Holds ASR results
        self.tutor_respond_t = [None]  # Holds AI tutor responses
        self.playsound_proc = None  # Process for playing audio
        
        # Initialize speech recognition
        self.speech_recognizer = sr.Recognizer()
        
        # Load environment variables and initialize AI tutor
        self.dotenv_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(self.dotenv_path, override=True)
        self.ai_agent = Tutor()
        
        # Initialize camera-related variables
        self.opencv_enabled = True
        self.is_recording = False
        self.video_codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_output = None
        
        # Initialize threading variables
        self.recorder_thread = None
        self.tutor_thread = None
        self.video_thread = None
        
        self.setup_ui()

    def record_and_recognize_speech(self, result_container):
        """
        Record audio from microphone and convert to text using speech recognition.
        
        Args:
            result_container (list): Container to store the recognition result
        """
        print("[Audio Recording] Starting recording...")
        
        try:
            # Record audio
            recording = sd.rec(
                int(RECORD_DURATION * AUDIO_FREQ), 
                samplerate=AUDIO_FREQ, 
                channels=2
            )
            sd.wait()
            write(RECORD_INPUT_PATH, AUDIO_FREQ, recording)
            
            # Convert audio format
            audio = AudioSegment.from_wav(RECORD_INPUT_PATH)
            audio.export(TEMP_AUDIO_PATH, format="wav")
            
            # Perform speech recognition
            audio_file = sr.AudioFile(TEMP_AUDIO_PATH)
            with audio_file as source:
                audio_data = self.speech_recognizer.record(source)
                recognition_result = self.speech_recognizer.recognize_google(
                    audio_data, language='zh-CN'
                )
            
            # Clean up temporary files
            if os.path.exists(RECORD_INPUT_PATH):
                os.remove(RECORD_INPUT_PATH)
            
            result_container[0] = recognition_result
            print(f"[Speech Recognition] Result: {recognition_result}")
            
        except Exception as e:
            print(f"[Error] Speech recognition failed: {e}")
            result_container[0] = ""

    def start_voice_interaction(self):
        """Start the voice interaction process."""
        self.recorder_thread = threading.Thread(
            target=self.record_and_recognize_speech, 
            args=(self.asr_receiver,)
        )
        self.recorder_thread.start()
        
        # Update UI
        self.start_button.hide()
        self.title_label.hide()
        self.status_label.setText("Start recording...")
        self.status_label.repaint()
        
        # Show recording animation
        self.loading_movie = QtGui.QMovie("loading.gif")
        self.loading_movie.setScaledSize(QtCore.QSize(300, 300))
        self.animation_label.setMovie(self.loading_movie)
        self.loading_movie.start()
        
        self.process_recording_result()

    def process_recording_result(self):
        """Process the result of voice recording and determine next action."""
        self.recorder_thread.join()
        self.status_label.setText("Recording finished...")
        print("[Processing] Recording completed")
        
        if self.asr_receiver and self._is_camera_command(self.asr_receiver[0]):
            print('[Mode] Camera activation requested')
            self._switch_to_camera_mode()
        else:
            print('[Mode] AI assistant query')
            self._process_ai_query()

    def _is_camera_command(self, text):
        """Check if the voice command is requesting camera activation."""
        camera_keywords = ['啓', '相', '機']
        return any(keyword in text for keyword in camera_keywords)

    def _switch_to_camera_mode(self):
        """Switch to camera recording mode."""
        self.camera_button.show()
        self.camera_label.show()
        self.start_button.hide()

    def _process_ai_query(self):
        """Process AI query and generate response."""
        self.video_thread = threading.Thread(target=self.run_opencv_camera)
        self.tutor_thread = threading.Thread(
            target=self.generate_ai_response, 
            args=(self.asr_receiver, self.tutor_respond_t)
        )
        self.tutor_thread.start()
        
        self.status_label.setText("Generating response...")
        self.status_label.repaint()
        print('[AI Tutor] Processing query...')
        self.update_progress_bar()

    def update_progress_bar(self):
        """Update progress bar with dynamic timing."""
        self.progress_bar.show()
        base_interval = int(5 * 1000 / 100)  # Base interval for progress updates
        increment = 100 / (5 * 1000 / base_interval)
        
        current_value = self.progress_bar.value() + increment
        self.progress_bar.setValue(min(int(current_value), 100))
        self.progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        self.progress_bar.setFormat('%p%')

        # Dynamic timing based on progress
        timing_map = {
            25: base_interval * 5,
            45: base_interval * 4,
            65: base_interval * 8,
            85: base_interval * 5,
            98: base_interval * 9
        }
        
        next_delay = None
        for threshold, delay in timing_map.items():
            if current_value < threshold:
                next_delay = delay
                break
        
        if next_delay:
            QtCore.QTimer.singleShot(int(next_delay), self.update_progress_bar)
        else:
            self._finalize_ai_response()

    def _finalize_ai_response(self):
        """Finalize AI response with TTS and UI updates."""
        self.tutor_thread.join()
        print('[TTS] Generating speech...')
        
        # Generate TTS audio
        tts_engine = gTTS(
            text=self.tutor_respond_t[0],
            lang="zh",
            slow=False
        )
        tts_engine.save(RESPOND_WAV_PATH)
        
        # Update progress and display response
        self.progress_bar.setValue(100)
        self.progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        self.progress_bar.setFormat('%p%')
        
        # Display response text
        self.status_label.setGeometry(QtCore.QRect(50, 150, WINDOW_WIDTH-200, 1000))
        self.status_label.setText(self.tutor_respond_t[0])
        self.status_label.setWordWrap(True)
        self.status_label.setFont(QtGui.QFont("Arial", 20))
        self.status_label.setAlignment(QtCore.Qt.AlignTop)
        self.status_label.repaint()

        # Play response audio
        playsound(RESPOND_WAV_PATH)
        QtCore.QTimer.singleShot(1000, lambda: self.home_button.show())

    def generate_ai_response(self, asr_text_container, response_container):
        """
        Generate AI response using the tutor agent.
        
        Args:
            asr_text_container (list): Container with ASR text
            response_container (list): Container to store AI response
        """
        user_input = asr_text_container[0]
        agent_response = self.ai_agent.invoke(user_input)
        response_container[0] = agent_response['speech']
        
        # Save schedule to file
        with open(SCHEDULE_MD_PATH, 'w') as f:
            f.write(agent_response['table'])
            print('[File Output] Schedule saved to markdown')

    def toggle_video_recording(self):
        """Toggle video recording on/off."""
        if not self.is_recording:
            # Start recording
            self.video_output = cv2.VideoWriter(
                VIDEO_OUTPUT_PATH, 
                self.video_codec, 
                20.0, 
                (WINDOW_WIDTH, int(WINDOW_WIDTH * SCALE_RATIO))
            )
            self.is_recording = True
            self.camera_button.setText('Stop Recording')
        else:
            # Stop recording
            self.video_output.release()
            self.is_recording = False
            self.camera_button.setText('Start Recording')
            slide_video_and_catch_hd(VIDEO_OUTPUT_PATH)

    def run_opencv_camera(self):
        """Run OpenCV camera capture and display."""
        print('[Camera] Initializing camera...')
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("[Error] Cannot open camera")
            return
            
        while self.opencv_enabled:
            ret, frame = camera.read()
            if not ret:
                print("[Error] Cannot receive frame")
                break
                
            # Resize frame to fit window
            frame = cv2.resize(frame, (WINDOW_WIDTH, int(WINDOW_WIDTH * SCALE_RATIO)))
            
            # Save frame if recording
            if self.is_recording and self.video_output:
                self.video_output.write(frame)
                
            # Convert and display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            bytes_per_line = channel * width
            q_image = QImage(frame_rgb, width, height, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(q_image))
        
        camera.release()

    def return_to_home(self):
        """Return to the home screen and reset UI state."""
        # Show/hide appropriate UI elements
        self.start_button.show()
        self.camera_button.hide()
        self.home_button.hide()
        self.status_label.hide()
        self.title_label.show()
        self.progress_bar.hide()
        self.progress_bar.setValue(0)

    def cleanup_on_close(self):
        """Clean up resources when closing the application."""
        self.opencv_enabled = False
        try:
            if self.video_output:
                self.video_output.release()
        except:
            pass

    def setup_ui(self):
        """Initialize and configure all UI elements."""
        # Main window configuration
        self.screen = QtWidgets.QApplication.desktop()
        self.setGeometry(200, 200, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setWindowTitle("Incar AI Assistant")
        self.closeEvent = self.cleanup_on_close

        # Center window on screen
        form_width = self.width()
        form_height = self.height()
        center_x = int((WINDOW_WIDTH - form_width) / 2)
        center_y = int((WINDOW_HEIGHT - form_height) / 2)
        self.move(center_x, center_y)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setGeometry(200, 800, 1400, PROGRESS_BAR_HEIGHT)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()

        # Status label
        self.status_label = QtWidgets.QLabel(self)
        self.status_label.setGeometry(QtCore.QRect(800, 400, 300, 400))
        self.status_label.setText("")

        # Title label with ASCII art
        self.title_label = QtWidgets.QLabel(self)
        self.title_label.setGeometry(QtCore.QRect(350, 20, 1115, 1000))
        self.title_label.setFont(QtGui.QFont("Arial", 10))
        self.title_label.setWordWrap(True)
        # self.label2.setContentsMargins(0,0,0,0)          # 設定邊界
        self.title_label.setText("明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明　　　明明明明明明明明明明明明明明明明明明　　　　　　　　　　明明明明明明明明明明　　　明明明明明明明明明明　　　　　　　　明　　　　　　　　　明明明明明明明明明明　　　明明明明明明明明明明　　　明明　　　明　　　明明明　　　明明明明明明明明明明　　　明明明明明明明明明明　　　明明　　　明　　　明明明　　　明明明明明明明明明明　　　明明明明明明明明明明　　　明明　　　明　　　明明明　　　明明明明明　　　　明　　　明　　　明明明明明明　　　明明　　　明　　　　　　　　　明明明明明　　　明明　　　明　　　　明明明明明　　　　　　　　明　　　明明明　　　明明明明　　　　明明　　　明明　　　明明明明明　　　明明　　　明　　　明明明　　　明明明明　　　明明明　　　明明　　　　明明明明　　　明明　　　明　　　明明明　　　明明明明　　　明明明　　　明明明　　　明明明明　　　明明　　　明　　　明明明　　　明明明　　　　明明明　　　明明明　　　　明明明　　　明明　　　　　　　　　　　　　明明明　　　明明明明　　　明明明明　　　明明明　　　　　　　　　　　明明明明　　　明明　　　　明明明明　　　明明明明　　　明明明　　　明明　　　　　　明明明明　　　明明　　　明明明明明　　　明明明明明明明明明明　　　明明　　　　　　明明明明　　　明明明明明明明明明明　　　明明明明明明明明明明　　　明明明明明　　　明明明明　　　明明明明明明明明明明　　　明明明明明明明明明明明明明明明明明　　　明明明明明　　　明明明明明明明　　　　　　明明明明明明明明明明明明明明明明　　　　明明　　　　　　明明明明明明明明　　　　明明明明明明明明明明明明明明明明明　　　明明明明　　　明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明明")

        # Animation label for loading GIF
        self.animation_label = QtWidgets.QLabel(self)
        self.animation_label.setGeometry(QtCore.QRect(0, 0, 300, 300))

        # Camera display label
        self.camera_label = QtWidgets.QLabel(self)
        self.camera_label.setGeometry(QtCore.QRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))
        self.camera_label.hide()

        # Start button
        self.start_button = QtWidgets.QPushButton(self)
        self.start_button.setGeometry(QtCore.QRect(900 - 150, 950, BUTTON_WIDTH, BUTTON_HEIGHT))
        self.start_button.setText("Start")
        self.start_button.setStyleSheet(BUTTON_STYLE)
        self.start_button.clicked.connect(self.start_voice_interaction)

        # Camera recording button
        self.camera_button = QtWidgets.QPushButton(self)
        self.camera_button.setGeometry(QtCore.QRect(900 - 150, 950, BUTTON_WIDTH, BUTTON_HEIGHT))
        self.camera_button.setText("Start Recording")
        self.camera_button.setStyleSheet(BUTTON_STYLE)
        self.camera_button.clicked.connect(self.toggle_video_recording)
        self.camera_button.hide()

        # Home button
        self.home_button = QtWidgets.QPushButton(self)
        self.home_button.setGeometry(QtCore.QRect(900 - 150, 950, BUTTON_WIDTH, BUTTON_HEIGHT))
        self.home_button.setText("Home")
        self.home_button.setStyleSheet(BUTTON_STYLE)
        self.home_button.clicked.connect(self.return_to_home)
        self.home_button.hide()

        # Start video thread
        self.video_thread = threading.Thread(target=self.run_opencv_camera)
        self.video_thread.start()


def main():
    """Main function to run the application."""
    app = QtWidgets.QApplication(sys.argv)
    assistant_window = IncarAIAssistant()
    assistant_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()