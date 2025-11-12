"""
Video Recording Module for Incar AI Assistant

Provides video recording functionality with threading support.
Designed to work with the smart camera integration system.
"""

import cv2
import threading
import time
from PyQt5.QtCore import QThread, pyqtSignal


class VideoRecorder:
    """
    Thread-based video recorder for capturing camera input.
    
    This class handles video recording with start/stop functionality
    and integrates with the main application's camera system.
    """
    
    def __init__(self, output_path='output.mp4'):
        """
        Initialize video recorder.
        
        Args:
            output_path (str): Path where the video will be saved
        """
        self.output_path = output_path
        self.stop_flag = False
        self.recording_thread = None
        self.is_recording = False
        
    def start_recording(self):
        """Start video recording in a separate thread."""
        if self.is_recording:
            print("Recording already in progress")
            return
            
        self.stop_flag = False
        self.recording_thread = threading.Thread(target=self._record_frames)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        self.is_recording = True
        print(f"Recording started: {self.output_path}")
        
    def stop_recording(self):
        """Stop video recording and wait for thread completion."""
        if not self.is_recording:
            print("No recording in progress")
            return
            
        self.stop_flag = True
        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)  # 5 second timeout
        self.is_recording = False
        print("Recording stopped")
        
    def _record_frames(self):
        """Internal method to handle frame capture and video writing."""
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Cannot access camera")
            return
            
        try:
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 20.0
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print("Error: Cannot create video writer")
                return
                
            frame_count = 0
            
            while cap.isOpened() and not self.stop_flag:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Cannot read frame from camera")
                    break
                    
                # Write frame to video file
                out.write(frame)
                frame_count += 1
                
                # Optional: Display frame (comment out for headless operation)
                # cv2.imshow('Recording', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                    
        except Exception as e:
            print(f"Recording error: {e}")
            
        finally:
            # Clean up resources
            cap.release()
            if 'out' in locals():
                out.release()
            cv2.destroyAllWindows()
            print(f"Recording saved: {self.output_path} ({frame_count} frames)")


class VideoRecorderQt(QThread):
    """
    Qt-based video recorder for integration with PyQt5 applications.
    
    Emits signals for status updates and can be used with Qt's
    event system for better GUI integration.
    """
    
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal()
    error_occurred = pyqtSignal(str)
    frame_count_updated = pyqtSignal(int)
    
    def __init__(self, output_path='output.mp4'):
        """
        Initialize Qt video recorder.
        
        Args:
            output_path (str): Path where the video will be saved
        """
        super().__init__()
        self.output_path = output_path
        self.stop_flag = False
        
    def set_output_path(self, path):
        """Set the output path for video recording."""
        self.output_path = path
        
    def stop_recording(self):
        """Signal the recording thread to stop."""
        self.stop_flag = True
        
    def run(self):
        """Main recording loop (runs in separate thread)."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            self.error_occurred.emit("Cannot access camera")
            return
            
        try:
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 20.0
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                self.error_occurred.emit("Cannot create video writer")
                return
                
            self.recording_started.emit()
            frame_count = 0
            
            while cap.isOpened() and not self.stop_flag:
                ret, frame = cap.read()
                
                if not ret:
                    self.error_occurred.emit("Cannot read frame from camera")
                    break
                    
                out.write(frame)
                frame_count += 1
                
                # Emit frame count every 30 frames (roughly every 1.5 seconds at 20fps)
                if frame_count % 30 == 0:
                    self.frame_count_updated.emit(frame_count)
                    
        except Exception as e:
            self.error_occurred.emit(f"Recording error: {str(e)}")
            
        finally:
            cap.release()
            if 'out' in locals():
                out.release()
            cv2.destroyAllWindows()
            self.recording_stopped.emit()


# Legacy function for backward compatibility
def Video_Recorder_p(pipe):
    """
    Legacy multiprocessing-based recorder (deprecated).
    
    Kept for backward compatibility. Use VideoRecorder class instead.
    """
    print("Warning: Using deprecated Video_Recorder_p function")
    recorder = VideoRecorder('./output_legacy.mp4')
    recorder.start_recording()
    
    try:
        msg = pipe.recv()  # Wait for stop message
        print(f"Received stop message: {msg}")
    except:
        print("Pipe communication failed")
    finally:
        recorder.stop_recording()


# Example usage and testing
if __name__ == '__main__':
    print("Testing Video Recorder")
    
    # Test basic recorder
    recorder = VideoRecorder('./test_output.mp4')
    recorder.start_recording()
    
    # Record for 5 seconds
    time.sleep(5)
    
    recorder.stop_recording()
    print("Test completed")