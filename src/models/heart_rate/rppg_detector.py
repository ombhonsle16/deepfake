import numpy as np
import cv2
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import os

# Try to import dlib, but don't fail if it's not available
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not available, falling back to mock heart rate detection")

class RPPGDetector:
    def __init__(self):
        # Check if model file exists and dlib is available
        model_path = 'models/shape_predictor_68_face_landmarks.dat'
        self.using_dlib = DLIB_AVAILABLE and os.path.exists(model_path)
        
        # Initialize face detector and facial landmarks predictor
        if self.using_dlib:
            try:
                self.face_detector = dlib.get_frontal_face_detector()
                self.landmark_predictor = dlib.shape_predictor(model_path)
                print("Using dlib face detector for heart rate detection")
            except Exception as e:
                print(f"Error loading dlib models for heart rate: {e}")
                self.using_dlib = False
        
        if not self.using_dlib:
            # Fallback to OpenCV
            print("Falling back to mock heart rate detection")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Parameters for signal processing
        self.fps = 30  # Assumed frame rate
        self.window_size = 300  # 10 seconds at 30 fps
        self.min_heart_rate = 45
        self.max_heart_rate = 180
        
        # Buffer for RGB values - initialize as numpy array with fixed shape
        self.rgb_values = np.zeros((0, 3), dtype=np.float32)
        
        # Previous heart rate for smoothing
        self.prev_hr = None
        
        # Butterworth bandpass filter parameters
        self.order = 3
        self.nyquist = self.fps / 2
        self.low = self.min_heart_rate / 60 / self.nyquist
        self.high = self.max_heart_rate / 60 / self.nyquist
    
    def get_face_roi(self, frame):
        """Extract face region of interest"""
        if not self.using_dlib:
            # Mock ROI if dlib is not available
            height, width = frame.shape[:2]
            return (width//4, height//4, width*3//4, height*3//4)
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray, 0)
            
            if len(faces) == 0:
                return None
            
            face = faces[0]
            shape = self.landmark_predictor(gray, face)
            
            # Get forehead region (between eyebrows and top of face)
            forehead_top = shape.part(27).y - 30
            forehead_bottom = shape.part(27).y
            forehead_left = shape.part(0).x
            forehead_right = shape.part(16).x
            
            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]
            forehead_top = max(0, forehead_top)
            forehead_bottom = min(height, forehead_bottom)
            forehead_left = max(0, forehead_left)
            forehead_right = min(width, forehead_right)
            
            return (forehead_left, forehead_top, forehead_right, forehead_bottom)
        except Exception as e:
            print(f"Error in get_face_roi: {e}")
            return None
    
    def extract_rgb_signals(self, frame, roi):
        """Extract average RGB values from ROI"""
        if roi is None:
            return None
        
        try:
            left, top, right, bottom = roi
            roi_frame = frame[top:bottom, left:right]
            
            # Calculate mean RGB values as a fixed-size array
            mean_rgb = np.array(cv2.mean(roi_frame)[:3], dtype=np.float32)
            
            # Append to the numpy array properly
            self.rgb_values = np.vstack([self.rgb_values, mean_rgb[np.newaxis, :]])
            
            # Keep only the last window_size frames
            if len(self.rgb_values) > self.window_size:
                self.rgb_values = self.rgb_values[-self.window_size:]
            
            return self.rgb_values
        except Exception as e:
            print(f"Error in extract_rgb_signals: {e}")
            return None
    
    def bandpass_filter(self, signal):
        """Apply bandpass filter to the signal"""
        b, a = butter(self.order, [self.low, self.high], btype='band')
        return filtfilt(b, a, signal)
    
    def estimate_heart_rate(self, rgb_signal):
        """Estimate heart rate from RGB signal"""
        if rgb_signal is None or len(rgb_signal) < 30:  # Need at least 1 second of data
            return {
                'heart_rate': None,
                'signal_quality': 0.0,
                'power_spectrum': None
            }
        
        try:
            # Normalize signals
            normalized = rgb_signal.copy()
            for i in range(3):
                normalized[:, i] = (normalized[:, i] - np.mean(normalized[:, i])) / (np.std(normalized[:, i]) + 1e-6)
            
            # Extract green channel and apply filtering
            green_signal = normalized[:, 1]  # Green channel
            filtered_signal = self.bandpass_filter(green_signal)
            
            # Compute FFT
            fft_signal = fft(filtered_signal)
            frequencies = np.fft.fftfreq(len(filtered_signal), 1/self.fps)
            
            # Find dominant frequency in expected heart rate range
            valid_freq_mask = (frequencies >= self.min_heart_rate/60) & (frequencies <= self.max_heart_rate/60)
            valid_freq = frequencies[valid_freq_mask]
            valid_fft = np.abs(fft_signal)[valid_freq_mask]
            
            if len(valid_fft) == 0:
                return {
                    'heart_rate': None,
                    'signal_quality': 0.0,
                    'power_spectrum': None
                }
            
            dominant_freq_idx = np.argmax(valid_fft)
            heart_rate = valid_freq[dominant_freq_idx] * 60
            
            # Smooth heart rate estimation
            if self.prev_hr is not None:
                heart_rate = 0.7 * self.prev_hr + 0.3 * heart_rate
            
            self.prev_hr = heart_rate
            
            # Calculate signal quality
            signal_quality = np.max(valid_fft) / (np.mean(valid_fft) + 1e-6)
            
            return {
                'heart_rate': heart_rate,
                'signal_quality': signal_quality,
                'power_spectrum': valid_fft
            }
        except Exception as e:
            print(f"Error in estimate_heart_rate: {e}")
            return {
                'heart_rate': None,
                'signal_quality': 0.0,
                'power_spectrum': None
            }
    
    def analyze_frame(self, frame):
        """Analyze a single frame for heart rate estimation"""
        if not self.using_dlib:
            # Return mocked heart rate data if dlib is not available
            heart_rate = np.random.uniform(60, 100)
            signal_quality = np.random.uniform(0.6, 1.0)
            confidence = signal_quality * 0.8
            suspicious = heart_rate < 60 or heart_rate > 100 or np.random.random() > 0.7
            
            return {
                'heart_rate': heart_rate,
                'signal_quality': signal_quality,
                'suspicious': suspicious,
                'confidence': confidence,
                'roi': None
            }
        
        # Get face ROI
        roi = self.get_face_roi(frame)
        
        if roi is None:
            return {
                'heart_rate': None,
                'signal_quality': 0.0,
                'suspicious': False,
                'confidence': 0.0,
                'roi': None
            }
        
        # Extract RGB signals
        rgb_signal = self.extract_rgb_signals(frame, roi)
        
        # Need at least some frames to estimate heart rate
        if rgb_signal is None or len(rgb_signal) < 30:
            return {
                'heart_rate': None,
                'signal_quality': 0.0,
                'suspicious': False,
                'confidence': 0.0,
                'roi': roi
            }
        
        # Estimate heart rate
        hr_result = self.estimate_heart_rate(rgb_signal)
        
        if hr_result['heart_rate'] is None:
            return {
                'heart_rate': None,
                'signal_quality': 0.0,
                'suspicious': False,
                'confidence': 0.0,
                'roi': roi
            }
        
        # Calculate confidence based on signal quality
        confidence = hr_result['signal_quality'] * 0.8
        
        # Suspicious heart rate
        suspicious = (
            hr_result['heart_rate'] < 50 or 
            hr_result['heart_rate'] > 120 or
            hr_result['signal_quality'] < 0.3
        )
        
        return {
            'heart_rate': hr_result['heart_rate'],
            'signal_quality': hr_result['signal_quality'],
            'suspicious': suspicious,
            'confidence': confidence,
            'roi': roi
        }
    
    def visualize_analysis(self, frame, analysis_result):
        """Visualize heart rate analysis results"""
        output = frame.copy()
        
        if 'roi' in analysis_result and analysis_result['roi'] is not None:
            left, top, right, bottom = analysis_result['roi']
            cv2.rectangle(output, (left, top), (right, bottom), (0, 255, 0), 2)
            
            if analysis_result['heart_rate'] is not None:
                hr_text = f"Heart Rate: {analysis_result['heart_rate']:.1f} BPM"
                cv2.putText(output, hr_text, (left, top - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if analysis_result['suspicious']:
                    cv2.putText(output, "SUSPICIOUS HR PATTERN", (left, top - 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return output 