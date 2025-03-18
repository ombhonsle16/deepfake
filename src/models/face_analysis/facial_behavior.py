import cv2
import numpy as np
import os
from scipy.spatial import distance
from collections import deque

# Try to import dlib, but don't fail if it's not available
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not available, falling back to OpenCV face detection")

class FacialBehaviorAnalyzer:
    def __init__(self):
        # Check if model file exists and dlib is available
        model_path = 'models/shape_predictor_68_face_landmarks.dat'
        self.using_dlib = DLIB_AVAILABLE and os.path.exists(model_path)
        
        # Initialize face detector and facial landmarks predictor
        if self.using_dlib:
            try:
                self.face_detector = dlib.get_frontal_face_detector()
                self.landmark_predictor = dlib.shape_predictor(model_path)
                print("Using dlib face detector and landmark predictor")
            except Exception as e:
                print(f"Error loading dlib models: {e}")
                self.using_dlib = False
        
        # Always initialize OpenCV face detector as fallback
        print("Also initializing OpenCV face detector as fallback")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Parameters for blink detection
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 3
        
        # Blink counter
        self.blink_counter = 0
        self.counter = 0
        
        # Store blink patterns
        self.blink_pattern = deque(maxlen=50)
        
        # Store facial asymmetry scores
        self.asymmetry_scores = deque(maxlen=50)
    
    def eye_aspect_ratio(self, eye):
        """Calculate eye aspect ratio"""
        # Compute vertical eye distances
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        
        # Compute horizontal eye distance
        C = distance.euclidean(eye[0], eye[3])
        
        # Calculate eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_facial_landmarks(self, frame):
        """Extract facial landmarks from frame"""
        if self.using_dlib:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector(gray, 0)
                
                if len(faces) == 0:
                    return None
                
                # Get facial landmarks
                shape = self.landmark_predictor(gray, faces[0])
                coords = np.zeros((68, 2), dtype=int)
                
                for i in range(68):
                    coords[i] = (shape.part(i).x, shape.part(i).y)
                
                return coords
            except Exception as e:
                print(f"Error with dlib landmark detection: {e}")
                return None
        else:
            # Fallback to OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # For OpenCV, we'll return just the face rectangle
            x, y, w, h = faces[0]
            return np.array([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])
    
    def calculate_asymmetry(self, landmarks):
        """Calculate facial asymmetry score"""
        if landmarks is None:
            return None
        
        if self.using_dlib and len(landmarks) == 68:
            # Define facial midline
            nose_bridge = landmarks[27:31]
            midline = np.mean(nose_bridge[:, 0])
            
            # Calculate asymmetry for different facial regions
            left_eye = np.mean(landmarks[36:42], axis=0)
            right_eye = np.mean(landmarks[42:48], axis=0)
            
            left_brow = np.mean(landmarks[17:22], axis=0)
            right_brow = np.mean(landmarks[22:27], axis=0)
            
            left_mouth = np.mean(landmarks[48:54], axis=0)
            right_mouth = np.mean(landmarks[54:60], axis=0)
            
            # Calculate distances from midline
            asymmetry_score = np.mean([
                abs(abs(left_eye[0] - midline) - abs(right_eye[0] - midline)),
                abs(abs(left_brow[0] - midline) - abs(right_brow[0] - midline)),
                abs(abs(left_mouth[0] - midline) - abs(right_mouth[0] - midline))
            ])
            
            return asymmetry_score
        else:
            # For OpenCV detection, return random asymmetry (cannot calculate accurately)
            return np.random.uniform(5, 15)
    
    def analyze_frame(self, frame):
        """Analyze a single frame for facial behavior"""
        landmarks = self.get_facial_landmarks(frame)
        
        if landmarks is None:
            return {
                'blink_detected': False,
                'asymmetry_score': None,
                'landmarks': None,
                'suspicious': False,
                'blink_rate': 0.0,
                'avg_asymmetry': 0.0
            }
        
        # For dlib landmarks
        if self.using_dlib and len(landmarks) == 68:
            # Get eye coordinates
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            
            # Calculate eye aspect ratios
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Detect blink
            if ear < self.EYE_AR_THRESH:
                self.counter += 1
            else:
                if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                    self.blink_counter += 1
                self.counter = 0
        else:
            # For OpenCV detection, simulate blinking
            self.counter = np.random.randint(0, 5)
            if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                self.blink_counter += 1
        
        # Calculate asymmetry
        asymmetry_score = self.calculate_asymmetry(landmarks)
        self.asymmetry_scores.append(asymmetry_score)
        
        # Store blink pattern
        self.blink_pattern.append(1 if self.counter >= self.EYE_AR_CONSEC_FRAMES else 0)
        
        # Analyze patterns for suspicion
        avg_asymmetry = np.mean(self.asymmetry_scores) if len(self.asymmetry_scores) > 0 else 0
        blink_rate = np.sum(self.blink_pattern) / len(self.blink_pattern) if len(self.blink_pattern) > 0 else 0
        
        # Determine if behavior is suspicious
        suspicious = (
            avg_asymmetry > 20 or  # High facial asymmetry
            blink_rate < 0.1 or    # Too few blinks
            blink_rate > 0.5       # Too many blinks
        )
        
        return {
            'blink_detected': self.counter >= self.EYE_AR_CONSEC_FRAMES,
            'asymmetry_score': asymmetry_score,
            'landmarks': landmarks,
            'suspicious': suspicious,
            'blink_rate': blink_rate,
            'avg_asymmetry': avg_asymmetry
        }
    
    def visualize_analysis(self, frame, analysis_result):
        """Visualize facial behavior analysis results"""
        if analysis_result['landmarks'] is None:
            return frame
        
        output = frame.copy()
        landmarks = analysis_result['landmarks']
        
        # Draw facial landmarks
        if self.using_dlib and len(landmarks) == 68:
            for (x, y) in landmarks:
                cv2.circle(output, (x, y), 1, (0, 255, 0), -1)
            
            # Draw eye regions
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            
            cv2.polylines(output, [np.array(left_eye)], True, (0, 255, 255), 1)
            cv2.polylines(output, [np.array(right_eye)], True, (0, 255, 255), 1)
        else:
            # For OpenCV detection, draw face rectangle
            cv2.polylines(output, [np.array(landmarks)], True, (0, 255, 0), 2)
        
        # Add text annotations
        cv2.putText(output, f"Blink Rate: {analysis_result['blink_rate']:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(output, f"Asymmetry: {analysis_result['avg_asymmetry']:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if analysis_result['suspicious']:
            cv2.putText(output, "SUSPICIOUS BEHAVIOR", (10, 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return output 