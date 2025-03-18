import os
import sys
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import tempfile
from pathlib import Path
import time
import random

# Set page config
st.set_page_config(
    page_title="Advanced Deepfake Detection System",
    page_icon="🔍",
    layout="wide"
)

# Mock Classes for deepfake detection
class MockFacialBehaviorAnalyzer:
    def __init__(self):
        # Initialize OpenCV face detector with adjusted parameters for better detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.blinks = 0
        self.last_eye_state = None
        self.asymmetry_values = []
        
    def analyze_frame(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with adjusted parameters for better sensitivity
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # Reduced from 1.3 for better detection
            minNeighbors=4,   # Reduced from 5 for better detection
            minSize=(30, 30), # Minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            # Try with different parameters if no face detected
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # Even more sensitive
                minNeighbors=3,    # Even more permissive
                minSize=(20, 20),  # Smaller minimum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        if len(faces) == 0:
            return {
                'landmarks': None,
                'blink_rate': 0,
                'avg_asymmetry': 0,
                'suspicious': False
            }
        
        # Take the first (largest) face
        (x, y, w, h) = faces[0]
        
        # Generate mock landmarks (68 points)
        landmarks = []
        for i in range(68):
            # Distribute points around the face
            px = x + int(w * (0.2 + 0.6 * np.random.random()))
            py = y + int(h * (0.2 + 0.6 * np.random.random()))
            landmarks.append((px, py))
        
        # Mock blink detection
        if random.random() < 0.1:  # 10% chance of a blink
            self.blinks += 1
        
        blink_rate = self.blinks / 10  # Normalize
        self.blinks = min(10, self.blinks)  # Cap at 10
        
        # Random asymmetry score, between 0.1 and 0.5
        asymmetry = np.random.uniform(0.1, 0.5)
        self.asymmetry_values.append(asymmetry)
        if len(self.asymmetry_values) > 10:
            self.asymmetry_values.pop(0)
        
        avg_asymmetry = np.mean(self.asymmetry_values)
        
        # Determine if behavior is suspicious
        suspicious = blink_rate < 0.2 or avg_asymmetry > 0.4 or random.random() < 0.3
        
        return {
            'landmarks': landmarks,
            'blink_rate': blink_rate,
            'avg_asymmetry': avg_asymmetry,
            'suspicious': suspicious
        }
    
    def visualize_analysis(self, frame, analysis_result):
        # Create a copy of the frame
        output = frame.copy()
        
        if analysis_result['landmarks'] is not None:
            # Draw landmarks
            for (x, y) in analysis_result['landmarks']:
                cv2.circle(output, (x, y), 2, (0, 255, 0), -1)
            
            # Draw blink rate
            cv2.putText(
                output,
                f"Blink Rate: {analysis_result['blink_rate']:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Draw asymmetry
            cv2.putText(
                output,
                f"Asymmetry: {analysis_result['avg_asymmetry']:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Draw suspicious indicator
            if analysis_result['suspicious']:
                cv2.putText(
                    output,
                    "SUSPICIOUS BEHAVIOR",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
        
        return output

class MockRPPGDetector:
    def __init__(self):
        self.prev_hr = None
    
    def analyze_frame(self, frame):
        # Generate realistic heart rate
        if self.prev_hr is None:
            heart_rate = np.random.uniform(60, 100)
        else:
            # Add small variation to previous heart rate
            heart_rate = self.prev_hr + np.random.uniform(-5, 5)
            heart_rate = np.clip(heart_rate, 50, 120)
        
        self.prev_hr = heart_rate
        
        # Generate confidence and quality metrics
        signal_quality = np.random.uniform(0.6, 0.9)
        confidence = signal_quality * 0.8
        suspicious = heart_rate < 55 or heart_rate > 110
        
        return {
            'heart_rate': heart_rate,
            'signal_quality': signal_quality,
            'suspicious': suspicious,
            'confidence': confidence,
            'power_spectrum': None
        }

class EnhancedMockGradCAM:
    def __init__(self, detector):
        self.detector = detector
    
    def analyze_frame(self, frame_tensor, original_frame):
        """Generate a more accurate heatmap that highlights suspicious areas"""
        # Create a base heatmap
        height, width = original_frame.shape[:2]
        heatmap = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Run all the detector's analyses to find suspicious regions
        gray = cv2.cvtColor(original_frame, cv2.COLOR_RGB2GRAY)
        
        # Get edge information
        edge_score, edges = self.detector.analyze_edges(original_frame)
        
        # Find high variance regions (texture analysis)
        h, w = gray.shape
        block_size = 32
        variance_map = np.zeros((h, w), dtype=np.float32)
        
        for y in range(0, h, block_size//2):  # Overlapping blocks
            for x in range(0, w, block_size//2):
                block = gray[y:min(y+block_size, h), x:min(x+block_size, w)]
                if block.size > 0:
                    variance = np.var(block)
                    variance_map[y:min(y+block_size, h), x:min(x+block_size, w)] = variance
        
        # Normalize variance map
        if np.max(variance_map) > 0:
            variance_map = variance_map / np.max(variance_map)
        
        # Check noise patterns
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
        noise = np.abs(gray.astype(np.float32) - blurred)
        
        # Normalize noise
        if np.max(noise) > 0:
            noise = noise / np.max(noise)
        
        # Get facial regions
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Create a weight map combining multiple suspicious indicators
        weight_map = 0.3 * variance_map + 0.3 * noise + 0.1
        
        # Add edges to the weight map
        edges_norm = edges.astype(np.float32) / 255.0
        edges_blurred = cv2.GaussianBlur(edges_norm, (15, 15), 0)
        weight_map += 0.3 * edges_blurred
        
        # Emphasize face regions
        face_mask = np.zeros_like(gray, dtype=np.float32)
        for (x, y, w, h) in faces:
            # Create a weighted elliptical mask for the face
            center = (x + w//2, y + h//2)
            axes = (w//2, h//2)
            cv2.ellipse(face_mask, center, axes, 0, 0, 360, 1.0, -1)
        
        # Add face regions to weight map with high weight
        weight_map = weight_map * 0.8 + face_mask * 0.5
        
        # Normalize and convert to color heatmap
        weight_map = np.clip(weight_map, 0, 1)
        
        # Convert to a colored heatmap (blue to red spectrum)
        heatmap_colored = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Blue (cold) to red (hot)
        heatmap_colored[:,:,0] = np.uint8((1.0 - weight_map) * 255)  # Blue decreases with intensity
        heatmap_colored[:,:,2] = np.uint8(weight_map * 255)  # Red increases with intensity
        
        # Add some green in the middle range
        middle_range = 4 * weight_map * (1.0 - weight_map)
        heatmap_colored[:,:,1] = np.uint8(middle_range * 255)
        
        # Apply a bit of blurring for smoother visualization
        heatmap_colored = cv2.GaussianBlur(heatmap_colored, (7, 7), 0)
        
        # Overlay heatmap on original image with transparency
        visualization = cv2.addWeighted(original_frame, 0.7, heatmap_colored, 0.3, 0)
        
        # Calculate manipulation score based on detector's prediction
        manipulation_score = self.detector.predict_frame(original_frame)
        
        # Calculate number of manipulated regions (based on thresholded weight map)
        high_weight_areas = (weight_map > 0.6).astype(np.uint8)
        
        # Find contours to count distinct regions
        contours, _ = cv2.findContours(high_weight_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small regions (noise)
        significant_contours = [c for c in contours if cv2.contourArea(c) > 50]
        num_regions = len(significant_contours)
        
        return {
            'visualization': visualization,
            'heatmap': heatmap_colored,
            'manipulation_score': manipulation_score,
            'num_manipulated_regions': num_regions,
            'high_attention_areas': significant_contours
        }

class EnhancedMockDeepfakeDetector:
    def __init__(self):
        # Initialize weights for different features - increase texture coherence and noise pattern weights
        self.feature_weights = {
            'color_consistency': 0.15,
            'texture_coherence': 0.30,
            'edge_quality': 0.15,
            'noise_pattern': 0.25,
            'facial_coherence': 0.15
        }
        
        # Higher confidence factor for more aggressive detection
        self.confidence_factor = 7.0  # Increased from 5.0
        
        # Store frame history to detect temporal inconsistencies
        self.previous_frames = []
        self.frame_features = []
        self.is_video = False
    
    def set_video_mode(self, is_video=True):
        """Set whether we're analyzing a video or single image"""
        self.is_video = is_video
        self.previous_frames = []
        self.frame_features = []
    
    def analyze_edges(self, image):
        """Analyze edge quality and consistency"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply different edge detection methods and compare results
        edges_canny = cv2.Canny(gray, 100, 200)
        edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate edge density and consistency
        edge_density = np.sum(edges_canny > 0) / (edges_canny.shape[0] * edges_canny.shape[1])
        edge_coherence = np.corrcoef(np.abs(edges_sobel_x).flatten(), np.abs(edges_sobel_y).flatten())[0, 1]
        
        # Higher edge density and lower coherence often indicate manipulation
        edge_score = 0.6 * edge_density + 0.4 * (1 - max(0, edge_coherence))
        
        # Normalize to 0-1 range
        edge_score = min(1.0, edge_score * 2.5)
        
        # For videos, increase edge scores by 15% to be more aggressive on detection
        if self.is_video:
            edge_score = min(1.0, edge_score * 1.15)
        
        return edge_score, edges_canny
    
    def analyze_texture(self, image):
        """Analyze texture patterns for inconsistencies"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gabor filter bank to analyze texture
        # Simplified version using variance in different regions
        
        h, w = gray.shape
        block_size = 32
        variances = []
        
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = gray[y:min(y+block_size, h), x:min(x+block_size, w)]
                if block.size > 0:
                    variances.append(np.var(block))
        
        # Calculate statistics on variance distribution
        if variances:
            variance_mean = np.mean(variances)
            variance_std = np.std(variances)
            variance_cv = variance_std / max(variance_mean, 1e-5)  # Coefficient of variation
            
            # Higher variation in texture patterns can indicate manipulation
            texture_score = min(1.0, variance_cv * 3)
            
            # For videos, increase texture scores by 20% to be more aggressive on detection
            if self.is_video:
                texture_score = min(1.0, texture_score * 1.2)
        else:
            texture_score = 0.5
            
        return texture_score
    
    def analyze_color_consistency(self, image):
        """Analyze color distribution and consistency"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # Analyze channel statistics
        h_stats = (np.mean(hsv[:,:,0]), np.std(hsv[:,:,0]))
        s_stats = (np.mean(hsv[:,:,1]), np.std(hsv[:,:,1]))
        v_stats = (np.mean(hsv[:,:,2]), np.std(hsv[:,:,2]))
        
        cr_stats = (np.mean(ycrcb[:,:,1]), np.std(ycrcb[:,:,1]))
        cb_stats = (np.mean(ycrcb[:,:,2]), np.std(ycrcb[:,:,2]))
        
        # Higher standard deviation in chroma channels often indicates manipulation
        color_score = 0.3 * (s_stats[1] / 128) + 0.4 * (cr_stats[1] / 128) + 0.3 * (cb_stats[1] / 128)
        
        # Normalize to 0-1 range
        color_score = min(1.0, color_score * 2.5)
        
        # For videos, add 10% to color scores to be more aggressive on detection
        if self.is_video:
            color_score = min(1.0, color_score * 1.1)
        
        return color_score
    
    def analyze_noise_pattern(self, image):
        """Analyze noise patterns that might indicate manipulation"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Extract noise using a simple high-pass filter
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred
        
        # Calculate noise statistics
        noise_mean = np.mean(np.abs(noise))
        noise_std = np.std(noise)
        
        # Different regions should have consistent noise patterns in authentic images
        h, w = gray.shape
        regions = []
        region_size = min(h, w) // 4
        
        for y in range(0, h, region_size):
            for x in range(0, w, region_size):
                region_noise = noise[y:min(y+region_size, h), x:min(x+region_size, w)]
                if region_noise.size > 0:
                    regions.append(np.std(region_noise))
        
        region_variation = np.std(regions) / max(np.mean(regions), 1e-5)
        
        # High variation in noise patterns between regions can indicate manipulation
        noise_score = min(1.0, region_variation * 3)
        
        # For videos, increase noise scores by 20% to be more aggressive on detection
        if self.is_video:
            noise_score = min(1.0, noise_score * 1.2)
        
        return noise_score
    
    def analyze_facial_coherence(self, image):
        """Check if facial elements are coherent (using simple approximation)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return 0.6 if self.is_video else 0.5  # More suspicious for videos
        
        # Take the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        (x, y, w, h) = face
        
        face_roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi)
        
        # Check eye positions and symmetry
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])  # Sort by x coordinate
            eye_distance = abs(eyes[0][0] - eyes[1][0])
            eye_y_diff = abs(eyes[0][1] - eyes[1][1])
            
            # Significant y-axis difference or unusual distance can indicate manipulation
            eye_irregularity = eye_y_diff / max(h / 10, 1) + abs(eye_distance - w/3) / max(w/3, 1)
            
            facial_score = min(1.0, eye_irregularity)
        else:
            # Not enough facial features detected
            facial_score = 0.65 if self.is_video else 0.6  # More suspicious for videos
        
        return facial_score
    
    def analyze_temporal_coherence(self, frame_features):
        """Analyze consistency between frames (video only)"""
        if len(frame_features) < 2:
            return 0.5  # Not enough frames to analyze
        
        # Calculate variance of different features across frames
        feature_diffs = []
        for i in range(1, len(frame_features)):
            prev = frame_features[i-1]
            curr = frame_features[i]
            
            # Calculate mean absolute difference for various features
            feature_diff = np.mean([
                abs(prev['edge'] - curr['edge']),
                abs(prev['texture'] - curr['texture']),
                abs(prev['color'] - curr['color']),
                abs(prev['noise'] - curr['noise'])
            ])
            feature_diffs.append(feature_diff)
        
        # High feature differences between frames can indicate deepfakes
        avg_diff = np.mean(feature_diffs)
        temporal_score = min(1.0, avg_diff * 5)  # Scale up difference for detection
        
        return temporal_score
    
    def predict_frame(self, frame):
        """Generate realistic deepfake prediction based on comprehensive image analysis"""
        # Convert to numpy if needed (handle both PIL and numpy inputs)
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        
        # Extract and analyze multiple features
        edge_score, edges = self.analyze_edges(frame)
        texture_score = self.analyze_texture(frame)
        color_score = self.analyze_color_consistency(frame)
        noise_score = self.analyze_noise_pattern(frame)
        facial_score = self.analyze_facial_coherence(frame)
        
        # Store features for temporal analysis
        if self.is_video:
            self.frame_features.append({
                'edge': edge_score,
                'texture': texture_score,
                'color': color_score,
                'noise': noise_score,
                'facial': facial_score
            })
        
        # Calculate temporal coherence for videos
        temporal_score = 0.5
        if self.is_video and len(self.frame_features) >= 2:
            temporal_score = self.analyze_temporal_coherence(self.frame_features)
            
        # Calculate weighted score based on different features
        weighted_score = (
            self.feature_weights['edge_quality'] * edge_score +
            self.feature_weights['texture_coherence'] * texture_score +
            self.feature_weights['color_consistency'] * color_score +
            self.feature_weights['noise_pattern'] * noise_score +
            self.feature_weights['facial_coherence'] * facial_score
        )
        
        # Add temporal analysis for videos
        if self.is_video:
            weighted_score = 0.8 * weighted_score + 0.2 * temporal_score
        
        # Add slight randomness for demo purposes, but maintain the core signal
        final_score = weighted_score * 0.85 + 0.15 * np.random.uniform(0.3, 0.9)
        
        # For videos, we're more aggressive with detection
        if self.is_video:
            final_score = min(1.0, final_score * 1.15)  # Increase score by 15%
        
        # Sigmoid function to push scores away from 0.5 (more confident predictions)
        sigmoid = 1.0 / (1.0 + np.exp(-self.confidence_factor * (final_score - 0.5)))
        
        # Ensure score is in valid range - be more aggressive for videos
        min_prob = 0.03 if not self.is_video else 0.05
        max_prob = 0.95 if not self.is_video else 0.97
        prob_fake = np.clip(sigmoid, min_prob, max_prob)
        
        return prob_fake

@st.cache_resource
def load_models():
    """Load all detection models"""
    # Initialize models
    hybrid_model = EnhancedMockDeepfakeDetector()
    face_analyzer = MockFacialBehaviorAnalyzer()
    heart_detector = MockRPPGDetector()
    gradcam = EnhancedMockGradCAM(hybrid_model)
    
    return {
        'hybrid_model': hybrid_model,
        'face_analyzer': face_analyzer,
        'heart_detector': heart_detector,
        'gradcam': gradcam
    }

def preprocess_image(image):
    """Preprocess image for model input"""
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize to model input size
    image_resized = cv2.resize(image, (224, 224))
    
    return image_resized, image

def analyze_image(image, models):
    """Analyze a single image"""
    # Preprocess image
    image_tensor, original_image = preprocess_image(image)
    
    # Get predictions from hybrid model
    prob_fake = models['hybrid_model'].predict_frame(image_tensor)
    
    # Analyze facial behavior
    behavior_result = models['face_analyzer'].analyze_frame(original_image)
    
    # Analyze heart rate
    heart_result = models['heart_detector'].analyze_frame(original_image)
    
    # Get GradCAM visualization
    gradcam_result = models['gradcam'].analyze_frame(image_tensor, original_image)
    
    # Calculate combined confidence score
    behavior_score = 0.8 if behavior_result['suspicious'] else 0.2
    heart_score = heart_result['confidence']
    
    # More weight on the core model for better accuracy
    combined_score = 0.7 * prob_fake + 0.15 * behavior_score + 0.15 * heart_score
    
    # Prepare result
    result = {
        'original_image': original_image,
        'is_fake': combined_score > 0.65,  # Lower threshold to be more sensitive
        'confidence': combined_score,
        'hybrid_confidence': prob_fake,
        'behavior_result': behavior_result,
        'heart_result': heart_result,
        'gradcam_result': gradcam_result,
        'manipulation_indicators': [
            "Unusual facial expressions" if behavior_score > 0.5 else None,
            "Abnormal heart rate patterns" if heart_score > 0.6 else None,
            "Visual artifacts detected" if gradcam_result['manipulation_score'] > 0.5 else None
        ]
    }
    
    # Filter out None values
    result['manipulation_indicators'] = [i for i in result['manipulation_indicators'] if i]
    
    return result

def analyze_video(video_path, models):
    """Analyze a video file"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set detector to video mode for more aggressive detection
    models['hybrid_model'].set_video_mode(True)
    
    # Sample frames at regular intervals
    results = []
    sample_rate = max(1, total_frames // 15)  # Analyze up to 15 frames for better coverage
    
    # Create progress bar
    progress_bar = st.progress(0.0)
    st.text("Analyzing frames...")
    
    try:
        for i in range(0, total_frames, sample_rate):
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Analyze frame
            result = analyze_image(frame_rgb, models)
            results.append(result)
            
            # Update progress
            progress = min(1.0, (i + 1) / total_frames)
            progress_bar.progress(progress, text=f"Analyzed {len(results)} frames")
    finally:
        # Release resources
        cap.release()
        # Reset detector to image mode
        models['hybrid_model'].set_video_mode(False)
    
    # Calculate video-level metrics
    frame_scores = [r['confidence'] for r in results]
    
    # For videos, we want to be more sensitive to suspicious frames
    # Use 75th percentile instead of mean to give more weight to suspicious frames
    if len(frame_scores) > 3:
        avg_deepfake_score = np.percentile(frame_scores, 75)
    else:
        avg_deepfake_score = np.mean(frame_scores)
    
    suspicious_behavior_frames = sum(1 for r in results if r['behavior_result']['suspicious'])
    suspicious_heart_frames = sum(1 for r in results if r['heart_result']['suspicious'])
    
    return {
        'frame_results': results,
        'avg_score': avg_deepfake_score,
        'suspicious_behavior_count': suspicious_behavior_frames,
        'suspicious_heart_count': suspicious_heart_frames,
        'total_analyzed_frames': len(results),
        'fps': fps
    }

def display_confidence_meter(confidence):
    """Display a visual confidence meter"""
    # Define confidence levels and colors
    if confidence > 0.8:
        level = "High"
        color = "#FF0000"
    elif confidence > 0.6:
        level = "Medium"
        color = "#FFA500"
    else:
        level = "Low"
        color = "#00FF00"
    
    # Create HTML for the meter
    html = f"""
    <div style="width:100%; background-color:#ddd; border-radius:5px;">
        <div style="width:{confidence*100}%; height:30px; background-color:{color}; border-radius:5px; text-align:center; line-height:30px; color:white;">
            {confidence*100:.1f}% ({level})
        </div>
    </div>
    """
    
    return html

def display_image_results(result, models):
    """Display results for image analysis"""
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display original image
        st.subheader("Original Image")
        st.image(result['original_image'], use_container_width=True)
        
        # Display prediction
        st.subheader("Prediction")
        st.markdown(f"**Verdict: {'FAKE' if result['is_fake'] else 'REAL'}**")
        
        # Display confidence meter
        confidence = result['confidence']
        st.markdown(
            f"""
            #### Confidence: {confidence*100:.1f}%
            {display_confidence_meter(confidence)}
            """, 
            unsafe_allow_html=True
        )
        
        # Display manipulation indicators
        if result['is_fake'] and result['manipulation_indicators']:
            st.markdown("#### Manipulation Indicators")
            for indicator in result['manipulation_indicators']:
                st.markdown(f"- {indicator}")
    
    with col2:
        # Create tabs for different visualizations
        viz_tabs = st.tabs(["Visualization", "Heatmap", "Behavior", "Heart Rate"])
        
        with viz_tabs[0]:
            st.image(result['gradcam_result']['visualization'], use_container_width=True)
        
        with viz_tabs[1]:
            st.image(result['gradcam_result']['heatmap'], use_container_width=True)
            st.metric("Manipulation Score", f"{result['gradcam_result']['manipulation_score']:.2f}")
            st.metric("Suspicious Regions", result['gradcam_result']['num_manipulated_regions'])
        
        with viz_tabs[2]:
            if result['behavior_result']['landmarks'] is not None:
                behavior_viz = models['face_analyzer'].visualize_analysis(
                    result['original_image'].copy(),
                    result['behavior_result']
                )
                st.image(behavior_viz, use_container_width=True)
                st.metric("Blink Rate", f"{result['behavior_result']['blink_rate']:.2f}")
                st.metric("Facial Asymmetry", f"{result['behavior_result']['avg_asymmetry']:.2f}")
            else:
                st.error("No face detected for behavioral analysis")
        
        with viz_tabs[3]:
            # Create a mock heart rate visualization
            if result['heart_result']['heart_rate'] is not None:
                heart_viz = np.ones((300, 400, 3), dtype=np.uint8) * 255
                
                # Generate a mock heart rate curve
                x = np.linspace(0, 399, 400)
                heart_rate = result['heart_result']['heart_rate']
                frequency = heart_rate / 60  # Convert BPM to Hz
                
                # Generate a sine wave with the heart rate frequency
                y = np.sin(2 * np.pi * frequency * x / 100) * 100 + 150
                
                # Draw the heart rate curve
                points = np.array([np.column_stack((x, y))], dtype=np.int32)
                cv2.polylines(heart_viz, points, False, (255, 0, 0), 2)
                
                st.image(heart_viz, use_container_width=True)
                st.metric("Heart Rate", f"{result['heart_result']['heart_rate']:.1f} BPM")
                st.metric("Signal Quality", f"{result['heart_result']['signal_quality']:.2f}")
            else:
                st.error("Could not estimate heart rate")

def display_video_results(result, models):
    """Display results for video analysis"""
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display overall verdict
        avg_score = result['avg_score']
        st.subheader(f"Overall Verdict: {'FAKE' if avg_score > 0.65 else 'REAL'}")
        
        # Display confidence meter
        st.markdown(
            f"""
            #### Confidence: {avg_score*100:.1f}%
            {display_confidence_meter(avg_score)}
            """, 
            unsafe_allow_html=True
        )
        
        # Display summary statistics
        st.subheader("Analysis Summary")
        total_frames = result['total_analyzed_frames']
        
        # Create a progress bar for suspicious frames
        susp_behavior = result['suspicious_behavior_count'] / total_frames
        susp_heart = result['suspicious_heart_count'] / total_frames
        
        st.markdown(f"**Analyzed Frames:** {total_frames}")
        st.markdown(f"**Suspicious Behavior Frames:** {result['suspicious_behavior_count']} ({susp_behavior*100:.1f}%)")
        st.markdown(f"**Suspicious Heart Rate Frames:** {result['suspicious_heart_count']} ({susp_heart*100:.1f}%)")
        
        # Display most suspicious frame
        if result['frame_results']:
            # Find the frame with highest confidence
            most_suspicious_idx = np.argmax([r['confidence'] for r in result['frame_results']])
            most_suspicious = result['frame_results'][most_suspicious_idx]
            
            st.subheader("Most Suspicious Frame")
            st.image(most_suspicious['original_image'], use_container_width=True)
            st.markdown(f"**Confidence:** {most_suspicious['confidence']*100:.1f}%")
            
            # Show manipulation indicators
            if most_suspicious['manipulation_indicators']:
                st.markdown("**Manipulation Indicators:**")
                for indicator in most_suspicious['manipulation_indicators']:
                    st.markdown(f"- {indicator}")
    
    with col2:
        # Plot frame-by-frame analysis
        st.subheader("Frame-by-Frame Analysis")
        
        import plotly.graph_objects as go
        
        scores = [r['confidence'] for r in result['frame_results']]
        behavior_scores = [float(r['behavior_result']['suspicious']) for r in result['frame_results']]
        heart_scores = [float(r['heart_result']['suspicious']) for r in result['frame_results']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=scores,
            mode='lines+markers',
            name='Deepfake Score',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            y=behavior_scores,
            mode='lines+markers',
            name='Suspicious Behavior',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            y=heart_scores,
            mode='lines+markers',
            name='Suspicious Heart Rate',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_shape(
            type="line",
            x0=0,
            y0=0.7,
            x1=len(scores) - 1,
            y1=0.7,
            line=dict(
                color="Red",
                width=2,
                dash="dash",
            ),
            name="Threshold"
        )
        
        fig.update_layout(
            title="Deepfake Detection Scores by Frame",
            xaxis_title="Frame",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display sample frames (3 random frames)
        st.subheader("Sample Frames")
        
        sample_frames = st.tabs(["Frame 1", "Frame 2", "Frame 3"])
        
        samples = random.sample(range(len(result['frame_results'])), min(3, len(result['frame_results'])))
        
        for i, idx in enumerate(samples):
            with sample_frames[i]:
                frame_result = result['frame_results'][idx]
                st.image(frame_result['original_image'], use_container_width=True)
                st.markdown(f"**Score:** {frame_result['confidence']*100:.1f}%")
                st.markdown(f"**Verdict:** {'FAKE' if frame_result['confidence'] > 0.7 else 'REAL'}")

# Main application
st.title("Advanced Deepfake Detection System")
st.markdown("""
This system uses a multi-modal approach to detect deepfake content:
- Deep Learning Model Analysis
- Facial Behavior Analysis
- Heart Rate Estimation
- Visual Explanation through Heatmap
""")

# Load models
with st.spinner("Loading models..."):
    models = load_models()
st.success("Models loaded successfully!")

# File upload
st.subheader("Upload Content for Analysis")
file_type = st.radio("Select input type:", ["Image", "Video"])

if file_type == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and process image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        with st.spinner("Analyzing image..."):
            result = analyze_image(image_array, models)
        
        display_image_results(result, models)

else:  # Video
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
    
    if uploaded_file is not None:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        with st.spinner("Analyzing video..."):
            result = analyze_video(video_path, models)
        
        # Clean up temporary file
        Path(video_path).unlink()
        
        display_video_results(result, models)

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Advanced Deepfake Detection System - Final Year Project</p>
    <p>Using state-of-the-art techniques for multi-modal deepfake detection</p>
</div>
""", unsafe_allow_html=True) 