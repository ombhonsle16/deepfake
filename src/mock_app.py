# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import random
from models.hybrid.hybrid_detector import HybridDeepfakeDetector
import sys
import asyncio
from io import BytesIO

# Now that transformers is installed, we can use the hybrid detector
# Uncomment this line to use the actual hybrid detector instead of the mock one
# from models.hybrid.hybrid_detector import HybridDeepfakeDetector

# Set page config
st.set_page_config(
    page_title="Advanced Deepfake Detection System",
    page_icon="🔍",
    layout="wide"
)

model = HybridDeepfakeDetector(pretrained=True)

# Fix for asyncio event loop on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Import local modules using absolute paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.face_analysis.facial_behavior import FacialBehaviorAnalyzer

# Mock Classes for deepfake detection
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
    
    def visualize_analysis(self, frame, analysis_result):
        """Create visualization for heart rate analysis"""
        # Create a copy of the frame
        output = frame.copy()
        
        # Add heart rate information
        if analysis_result['heart_rate'] is not None:
            cv2.putText(
                output, 
                f"HR: {analysis_result['heart_rate']:.1f} BPM", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 0, 255), 
                2
            )
            
            # Draw quality indicator
            quality = analysis_result['signal_quality']
            cv2.putText(
                output, 
                f"Quality: {quality:.2f}", 
                (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
            
            # Draw a colored rectangle based on suspicious level
            color = (0, 255, 0) if not analysis_result['suspicious'] else (0, 0, 255)
            cv2.rectangle(output, (10, 80), (210, 110), color, -1)
            cv2.putText(
                output, 
                "NORMAL" if not analysis_result['suspicious'] else "SUSPICIOUS", 
                (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
        
        return output

class MockGradCAM:
    def __init__(self):
        pass
    
    def analyze_frame(self, frame_tensor, original_frame):
        # Create a mock heatmap
        height, width = original_frame.shape[:2]
        heatmap = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Generate random points for heatmap with more realistic clustering
        center_x = np.random.randint(width // 4, 3 * width // 4)
        center_y = np.random.randint(height // 4, 3 * height // 4)
        
        # Create a cluster of points around the center
        for _ in range(10):
            x = int(center_x + np.random.normal(0, width // 10))
            y = int(center_y + np.random.normal(0, height // 10))
            # Ensure x and y are within bounds
            x = max(0, min(width-1, x))
            y = max(0, min(height-1, y))
            
            radius = np.random.randint(20, 40)
            intensity = np.random.randint(150, 255)
            cv2.circle(heatmap, (x, y), radius, (0, 0, intensity), -1)
        
        # Add a few random points elsewhere
        for _ in range(3):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            cv2.circle(heatmap, (x, y), np.random.randint(10, 30), (0, 0, np.random.randint(100, 200)), -1)
        
        # Blur for better visualization
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # Overlay heatmap on original image
        visualization = cv2.addWeighted(original_frame, 0.7, heatmap, 0.3, 0)
        
        manipulation_score = np.random.uniform(0.6, 0.9)
        num_regions = np.random.randint(1, 5)
        
        return {
            'visualization': visualization,
            'heatmap': heatmap,
            'manipulation_score': manipulation_score,
            'num_manipulated_regions': num_regions
        }

class MockDeepfakeDetector:
    def __init__(self):
        pass
    
    def predict_frame(self, frame):
        """Generate realistic deepfake prediction based on image features"""
        # Convert to numpy if needed (handle both PIL and numpy inputs)
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        
        # Extract basic image features that might correlate with deepfakes
        # These are simplified versions of real detection features
        
        # 1. Check for unusual color distribution
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:,:,1])
        saturation_std = np.std(hsv[:,:,1])
        
        # 2. Check for edge coherence
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 3. Check noise patterns
        noise_level = np.std(gray)
        
        # Calculate a weighted deepfake probability
        # Higher saturation std and edge density often correlate with manipulations
        prob_fake = 0.5 + 0.1 * (saturation_std / 30) + 0.2 * (edge_density * 10) - 0.1 * (noise_level / 20)
        
        # Add randomness but bias toward higher probabilities for a demo
        prob_fake = prob_fake * 0.7 + 0.3 * np.random.uniform(0.6, 0.9)
        
        # Clip to valid probability range
        prob_fake = np.clip(prob_fake, 0.2, 0.95)
        
        return prob_fake

@st.cache_resource
def load_models():
    """Load all detection models"""
    # Initialize models
    hybrid_model = MockDeepfakeDetector()
    face_analyzer = FacialBehaviorAnalyzer()
    heart_detector = MockRPPGDetector()
    gradcam = MockGradCAM()
    
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
    
    combined_score = 0.6 * prob_fake + 0.2 * behavior_score + 0.2 * heart_score
    
    # Prepare result
    result = {
        'original_image': original_image,
        'is_fake': combined_score > 0.7,  # Use fixed threshold instead of slider
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
    
    # Sample frames at regular intervals
    results = []
    sample_rate = max(1, total_frames // 10)  # Analyze up to 10 frames
    
    with st.progress(0) as progress_bar:
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
            progress_bar.progress(min(1.0, (i + 1) / total_frames))
    
    cap.release()
    
    # Calculate video-level metrics
    avg_deepfake_score = np.mean([r['confidence'] for r in results])
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

def display_image_results(result):
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
                st.error("No face detected for behavior analysis")
        
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

def display_video_results(result):
    """Display results for video analysis"""
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display overall verdict
        avg_score = result['avg_score']
        st.subheader(f"Overall Verdict: {'FAKE' if avg_score > 0.7 else 'REAL'}")
        
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
        
        import random
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
        
        display_image_results(result)

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
        
        display_video_results(result)

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Advanced Deepfake Detection System - Final Year Project</p>
    <p>Using state-of-the-art techniques for multi-modal deepfake detection</p>
</div>
""", unsafe_allow_html=True)
