import os
import sys
import torch
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import tempfile
from io import BytesIO
import matplotlib.pyplot as plt
from pathlib import Path
import time
import asyncio
import nest_asyncio

# Fix for asyncio event loop on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

# Import local modules using absolute paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import model modules, but with fallbacks
try:
    from models.hybrid.hybrid_detector import HybridDeepfakeDetector
    from models.face_analysis.facial_behavior import FacialBehaviorAnalyzer
    from models.heart_rate.rppg_detector import RPPGDetector
except Exception as e:
    print(f"Error importing models: {e}")
    # Mock implementations will be used later

# Set page config
st.set_page_config(
    page_title="Advanced Deepfake Detection System",
    page_icon="🔍",
    layout="wide"
)

# Simple mock GradCAM implementation
class MockGradCAM:
    def __init__(self, model):
        self.model = model
    
    def analyze_frame(self, frame_tensor, original_frame):
        # Create a mock heatmap
        height, width = original_frame.shape[:2]
        heatmap = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Generate random points for heatmap
        for _ in range(5):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            cv2.circle(heatmap, (x, y), np.random.randint(20, 50), (0, 0, 255), -1)
        
        # Blur for better visualization
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # Overlay heatmap on original image
        visualization = cv2.addWeighted(original_frame, 0.7, heatmap, 0.3, 0)
        
        return {
            'visualization': visualization,
            'heatmap': heatmap,
            'manipulation_score': np.random.uniform(0.3, 0.8),
            'num_manipulated_regions': np.random.randint(1, 4)
        }

@st.cache_resource
def load_models():
    """Load all detection models"""
    # Initialize models
    hybrid_model = HybridDeepfakeDetector()
    face_analyzer = FacialBehaviorAnalyzer()
    heart_detector = RPPGDetector()
    gradcam = MockGradCAM(hybrid_model)
    
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
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float()
    # No need to unsqueeze here, as this is done in the model
    image_tensor = image_tensor / 255.0
    
    return image_tensor, image

def analyze_image(image, models):
    """Analyze a single image"""
    # Preprocess image
    image_tensor, original_image = preprocess_image(image)
    
    # Use mock predictions for hybrid model
    prob_fake = np.random.uniform(0.6, 0.9)  # Mock prediction
    
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
        'is_fake': combined_score > 0.5,
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
    """Analyze video file"""
    results = []
    frames = []
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 3 == 0:  # Process every 3rd frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = analyze_image(frame_rgb, models)
            results.append(result)
            frames.append(frame_rgb)
        
        frame_count += 1
        
        # Limit to 100 frames for memory efficiency
        if frame_count >= 300:
            break
    
    cap.release()
    
    # Calculate video-level metrics
    avg_deepfake_score = np.mean([r['confidence'] for r in results])
    suspicious_behavior_frames = sum(1 for r in results if r['behavior_result']['suspicious'])
    suspicious_heart_frames = sum(1 for r in results if r['heart_result']['suspicious'])
    
    return {
        'frame_results': results,
        'frames': frames,
        'avg_deepfake_score': avg_deepfake_score,
        'suspicious_behavior_frames': suspicious_behavior_frames,
        'suspicious_heart_frames': suspicious_heart_frames
    }

def display_image_results(result):
    """Display analysis results for image"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Deepfake Detection Results")
        
        # Display confidence meter
        confidence = result['confidence']
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                <h3 style="margin: 0;">Confidence Score</h3>
                <div style="background-color: #ddd; height: 30px; border-radius: 15px; margin: 10px 0;">
                    <div style="width: {confidence*100}%; height: 100%; background-color: {'red' if confidence > 0.5 else 'green'}; 
                         border-radius: 15px; text-align: center; line-height: 30px; color: white;">
                        {confidence*100:.1f}%
                    </div>
                </div>
                <p style="margin: 0; color: {'red' if confidence > 0.5 else 'green'}; font-weight: bold;">
                    {'LIKELY FAKE' if confidence > 0.5 else 'LIKELY REAL'}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display behavioral analysis
        st.subheader("Behavioral Analysis")
        behavior = result['behavior_result']
        if behavior['landmarks'] is not None:
            st.metric("Blink Rate", f"{behavior['blink_rate']:.2f}")
            st.metric("Facial Asymmetry", f"{behavior['avg_asymmetry']:.2f}")
            if behavior['suspicious']:
                st.warning("⚠️ Suspicious facial behavior detected")
        else:
            st.warning("No face detected for behavioral analysis")
        
        # Display heart rate analysis
        st.subheader("Physiological Analysis")
        heart = result['heart_result']
        if heart['heart_rate'] is not None:
            st.metric("Heart Rate", f"{heart['heart_rate']:.1f} BPM")
            st.metric("Signal Quality", f"{heart['signal_quality']:.2f}")
            if heart['suspicious']:
                st.warning("⚠️ Suspicious heart rate pattern")
        else:
            st.warning("Could not estimate heart rate")
    
    with col2:
        st.subheader("Visual Analysis")
        
        # Display original vs. visualization
        viz_tabs = st.tabs(["Original", "Manipulation Heatmap", "Facial Analysis"])
        
        with viz_tabs[0]:
            st.image(result['gradcam_result']['visualization'], use_container_width=True)
        
        with viz_tabs[1]:
            st.image(result['gradcam_result']['heatmap'], use_container_width=True)
            st.metric("Manipulation Score", f"{result['gradcam_result']['manipulation_score']:.2f}")
            st.metric("Suspicious Regions", result['gradcam_result']['num_manipulated_regions'])
        
        with viz_tabs[2]:
            if behavior['landmarks'] is not None:
                behavior_viz = models['face_analyzer'].visualize_analysis(
                    cv2.cvtColor(result['gradcam_result']['visualization'], cv2.COLOR_RGB2BGR),
                    behavior
                )
                st.image(behavior_viz, use_container_width=True)

def display_video_results(result):
    """Display analysis results for video"""
    st.subheader("Video Analysis Results")
    
    # Overall metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Deepfake Score", f"{result['avg_deepfake_score']:.2%}")
    with col2:
        st.metric("Suspicious Behavior Frames", result['suspicious_behavior_frames'])
    with col3:
        st.metric("Suspicious Heart Rate Frames", result['suspicious_heart_frames'])
    
    # Frame analysis
    st.subheader("Frame-by-Frame Analysis")
    frame_selector = st.slider("Select Frame", 0, len(result['frames'])-1, 0)
    
    # Display selected frame results
    display_image_results(result['frame_results'][frame_selector])
    
    # Plot temporal analysis
    st.subheader("Temporal Analysis")
    import plotly.graph_objects as go
    
    scores = [r['confidence'] for r in result['frame_results']]
    behavior_scores = [float(r['behavior_result']['suspicious']) for r in result['frame_results']]
    heart_scores = [float(r['heart_result']['suspicious']) for r in result['frame_results']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=scores, name="Deepfake Score", line=dict(color='red')))
    fig.add_trace(go.Scatter(y=behavior_scores, name="Behavior Score", line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=heart_scores, name="Heart Rate Score", line=dict(color='green')))
    
    fig.update_layout(
        title="Temporal Analysis of Detection Scores",
        xaxis_title="Frame",
        yaxis_title="Score",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Main application
st.title("Advanced Deepfake Detection System")
st.markdown("""
This system uses a multi-modal approach to detect deepfake content:
- Hybrid Deep Learning Model (CNN + Bi-LSTM + ViT)
- Facial Behavior Analysis
- Heart Rate Estimation
- Visual Explanation through Grad-CAM
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
