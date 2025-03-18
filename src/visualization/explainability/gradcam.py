import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Attach hooks
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate class activation map"""
        # Forward pass
        model_output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(model_output)
        
        # Clear gradients
        self.model.zero_grad()
        
        # Backward pass
        class_loss = model_output[0, target_class]
        class_loss.backward()
        
        # Get weights
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weight combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        return cam.detach().cpu().numpy()
    
    def __call__(self, input_tensor, target_class=None):
        """Generate heatmap for input tensor"""
        # Generate CAM
        cam = self.generate_cam(input_tensor, target_class)
        
        # Normalize
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

class DeepfakeGradCAM:
    def __init__(self, model):
        """Initialize Grad-CAM for deepfake detection"""
        self.model = model
        
        # Get the last convolutional layer
        if hasattr(model, 'cnn'):
            # For hybrid model, use the last conv layer of CNN
            target_layer = None
            for module in model.cnn.modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
            self.gradcam = GradCAM(model, target_layer)
        else:
            raise ValueError("Model architecture not supported for Grad-CAM")
    
    def generate_heatmap(self, image_tensor, target_size=None):
        """Generate heatmap for an image tensor"""
        # Ensure model is in eval mode
        self.model.eval()
        
        # Generate Grad-CAM
        cam = self.gradcam(image_tensor)
        
        # Resize to match input image size if specified
        if target_size is not None:
            cam = cv2.resize(cam, target_size)
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        return heatmap
    
    def overlay_heatmap(self, image, heatmap, alpha=0.5):
        """Overlay heatmap on original image"""
        # Ensure image and heatmap have same size
        if image.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert image to BGR if it's not
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        # Overlay heatmap
        output = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return output
    
    def analyze_frame(self, frame_tensor, original_frame):
        """Analyze a single frame and generate visualization"""
        # Generate heatmap
        heatmap = self.generate_heatmap(frame_tensor, target_size=(original_frame.shape[1], original_frame.shape[0]))
        
        # Overlay on original frame
        visualization = self.overlay_heatmap(original_frame, heatmap)
        
        # Calculate manipulation score based on heatmap intensity
        manipulation_score = np.mean(heatmap) / 255.0
        
        # Identify regions of high manipulation
        threshold = 0.7
        high_manip_mask = heatmap[:, :, 0] > (threshold * 255)
        regions = cv2.connectedComponents(high_manip_mask.astype(np.uint8))[1]
        num_regions = len(np.unique(regions)) - 1  # Subtract background
        
        return {
            'visualization': visualization,
            'heatmap': heatmap,
            'manipulation_score': manipulation_score,
            'num_manipulated_regions': num_regions
        }
    
    def analyze_video(self, frame_tensors, original_frames):
        """Analyze a sequence of frames"""
        results = []
        for frame_tensor, original_frame in zip(frame_tensors, original_frames):
            result = self.analyze_frame(frame_tensor, original_frame)
            results.append(result)
        
        # Calculate temporal consistency
        scores = [r['manipulation_score'] for r in results]
        temporal_consistency = np.std(scores)  # Lower std means more consistent
        
        # Identify suspicious frames (high manipulation score)
        suspicious_frames = [i for i, r in enumerate(results) if r['manipulation_score'] > 0.7]
        
        return {
            'frame_results': results,
            'temporal_consistency': temporal_consistency,
            'suspicious_frames': suspicious_frames,
            'average_manipulation': np.mean(scores)
        }
    
    def visualize_results(self, analysis_result, frame_idx=None):
        """Create visualization with annotations"""
        if frame_idx is not None:
            # Single frame visualization
            result = analysis_result['frame_results'][frame_idx]
            output = result['visualization'].copy()
            
            # Add annotations
            cv2.putText(output, f"Manipulation Score: {result['manipulation_score']:.2f}",
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(output, f"Regions: {result['num_manipulated_regions']}",
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Summary visualization
            output = analysis_result['frame_results'][0]['visualization'].copy()
            
            cv2.putText(output, f"Avg Manipulation: {analysis_result['average_manipulation']:.2f}",
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(output, f"Temporal Consistency: {analysis_result['temporal_consistency']:.2f}",
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(output, f"Suspicious Frames: {len(analysis_result['suspicious_frames'])}",
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output 