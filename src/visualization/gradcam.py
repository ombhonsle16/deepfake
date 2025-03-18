import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import argparse
from tqdm import tqdm

class GradCAM:
    def __init__(self, model, target_layer_name, device):
        self.model = model
        self.target_layer_name = target_layer_name
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'backbone'):
            if 'efficientnet' in self.target_layer_name:
                self.target_layer = model.encoder.backbone.features[-1]
            elif 'resnet' in self.target_layer_name:
                self.target_layer = model.encoder.backbone.layer4[-1]
            else:
                raise ValueError(f"Unsupported backbone for target layer: {self.target_layer_name}")
        else:
            raise ValueError("Model structure not supported for Grad-CAM")
        
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def __call__(self, input_tensor, target_class=None):
        if len(input_tensor.shape) == 5:
            batch_size, seq_len, c, h, w = input_tensor.shape
            input_reshaped = input_tensor.view(batch_size * seq_len, c, h, w)
            
            with torch.no_grad():
                cnn_features = self.model.encoder(input_reshaped)
                cnn_features = cnn_features.view(batch_size, seq_len, -1)
            
            output = self.model(input_tensor)
            input_tensor = input_tensor[:, 0]
        else:
            output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.zeros(output.size(0), dtype=torch.long, device=self.device)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output, device=self.device)
        one_hot[:, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().detach(), output.cpu().detach()

def preprocess_image(image_path, input_size=224):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_size, input_size))
    
    image_np = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np - mean) / std
    
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0)
    return tensor

def generate_gradcam_visualization(image_path, model, target_layer_name, device, output_path=None, alpha=0.5):
    input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    grad_cam = GradCAM(model, target_layer_name, device)
    cam, output = grad_cam(input_tensor)
    cam = cam.squeeze().numpy()
    
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    visualization = heatmap * alpha + image * (1 - alpha)
    visualization = np.uint8(visualization)
    
    prob = torch.sigmoid(output).item()
    pred_class = "Fake" if prob > 0.5 else "Real"
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    ax2.imshow(heatmap)
    ax2.set_title("Grad-CAM Heatmap")
    ax2.axis('off')
    
    ax3.imshow(visualization)
    ax3.set_title(f"Overlay (Prediction: {pred_class}, Confidence: {prob:.2f})")
    ax3.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
    
    return visualization, prob

def batch_process_images(image_dir, model, target_layer_name, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results = []
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        output_path = os.path.join(output_dir, f"gradcam_{image_file}")
        
        try:
            _, prob = generate_gradcam_visualization(
                image_path, model, target_layer_name, device, output_path
            )
            
            results.append({
                'image': image_file,
                'prediction': "Fake" if prob > 0.5 else "Real",
                'confidence': prob
            })
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    return results 