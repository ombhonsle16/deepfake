import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel

class HybridDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(HybridDeepfakeDetector, self).__init__()
        
        # CNN Feature Extractor (EfficientNet)
        self.cnn = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        cnn_features = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Identity()
        
        # Vision Transformer
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        vit_features = self.vit.config.hidden_size
        
        # Bi-LSTM for temporal features
        self.lstm_hidden_size = 256
        self.bilstm = nn.LSTM(
            input_size=cnn_features + vit_features,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_hidden_size * 2,
            num_heads=8
        )
        
    def forward(self, x, sequence_length=16):
        batch_size = x.size(0) // sequence_length
        
        # Reshape input for sequence processing
        x = x.view(batch_size, sequence_length, 3, 224, 224)
        
        # Process each frame in the sequence
        cnn_features_seq = []
        vit_features_seq = []
        
        for t in range(sequence_length):
            # Extract CNN features
            frame = x[:, t, :, :, :]
            cnn_feat = self.cnn(frame)
            cnn_features_seq.append(cnn_feat)
            
            # Extract ViT features
            vit_output = self.vit(frame)
            vit_feat = vit_output.last_hidden_state[:, 0, :]  # Use [CLS] token
            vit_features_seq.append(vit_feat)
        
        # Combine CNN and ViT features
        cnn_features = torch.stack(cnn_features_seq, dim=1)
        vit_features = torch.stack(vit_features_seq, dim=1)
        combined_features = torch.cat([cnn_features, vit_features], dim=-1)
        
        # Process with Bi-LSTM
        lstm_out, _ = self.bilstm(combined_features)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(
            lstm_out.permute(1, 0, 2),
            lstm_out.permute(1, 0, 2),
            lstm_out.permute(1, 0, 2)
        )
        attn_output = attn_output.permute(1, 0, 2)
        
        # Get sequence representation
        sequence_repr = attn_output.mean(dim=1)
        
        # Final classification
        output = self.classifier(sequence_repr)
        
        return output
    
    def predict_frame(self, frame):
        """Predict single frame for real-time detection"""
        with torch.no_grad():
            # Extract CNN features
            cnn_feat = self.cnn(frame.unsqueeze(0))
            
            # Extract ViT features
            vit_output = self.vit(frame.unsqueeze(0))
            vit_feat = vit_output.last_hidden_state[:, 0, :]
            
            # Combine features
            combined_feat = torch.cat([cnn_feat, vit_feat], dim=-1)
            
            # Process with a simplified version of the temporal model
            lstm_out, _ = self.bilstm(combined_feat.unsqueeze(1))
            
            # Final classification
            output = self.classifier(lstm_out.squeeze(1))
            
            return torch.softmax(output, dim=1) 