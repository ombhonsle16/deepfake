import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CNNEncoder(nn.Module):
    def __init__(self, backbone='efficientnet_b0', pretrained=True, freeze_backbone=False):
        super(CNNEncoder, self).__init__()
        
        if backbone == 'efficientnet_b0':
            try:
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = models.efficientnet_b0(weights=weights)
                self.feature_dim = self.backbone.classifier[1].in_features
            except (AttributeError, TypeError):
                self.backbone = models.efficientnet_b0(pretrained=pretrained)
                self.feature_dim = self.backbone.classifier[1].in_features
            
            self.backbone.classifier = nn.Identity()
        elif backbone == 'xception':
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = models.resnet50(weights=weights)
            except (AttributeError, TypeError):
                self.backbone = models.resnet50(pretrained=pretrained)
                
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet50':
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = models.resnet50(weights=weights)
            except (AttributeError, TypeError):
                self.backbone = models.resnet50(pretrained=pretrained)
                
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)

class DeepfakeCNNLSTM(nn.Module):
    def __init__(self, backbone='efficientnet_b0', pretrained=True, freeze_backbone=False,
                 lstm_hidden_size=512, lstm_num_layers=2, dropout=0.5):
        super(DeepfakeCNNLSTM, self).__init__()
        
        self.encoder = CNNEncoder(backbone, pretrained, freeze_backbone)
        
        self.lstm = nn.LSTM(
            input_size=self.encoder.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 1),
            nn.Tanh()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, 1)
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        cnn_features = self.encoder(x_reshaped)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(cnn_features)
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.classifier(context)
        return output

class DeepfakeCNN(nn.Module):
    def __init__(self, backbone='efficientnet_b0', pretrained=True, dropout=0.5):
        super(DeepfakeCNN, self).__init__()
        
        self.encoder = CNNEncoder(backbone, pretrained, False)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        if len(x.shape) == 4:
            features = self.encoder(x)
        elif len(x.shape) == 5:
            features = self.encoder(x[:, 0])
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        return self.classifier(features)

def get_model(model_type='cnn_lstm', backbone='efficientnet_b0', pretrained=True, **kwargs):
    if model_type == 'cnn_lstm':
        return DeepfakeCNNLSTM(backbone, pretrained, **kwargs)
    elif model_type == 'cnn':
        return DeepfakeCNN(backbone, pretrained, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 