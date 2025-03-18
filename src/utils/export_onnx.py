import os
import argparse
import torch
import numpy as np

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_lstm import get_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--output_path', type=str, default='model.onnx', help='Path to save the ONNX model')
    parser.add_argument('--model_type', type=str, default='cnn_lstm', choices=['cnn_lstm', 'cnn'], help='Model type')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', help='CNN backbone')
    parser.add_argument('--sequence_length', type=int, default=20, help='Sequence length for LSTM')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    return parser.parse_args()

# Create a wrapper class that applies sigmoid to the output
class ModelWithSigmoid(torch.nn.Module):
    def __init__(self, model):
        super(ModelWithSigmoid, self).__init__()
        self.model = model
    
    def forward(self, x):
        return torch.sigmoid(self.model(x))

def export_to_onnx(model, output_path, model_type, sequence_length, input_size, device):
    """Export PyTorch model to ONNX format."""
    model.eval()
    
    # Create dummy input
    if model_type == 'cnn_lstm':
        # For CNN-LSTM, input shape is (batch_size, sequence_length, channels, height, width)
        dummy_input = torch.randn(1, sequence_length, 3, input_size, input_size, device=device)
    else:
        # For CNN-only, input shape is (batch_size, channels, height, width)
        dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    
    # Wrap the model to include sigmoid activation
    wrapped_model = ModelWithSigmoid(model)
    
    # Export to ONNX
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")
    
    # Verify the exported model
    try:
        import onnx
        import onnxruntime
        
        # Load the ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Create an ONNX Runtime session
        ort_session = onnxruntime.InferenceSession(output_path)
        
        # Run the model
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Compare PyTorch and ONNX Runtime outputs
        pytorch_output = torch.sigmoid(model(dummy_input)).detach().cpu().numpy()
        np.testing.assert_allclose(pytorch_output, ort_outputs[0], rtol=1e-03, atol=1e-05)
        
        print("Exported model has been verified!")
        
        # Print model info
        print("\nModel Input:")
        print(f"  Name: {ort_session.get_inputs()[0].name}")
        print(f"  Shape: {ort_session.get_inputs()[0].shape}")
        print(f"  Type: {ort_session.get_inputs()[0].type}")
        
        print("\nModel Output:")
        print(f"  Name: {ort_session.get_outputs()[0].name}")
        print(f"  Shape: {ort_session.get_outputs()[0].shape}")
        print(f"  Type: {ort_session.get_outputs()[0].type}")
        
    except ImportError:
        print("ONNX or ONNX Runtime not installed. Skipping verification.")
    except Exception as e:
        print(f"Error verifying the exported model: {e}")

def main():
    args = parse_args()
    
    # Create model
    print(f"Creating {args.model_type} model with {args.backbone} backbone...")
    model = get_model(
        model_type=args.model_type,
        backbone=args.backbone,
        pretrained=False  # We'll load weights from checkpoint
    )
    
    # Load model weights
    print(f"Loading model weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    
    # Export to ONNX
    print("Exporting model to ONNX...")
    export_to_onnx(
        model,
        args.output_path,
        args.model_type,
        args.sequence_length,
        args.input_size,
        args.device
    )

if __name__ == "__main__":
    main() 