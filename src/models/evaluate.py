import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataloader import create_dataloaders, create_single_frame_dataloaders
from models.cnn_lstm import get_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing test data')
    parser.add_argument('--csv_file', type=str, default='test.csv', help='CSV file with test data information')
    parser.add_argument('--output_dir', type=str, default='evaluation', help='Directory to save evaluation results')
    parser.add_argument('--model_type', type=str, default='cnn_lstm', choices=['cnn_lstm', 'cnn'], help='Model type')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', help='CNN backbone')
    parser.add_argument('--sequence_length', type=int, default=20, help='Sequence length for LSTM')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    return parser.parse_args()

def evaluate_model(model, dataloader, device):
    """Evaluate model on the given dataloader."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.numpy()
            
            # Forward pass
            outputs = model(inputs)
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }

def plot_confusion_matrix(labels, predictions, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(labels, probabilities, output_dir):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def analyze_errors(dataloader, model, device, output_dir, data_dir, top_k=10):
    """Analyze prediction errors."""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Analyzing errors'):
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.numpy()
            
            # Forward pass
            outputs = model(inputs)
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
            
            # Find errors
            for i in range(batch_size):
                if preds[i] != labels[i]:
                    errors.append({
                        'true_label': labels[i],
                        'predicted': preds[i],
                        'confidence': probs[i][0],
                        'sample_idx': len(errors)
                    })
    
    # Convert to DataFrame
    error_df = pd.DataFrame(errors)
    
    # Save error analysis
    error_df.to_csv(os.path.join(output_dir, 'error_analysis.csv'), index=False)
    
    # Plot error distribution
    if not error_df.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(error_df, x='confidence', hue='true_label', bins=20, kde=True)
        plt.title('Distribution of Confidence Scores for Errors')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'error_confidence_distribution.png'))
        plt.close()
    
    return error_df

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataloader
    print("Creating dataloader...")
    if args.model_type == 'cnn_lstm':
        _, _, test_loader = create_dataloaders(
            data_dir=args.data_dir,
            csv_file=args.csv_file,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_workers=args.num_workers
        )
    else:
        _, _, test_loader = create_single_frame_dataloaders(
            data_dir=args.data_dir,
            csv_file=args.csv_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    
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
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, args.device)
    
    # Print metrics
    print("Evaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n")
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(results['labels'], results['predictions'], args.output_dir)
    
    # Plot ROC curve
    print("Plotting ROC curve...")
    plot_roc_curve(results['labels'], results['probabilities'], args.output_dir)
    
    # Analyze errors
    print("Analyzing errors...")
    error_df = analyze_errors(test_loader, model, args.device, args.output_dir, args.data_dir)
    
    print(f"Evaluation completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 