import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataloader import create_dataloaders, create_single_frame_dataloaders
from models.cnn_lstm import get_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing processed data')
    parser.add_argument('--csv_file', type=str, default='dataset.csv', help='CSV file with dataset information')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save model and results')
    parser.add_argument('--model_type', type=str, default='cnn_lstm', choices=['cnn_lstm', 'cnn'], help='Model type')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', help='CNN backbone')
    parser.add_argument('--sequence_length', type=int, default=20, help='Sequence length for LSTM')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in tqdm(dataloader, desc='Training', leave=False):
        inputs = inputs.to(device)
        labels = labels.float().to(device).view(-1, 1)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)  # Apply sigmoid for binary classification
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        
        # Convert outputs to binary predictions
        preds = (outputs > 0.5).float().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation', leave=False):
            inputs = inputs.to(device)
            labels = labels.float().to(device).view(-1, 1)
            
            # Forward pass
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid for binary classification
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Convert outputs to binary predictions
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1

def plot_metrics(train_metrics, val_metrics, metric_name, output_dir):
    """Plot training and validation metrics."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{metric_name.lower()}.png'))
    plt.close()

def plot_confusion_matrix(model, dataloader, device, output_dir):
    """Plot confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.numpy()
            
            # Forward pass
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid for binary classification
            preds = (outputs > 0.5).float().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def save_results(train_metrics, val_metrics, output_dir):
    """Save training and validation metrics to CSV."""
    results = {
        'epoch': list(range(1, len(train_metrics['loss']) + 1)),
        'train_loss': train_metrics['loss'],
        'train_acc': train_metrics['acc'],
        'train_precision': train_metrics['precision'],
        'train_recall': train_metrics['recall'],
        'train_f1': train_metrics['f1'],
        'val_loss': val_metrics['loss'],
        'val_acc': val_metrics['acc'],
        'val_precision': val_metrics['precision'],
        'val_recall': val_metrics['recall'],
        'val_f1': val_metrics['f1']
    }
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'training_results.csv'), index=False)

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataloaders
    print("Creating dataloaders...")
    if args.model_type == 'cnn_lstm':
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=args.data_dir,
            csv_file=args.csv_file,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_workers=args.num_workers
        )
    else:
        train_loader, val_loader, test_loader = create_single_frame_dataloaders(
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
        pretrained=True,
        dropout=args.dropout
    )
    model = model.to(args.device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Initialize metrics
    train_metrics = {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []}
    val_metrics = {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []}
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_epoch(
            model, val_loader, criterion, args.device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save metrics
        train_metrics['loss'].append(train_loss)
        train_metrics['acc'].append(train_acc)
        train_metrics['precision'].append(train_precision)
        train_metrics['recall'].append(train_recall)
        train_metrics['f1'].append(train_f1)
        
        val_metrics['loss'].append(val_loss)
        val_metrics['acc'].append(val_acc)
        val_metrics['precision'].append(val_precision)
        val_metrics['recall'].append(val_recall)
        val_metrics['f1'].append(val_f1)
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1
            }, os.path.join(args.output_dir, 'best_model.pth'))
            
            print("Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_loss, test_acc, test_precision, test_recall, test_f1 = validate_epoch(
        model, test_loader, criterion, args.device
    )
    
    print("Test Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    
    # Save test results
    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Loss: {test_loss:.4f}\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"Precision: {test_precision:.4f}\n")
        f.write(f"Recall: {test_recall:.4f}\n")
        f.write(f"F1 Score: {test_f1:.4f}\n")
    
    # Plot metrics
    plot_metrics(train_metrics['loss'], val_metrics['loss'], 'Loss', args.output_dir)
    plot_metrics(train_metrics['acc'], val_metrics['acc'], 'Accuracy', args.output_dir)
    plot_metrics(train_metrics['f1'], val_metrics['f1'], 'F1 Score', args.output_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(model, test_loader, args.device, args.output_dir)
    
    # Save all results
    save_results(train_metrics, val_metrics, args.output_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 