import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection."""
    
    def __init__(self, data_dir, csv_file, transform=None, sequence_length=20, is_train=True):
        """
        Args:
            data_dir (str): Directory with all the images.
            csv_file (str): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            sequence_length (int): Number of frames to use for sequence-based models.
            is_train (bool): Whether this is training set or not.
        """
        self.data_dir = data_dir
        self.df = pd.read_csv(os.path.join(data_dir, csv_file))
        self.transform = transform
        self.sequence_length = sequence_length
        self.is_train = is_train
        
        # Convert labels to numeric
        self.label_map = {'real': 0, 'fake': 1}
        self.df['numeric_label'] = self.df['label'].map(self.label_map)
        
        # Group by video to handle sequences
        if 'video_id' in self.df.columns:
            self.video_groups = self.df.groupby('video_id')
        else:
            # If no video_id, create artificial groups based on filename patterns
            self.df['video_id'] = self.df['path'].apply(lambda x: os.path.basename(os.path.dirname(x)))
            self.video_groups = self.df.groupby('video_id')
    
    def __len__(self):
        if self.is_train:
            # For training, each video can start a sequence
            return len(self.df)
        else:
            # For validation/testing, use one sequence per video
            return len(self.video_groups)
    
    def __getitem__(self, idx):
        if self.is_train:
            # Get a random starting point within a video
            row = self.df.iloc[idx]
            video_id = row['video_id']
            video_frames = self.video_groups.get_group(video_id)
            
            # If not enough frames in video, repeat the last frame
            if len(video_frames) < self.sequence_length:
                frames_to_use = video_frames.sample(self.sequence_length, replace=True)
            else:
                # Try to get a contiguous sequence if possible
                start_idx = np.random.randint(0, max(1, len(video_frames) - self.sequence_length + 1))
                frames_to_use = video_frames.iloc[start_idx:start_idx + self.sequence_length]
                
                # If still not enough frames, sample with replacement
                if len(frames_to_use) < self.sequence_length:
                    frames_to_use = video_frames.sample(self.sequence_length, replace=True)
        else:
            # For validation/testing, use the first sequence from each video
            video_id = list(self.video_groups.groups.keys())[idx]
            video_frames = self.video_groups.get_group(video_id)
            
            # If not enough frames, repeat the last frame
            if len(video_frames) < self.sequence_length:
                frames_to_use = video_frames.sample(self.sequence_length, replace=True)
            else:
                frames_to_use = video_frames.iloc[:self.sequence_length]
        
        # Load images and create sequence
        sequence = []
        for _, frame_row in frames_to_use.iterrows():
            img_path = os.path.join(self.data_dir, frame_row['path'])
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            sequence.append(image)
        
        # Stack sequence
        if isinstance(sequence[0], torch.Tensor):
            sequence = torch.stack(sequence)
        else:
            sequence = np.stack(sequence)
        
        # Get label (all frames in a video have the same label)
        label = frames_to_use['numeric_label'].iloc[0]
        
        return sequence, label

class SingleFrameDataset(Dataset):
    """Dataset for frame-by-frame deepfake detection."""
    
    def __init__(self, data_dir, csv_file, transform=None):
        """
        Args:
            data_dir (str): Directory with all the images.
            csv_file (str): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.df = pd.read_csv(os.path.join(data_dir, csv_file))
        self.transform = transform
        
        # Convert labels to numeric
        self.label_map = {'real': 0, 'fake': 1}
        self.df['numeric_label'] = self.df['label'].map(self.label_map)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['path'])
        
        image = Image.open(img_path).convert('RGB')
        label = row['numeric_label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(input_size=224):
    """Get data transformations for training and validation."""
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_dataloaders(data_dir, csv_file, batch_size=16, sequence_length=20, input_size=224, num_workers=4):
    """Create train, validation, and test dataloaders."""
    # Read the dataset
    df = pd.read_csv(os.path.join(data_dir, csv_file))
    
    # Split into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    # Save splits to CSV
    train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    
    # Get transforms
    train_transform, val_transform = get_transforms(input_size)
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        data_dir=data_dir,
        csv_file='train.csv',
        transform=train_transform,
        sequence_length=sequence_length,
        is_train=True
    )
    
    val_dataset = DeepfakeDataset(
        data_dir=data_dir,
        csv_file='val.csv',
        transform=val_transform,
        sequence_length=sequence_length,
        is_train=False
    )
    
    test_dataset = DeepfakeDataset(
        data_dir=data_dir,
        csv_file='test.csv',
        transform=val_transform,
        sequence_length=sequence_length,
        is_train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def create_single_frame_dataloaders(data_dir, csv_file, batch_size=32, input_size=224, num_workers=4):
    """Create train, validation, and test dataloaders for single-frame models."""
    # Read the dataset
    df = pd.read_csv(os.path.join(data_dir, csv_file))
    
    # Split into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    # Save splits to CSV
    train_df.to_csv(os.path.join(data_dir, 'train_frames.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'val_frames.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test_frames.csv'), index=False)
    
    # Get transforms
    train_transform, val_transform = get_transforms(input_size)
    
    # Create datasets
    train_dataset = SingleFrameDataset(
        data_dir=data_dir,
        csv_file='train_frames.csv',
        transform=train_transform
    )
    
    val_dataset = SingleFrameDataset(
        data_dir=data_dir,
        csv_file='val_frames.csv',
        transform=val_transform
    )
    
    test_dataset = SingleFrameDataset(
        data_dir=data_dir,
        csv_file='test_frames.csv',
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 