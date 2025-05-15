"""
Fine-tuning script for GraphCNN model with k-fold cross-validation.

This script allows fine-tuning a pre-trained GraphCNN model on a new dataset
using k-fold cross-validation to ensure robust performance evaluation.
"""

import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

from graph_cnn import SimplifiedGraphCNN, GraphDataset, Config


class FinetuneConfig:
    """Configuration parameters for model fine-tuning."""
    # Default values that can be overridden by command line arguments
    BATCH_SIZE = 8          # Smaller batch size for fine-tuning
    EPOCHS = 30             # Fewer epochs for fine-tuning
    LEARNING_RATE = 1.2 e-4  # Lower learning rate for fine-tuning
    K_FOLDS = 7             # Number of folds for cross-validation
    MODEL_DIR = "models"    # Directory to save fine-tuned models
    FREEZE_BACKBONE = False # Whether to freeze the backbone network


class TransformDataset(Dataset):
    """Dataset wrapper that applies transforms to a subset of another dataset.
    
    Args:
        dataset (Dataset): Base dataset.
        indices (list): Indices to use from the base dataset.
        transform (callable): Transform to apply to the data.
    """
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
            
    def __getitem__(self, idx):
        image, adjacency_matrix = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, adjacency_matrix


def get_transforms(image_size, augment=False):
    """Get image transformations for training and validation.
    
    Args:
        image_size (int): Size to resize images to.
        augment (bool): Whether to apply data augmentation.
        
    Returns:
        transforms.Compose: Composed transformations.
    """
    if augment:
        # Transforms with augmentation for training (avoiding rotation)
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            transforms.RandomGrayscale(p=0.3),  # Occasional grayscale for robustness
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # Simulate blurry images
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        # For validation, we don't use augmentation
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])


def load_model_for_finetuning(pretrained_path, num_nodes, device, freeze_backbone=False):
    """Load a pretrained model and prepare it for fine-tuning.
    
    Args:
        pretrained_path (str): Path to the pretrained model weights.
        num_nodes (int): Number of nodes in the graph.
        device: Device to load the model on.
        freeze_backbone (bool): Whether to freeze the backbone network.
        
    Returns:
        nn.Module: Model prepared for fine-tuning.
    """
    model = SimplifiedGraphCNN(num_nodes=num_nodes).to(device)
    
    # Load the pretrained weights
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    print(f"Loaded pretrained model from {pretrained_path}")
    
    # Optionally freeze the backbone
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        
        # Count trainable vs frozen parameters
        trainable_params = 0
        frozen_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"Trainable: {name}")
            else:
                frozen_params += param.numel()
        
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
    
    return model


def finetune_fold(fold, k_folds, train_indices, val_indices, full_dataset, 
                  pretrained_model_path, config, device):
    """Fine-tune model using a single fold of the data.
    
    Args:
        fold (int): Current fold number.
        k_folds (int): Total number of folds.
        train_indices (list): Indices for training data.
        val_indices (list): Indices for validation data.
        full_dataset (Dataset): Complete dataset.
        pretrained_model_path (str): Path to pretrained model.
        config: Configuration parameters.
        device: Device to train on.
        
    Returns:
        dict: Results of the fold, including best validation loss and model path.
    """
    print(f"\n{'='*50}")
    print(f"FOLD {fold+1}/{k_folds}")
    print(f"{'='*50}")
    
    # Get transforms
    train_transform = get_transforms(config.IMAGE_SIZE, augment=True)
    val_transform = get_transforms(config.IMAGE_SIZE, augment=False)
    
    # Create train and validation datasets with appropriate transforms
    train_dataset = TransformDataset(full_dataset, train_indices, train_transform)
    val_dataset = TransformDataset(full_dataset, val_indices, val_transform)
    
    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    
    # Load pretrained model for fine-tuning
    model = load_model_for_finetuning(
        pretrained_model_path, 
        config.NUM_NODES, 
        device, 
        freeze_backbone=config.FREEZE_BACKBONE
    )
    
    # Setup loss and optimizer with a lower learning rate
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.LEARNING_RATE, 
        weight_decay=1e-4
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Learning rate scheduler for better fine-tuning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training Loop
    best_val_loss = float('inf')
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    best_fold_model_path = os.path.join(config.MODEL_DIR, f"best_finetuned_fold_{fold+1}.pth")
    
    for epoch in range(config.EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        for images, adjacencies in tqdm(train_loader, desc=f"Fold {fold+1}, Epoch {epoch+1}/{config.EPOCHS}"):
            images, adjacencies = images.to(device), adjacencies.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, adjacencies)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct_edges = 0
        total_edges = 0
        
        with torch.no_grad():
            for images, adjacencies in val_loader:
                images, adjacencies = images.to(device), adjacencies.to(device)
                outputs = model(images)
                loss = criterion(outputs, adjacencies)
                val_loss += loss.item()
                
                # Calculate edge prediction accuracy
                predictions = (outputs > 0.5).float()
                correct_edges += (predictions == adjacencies).sum().item()
                total_edges += adjacencies.numel()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct_edges / total_edges
        
        print(f"Fold [{fold+1}/{k_folds}], Epoch [{epoch+1}/{config.EPOCHS}], "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Accuracy: {accuracy:.4f}")
        
        # Update learning rate based on validation performance
        scheduler.step(avg_val_loss)
        
        # Save the best model for this fold
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_fold_model_path)
            print(f"Saved best model for fold {fold+1} with validation loss: {avg_val_loss:.4f}")
    
    return {
        'fold': fold + 1,
        'best_val_loss': best_val_loss,
        'model_path': best_fold_model_path
    }


def finetune_with_kfold(args, config):
    """Fine-tune model with k-fold cross-validation.
    
    Args:
        args: Command line arguments.
        config: Configuration parameters.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset without transforms initially
    full_dataset = GraphDataset(
        json_path=args.json_path,
        image_dir=args.image_dir,
        transform=None,
        num_nodes=config.NUM_NODES
    )
    
    dataset_size = len(full_dataset)
    print(f"Dataset loaded with {dataset_size} samples")
    
    # Setup k-fold cross validation
    k_fold = KFold(n_splits=config.K_FOLDS, shuffle=True, random_state=42)
    
    # Store fold results
    fold_results = []
    
    # K-Fold Cross Validation
    for fold, (train_indices, val_indices) in enumerate(k_fold.split(range(dataset_size))):
        fold_result = finetune_fold(
            fold=fold,
            k_folds=config.K_FOLDS,
            train_indices=train_indices,
            val_indices=val_indices,
            full_dataset=full_dataset,
            pretrained_model_path=args.pretrained_model,
            config=config,
            device=device
        )
        fold_results.append(fold_result)
    
    # Analyze and report cross-validation results
    print("\n" + "="*50)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("="*50)
    
    avg_val_loss = 0
    best_fold = None
    best_loss = float('inf')
    
    for result in fold_results:
        print(f"Fold {result['fold']}: Best validation loss = {result['best_val_loss']:.4f}")
        avg_val_loss += result['best_val_loss']
        
        if result['best_val_loss'] < best_loss:
            best_loss = result['best_val_loss']
            best_fold = result['fold']
    
    avg_val_loss /= config.K_FOLDS
    print(f"\nAverage validation loss across all folds: {avg_val_loss:.4f}")
    print(f"Best performing fold: {best_fold} with loss {best_loss:.4f}")
    
    # Create final model using the best fold's weights
    best_model_path = os.path.join(config.MODEL_DIR, f"best_finetuned_fold_{best_fold}.pth")
    final_model_path = os.path.join(config.MODEL_DIR, "final_finetuned_model.pth")
    
    # Copy the best fold's model to be the final model
    model = SimplifiedGraphCNN(config.NUM_NODES).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    torch.save(model.state_dict(), final_model_path)
    
    print(f"\nSaved final fine-tuned model (from best fold {best_fold}) to {final_model_path}")
    print("Fine-tuning with k-fold cross-validation completed!")


def evaluate_model(args, config):
    """Evaluate model on a test dataset.
    
    Args:
        args: Command line arguments.
        config: Configuration parameters.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the fine-tuned model
    model = SimplifiedGraphCNN(config.NUM_NODES).to(device)
    model.load_state_dict(torch.load(args.eval_model, map_location=device))
    model.eval()
    
    # Setup transforms for evaluation
    eval_transform = get_transforms(config.IMAGE_SIZE, augment=False)
    
    # Load test dataset
    test_dataset = GraphDataset(
        json_path=args.eval_json,
        image_dir=args.eval_dir,
        transform=eval_transform,
        num_nodes=config.NUM_NODES
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    # Evaluation metrics
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    correct_edges = 0
    total_edges = 0
    
    with torch.no_grad():
        for images, adjacencies in tqdm(test_loader, desc="Evaluating"):
            images, adjacencies = images.to(device), adjacencies.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, adjacencies)
            total_loss += loss.item()
            
            # Calculate edge prediction accuracy (threshold at 0.5)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_edges += (predictions == adjacencies).sum().item()
            total_edges += adjacencies.numel()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_edges / total_edges
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Edge Prediction Accuracy: {accuracy:.4f} ({correct_edges}/{total_edges})")
    
    return avg_loss, accuracy


def main(args):
    """Main function for fine-tuning and evaluation."""
    # Create config object that combines defaults with command line parameters
    class CombinedConfig:
        pass
    
    config = CombinedConfig()
    
    # Copy attributes from Config
    for attr in dir(Config):
        if not attr.startswith('__'):
            setattr(config, attr, getattr(Config, attr))
    
    # Override with FinetuneConfig values
    for attr in dir(FinetuneConfig):
        if not attr.startswith('__'):
            setattr(config, attr, getattr(FinetuneConfig, attr))
    
    # Override with command line arguments if provided
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if hasattr(args, 'epochs') and args.epochs is not None:
        config.EPOCHS = args.epochs
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
    if hasattr(args, 'k_folds') and args.k_folds is not None:
        config.K_FOLDS = args.k_folds
    if hasattr(args, 'model_dir') and args.model_dir is not None:
        config.MODEL_DIR = args.model_dir
    if hasattr(args, 'freeze_backbone'):
        config.FREEZE_BACKBONE = args.freeze_backbone
    
    # Fine-tune or evaluate based on command
    if args.command == 'finetune':
        finetune_with_kfold(args, config)
    elif args.command == 'evaluate':
        evaluate_model(args, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune or evaluate a GraphCNN model")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Fine-tuning command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune the model")
    finetune_parser.add_argument("--pretrained_model", type=str, required=True,
                                help="Path to the pretrained model")
    finetune_parser.add_argument("--json_path", type=str, required=True,
                                help="Path to the JSON file with adjacency lists")
    finetune_parser.add_argument("--image_dir", type=str, required=True,
                                help="Directory containing the images")
    finetune_parser.add_argument("--batch_size", type=int, default=None,
                                help="Batch size for fine-tuning")
    finetune_parser.add_argument("--epochs", type=int, default=None,
                                help="Number of epochs for fine-tuning")
    finetune_parser.add_argument("--learning_rate", type=float, default=None,
                                help="Learning rate for fine-tuning")
    finetune_parser.add_argument("--k_folds", type=int, default=None,
                                help="Number of folds for cross-validation")
    finetune_parser.add_argument("--model_dir", type=str, default=None,
                                help="Directory to save fine-tuned models")
    finetune_parser.add_argument("--freeze_backbone", action="store_true",
                                help="Freeze the backbone network during fine-tuning")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--eval_model", type=str, required=True,
                            help="Path to the model to evaluate")
    eval_parser.add_argument("--eval_json", type=str, required=True,
                            help="Path to the JSON file with test adjacency lists")
    eval_parser.add_argument("--eval_dir", type=str, required=True,
                            help="Directory containing the test images")
    eval_parser.add_argument("--batch_size", type=int, default=None,
                            help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
    else:
        main(args)