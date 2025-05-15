"""
Batch inference script for processing multiple images with a trained GraphCNN model.

This script allows you to run inference on a directory of images and save the results.
"""

import argparse
import os
import sys
import json
import glob
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from quick_inference import SimplifiedGraphCNN, process_image, predict_graph


def find_images(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """Find all images in a directory with specified extensions.
    
    Args:
        directory (str): Directory to search for images.
        extensions (tuple): File extensions to include.
        
    Returns:
        list: List of image paths.
    """
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    
    return sorted(image_paths)


def batch_predict(model, image_paths, image_size, threshold, device):
    """Run batch prediction on multiple images.
    
    Args:
        model: Trained model.
        image_paths (list): List of image paths.
        image_size (int): Size to resize images to.
        threshold (float): Threshold for binarizing predictions.
        device: Device to run inference on.
        
    Returns:
        dict: Dictionary mapping image paths to adjacency matrices.
    """
    results = {}
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Process image
            image_tensor, _ = process_image(image_path, image_size, device)
            
            # Get prediction
            adjacency_matrix, _ = predict_graph(model, image_tensor, threshold)
            
            # Store result
            results[os.path.basename(image_path)] = adjacency_matrix
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    return results


def save_results_as_adjacency_lists(results, output_path):
    """Save results as adjacency lists in JSON format.
    
    Args:
        results (dict): Dictionary mapping image paths to adjacency matrices.
        output_path (str): Path to save the JSON file.
    """
    adjacency_lists = {}
    
    for image_name, adjacency_matrix in results.items():
        # Convert adjacency matrix to adjacency list
        adj_list = {}
        for i in range(adjacency_matrix.shape[0]):
            # Get list of neighbors (1-indexed)
            neighbors = [j+1 for j in range(adjacency_matrix.shape[1]) 
                       if adjacency_matrix[i, j] > 0]
            
            # Only include nodes with neighbors
            if neighbors:
                adj_list[str(i+1)] = neighbors
        
        adjacency_lists[image_name] = adj_list
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(adjacency_lists, f, indent=2)
    
    print(f"Results saved to {output_path}")


def save_results_as_numpy(results, output_dir):
    """Save results as individual numpy files.
    
    Args:
        results (dict): Dictionary mapping image paths to adjacency matrices.
        output_dir (str): Directory to save numpy files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for image_name, adjacency_matrix in results.items():
        base_name = os.path.splitext(image_name)[0]
        output_path = os.path.join(output_dir, f"{base_name}.npy")
        np.save(output_path, adjacency_matrix)
    
    print(f"Results saved as numpy files in {output_dir}")


def load_model(model_path, num_nodes, device):
    """Load a trained model.
    
    Args:
        model_path (str): Path to the model weights.
        num_nodes (int): Number of nodes in the graph.
        device: Device to load the model on.
        
    Returns:
        SimplifiedGraphCNN: Loaded model.
    """
    model = SimplifiedGraphCNN(num_nodes).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model successfully loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    return model


def run_batch_inference(args):
    """Run batch inference on a directory of images.
    
    Args:
        args: Command line arguments.
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, args.num_nodes, device)
    
    # Find all images in the directory
    image_paths = find_images(args.image_dir, extensions=args.extensions.split(','))
    print(f"Found {len(image_paths)} images in {args.image_dir}")
    
    if not image_paths:
        print("No images found. Check the directory and extensions.")
        return
    
    # Run batch prediction
    results = batch_predict(model, image_paths, args.image_size, args.threshold, device)
    print(f"Processed {len(results)} images successfully")
    
    # Save results
    if args.output_json:
        save_results_as_adjacency_lists(results, args.output_json)
    
    if args.output_dir:
        save_results_as_numpy(results, args.output_dir)
    
    print("Batch inference completed successfully")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Batch inference for GraphCNN model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model weights")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing input images")
    parser.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png,.bmp",
                       help="Comma-separated list of image extensions to process")
    parser.add_argument("--num_nodes", type=int, default=17,
                       help="Number of nodes in the graph")
    parser.add_argument("--image_size", type=int, default=256,
                       help="Size to resize the images to")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for binarizing the adjacency matrix")
    parser.add_argument("--output_json", type=str,
                       help="Path to save results as a JSON file")
    parser.add_argument("--output_dir", type=str,
                       help="Directory to save individual numpy result files")
    parser.add_argument("--cpu", action="store_true",
                       help="Force using CPU even if CUDA is available")
    
    args = parser.parse_args()
    
    # Ensure at least one output format is specified
    if not args.output_json and not args.output_dir:
        print("Error: At least one of --output_json or --output_dir must be specified")
        parser.print_help()
        sys.exit(1)
    
    run_batch_inference(args)


if __name__ == "__main__":
    main()