"""
Quick inference script for testing GraphCNN model performance.

This script provides a simple way to test a trained model on individual images 
and visualize the predicted graph structure.
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Check if the required packages are installed
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class SimplifiedGraphCNN(nn.Module):
    """CNN model for predicting graph structure (adjacency matrix) from images.
    
    Args:
        num_nodes (int): Number of nodes in the graph.
        pretrained (bool): Whether to use pretrained weights for ResNet18.
    """
    def __init__(self, num_nodes, pretrained=True):
        super(SimplifiedGraphCNN, self).__init__()
        self.num_nodes = num_nodes
        
        # Load pretrained ResNet18 and remove the final layer
        base_model = torchvision.models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(base_model.children())[:-1]) 

        self.final_layer = nn.Linear(base_model.fc.in_features, num_nodes * num_nodes)

    def forward(self, x):
        x = self.features(x)  # Output shape: (batch_size, 512, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 512)
        x = self.final_layer(x)    # Shape: (batch_size, num_nodes * num_nodes)
        return x.view(-1, self.num_nodes, self.num_nodes)


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


def process_image(image_path, image_size, device):
    """Process an image for inference.
    
    Args:
        image_path (str): Path to the image.
        image_size (int): Size to resize the image to.
        device: Device to process the image on.
        
    Returns:
        tuple: (image_tensor, original_image)
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    try:
        original_image = Image.open(image_path).convert('RGB')
        image_tensor = transform(original_image).unsqueeze(0).to(device)
        return image_tensor, original_image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        sys.exit(1)


def predict_graph(model, image_tensor, threshold=0.5):
    """Predict adjacency matrix for an image.
    
    Args:
        model: Trained model.
        image_tensor: Preprocessed image tensor.
        threshold: Threshold for binarizing the prediction.
        
    Returns:
        np.ndarray: Predicted adjacency matrix.
        np.ndarray: Raw predictions (probabilities).
    """
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.sigmoid(output)
        adjacency_matrix = (probabilities > threshold).float()
    
    return adjacency_matrix.squeeze().cpu().numpy(), probabilities.squeeze().cpu().numpy()


def print_adjacency_matrix(adjacency_matrix):
    """Print the adjacency matrix in a readable format.
    
    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix to print.
    """
    print("\nPredicted Adjacency Matrix:")
    print("-" * 50)
    
    # Print column headers
    print("    ", end="")
    for j in range(adjacency_matrix.shape[1]):
        print(f"{j+1:2d} ", end="")
    print()
    
    # Print rows with row headers
    for i in range(adjacency_matrix.shape[0]):
        print(f"{i+1:2d} | ", end="")
        for j in range(adjacency_matrix.shape[1]):
            print(f"{int(adjacency_matrix[i, j]):1d}  ", end="")
        print()
    
    print("-" * 50)


def visualize_graph(adjacency_matrix, original_image=None, output_path=None):
    """Visualize the graph structure from an adjacency matrix.
    
    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix.
        original_image (PIL.Image, optional): Original image for reference.
        output_path (str, optional): Path to save the visualization.
    """
    if not HAS_NETWORKX:
        print("NetworkX is not installed. Cannot visualize graph.")
        print("Install with: pip install networkx")
        return
    
    # Create graph from adjacency matrix
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(adjacency_matrix.shape[0]):
        G.add_node(i+1)  # Use 1-based indexing for visualization
    
    # Add edges
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] > 0:
                G.add_edge(i+1, j+1)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    if original_image:
        # Plot the original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Plot the graph
        plt.subplot(1, 2, 2)
    
    # Layout for the graph visualization
    pos = nx.spring_layout(G, seed=42)
    
    nx.draw(G, pos, with_labels=True, 
            node_color='lightblue', 
            node_size=500, 
            arrowsize=15, 
            arrows=True,
            connectionstyle='arc3,rad=0.1',
            font_weight='bold')
    
    plt.title("Predicted Graph Structure")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()


def export_adjacency_list(adjacency_matrix, output_path):
    """Export the adjacency list to a text file.
    
    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix.
        output_path (str): Path to save the adjacency list.
    """
    with open(output_path, 'w') as f:
        f.write("# Adjacency List (Node: [Neighbors])\n")
        f.write("# Format: node -> neighbors\n\n")
        
        for i in range(adjacency_matrix.shape[0]):
            neighbors = [j+1 for j in range(adjacency_matrix.shape[1]) if adjacency_matrix[i, j] > 0]
            f.write(f"{i+1} -> {neighbors}\n")
    
    print(f"Adjacency list exported to {output_path}")


def run_inference(args):
    """Run inference on a single image.
    
    Args:
        args: Command line arguments.
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, args.num_nodes, device)
    
    # Process image
    image_tensor, original_image = process_image(args.image_path, args.image_size, device)
    
    # Get prediction
    adjacency_matrix, probabilities = predict_graph(model, image_tensor, args.threshold)
    
    # Print the result
    print_adjacency_matrix(adjacency_matrix)
    
    # Calculate basic statistics
    num_edges = np.sum(adjacency_matrix)
    num_possible_edges = args.num_nodes * args.num_nodes
    sparsity = 1.0 - (num_edges / num_possible_edges)
    
    print(f"Graph Statistics:")
    print(f"- Number of nodes: {args.num_nodes}")
    print(f"- Number of edges: {int(num_edges)}")
    print(f"- Graph density: {1.0 - sparsity:.2f}")
    print(f"- Graph sparsity: {sparsity:.2f}")
    
    # Export adjacency list if requested
    if args.export_adj_list:
        output_dir = os.path.dirname(args.export_adj_list)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        export_adjacency_list(adjacency_matrix, args.export_adj_list)
    
    # Visualize if requested
    if args.visualize or args.output_image:
        output_image = args.output_image if args.output_image else None
        visualize_graph(adjacency_matrix, original_image, output_image)
    
    return adjacency_matrix


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Quick inference for GraphCNN model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model weights")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to the input image")
    parser.add_argument("--num_nodes", type=int, default=17,
                       help="Number of nodes in the graph")
    parser.add_argument("--image_size", type=int, default=256,
                       help="Size to resize the image to")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for binarizing the adjacency matrix")
    parser.add_argument("--visualize", action="store_true",
                       help="Visualize the predicted graph")
    parser.add_argument("--output_image", type=str,
                       help="Path to save the graph visualization")
    parser.add_argument("--export_adj_list", type=str,
                       help="Path to export the adjacency list")
    parser.add_argument("--cpu", action="store_true",
                       help="Force using CPU even if CUDA is available")
    
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()