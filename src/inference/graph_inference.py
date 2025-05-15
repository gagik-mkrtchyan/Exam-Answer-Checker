"""
Graph inference - Extract adjacency matrix from a graph image.
"""

import itertools
import math
import cv2
import numpy as np
import os
import argparse
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms


def cross(o, a, b):
    """Calculate cross product for point-in-polygon test."""
    return (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0])


def is_point_in_rectangle(point, rect_points):
    """
    Test if a point is inside a rectangle.
    
    Args:
        point: (x, y) tuple
        rect_points: list or array of 4 (x, y) points in order
    
    Returns:
        bool: True if point is inside rectangle
    """
    signs = []
    for i in range(4):
        o = rect_points[i]
        a = rect_points[(i + 1) % 4]
        signs.append(cross(o, a, point))
    
    # If all cross products have the same sign, point is inside
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


def load_models(node_model_path, edge_model_path, num_classes=2):
    """
    Load both node detection and edge classification models.
    
    Args:
        node_model_path: Path to YOLO node detection model
        edge_model_path: Path to edge classification model
        
    Returns:
        tuple: (node_model, edge_model, device)
    """
    # Load node detection model (YOLO)
    try:
        from ultralytics import YOLO
        node_model = YOLO(node_model_path)
        print(f"Node detection model loaded from {node_model_path}")
    except Exception as e:
        print(f"Error loading node detection model: {e}")
        return None, None, None
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load edge classification model (ConvNeXt)
    try:
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        
        # Load the model
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        edge_model = convnext_tiny(weights=weights)
        
        # Replace the classifier
        edge_model.classifier[2] = torch.nn.Linear(edge_model.classifier[2].in_features, num_classes)
        
        # Load saved weights
        edge_model.load_state_dict(torch.load(edge_model_path, map_location=device))
        edge_model = edge_model.to(device)
        edge_model.eval()
        
        print(f"Edge classification model loaded from {edge_model_path}")
    except Exception as e:
        print(f"Error loading edge classification model: {e}")
        return node_model, None, device
    
    return node_model, edge_model, device


def predict_image(image, model, transform, class_names, device):
    """
    Classify a corridor image using edge classification model.
    
    Args:
        image: PIL image to classify
        model: Edge classification model
        transform: Image transformations
        class_names: List of class names
        device: Computing device
        
    Returns:
        str: Predicted class
    """
    # Transform
    input_tensor = transform(image).unsqueeze(0)  # add batch dimension
    input_tensor = input_tensor.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
    
    predicted_class = class_names[preds.item()]
    return predicted_class


def extract_corridors(image_path, node_model, edge_model, device, output_dir="test_inference", 
                     box_expansion=1.5, min_corridor_width=15, show_visualization=False):
    """
    Extract corridors between bounding boxes and classify edges to build an adjacency matrix.
    
    Args:
        image_path: Path to the input image
        node_model: Pre-trained YOLO model for node detection
        edge_model: Pre-trained model for edge classification
        device: Computing device
        output_dir: Directory to save results
        box_expansion: Factor to expand boxes
        min_corridor_width: Minimum corridor width
        show_visualization: Whether to show node visualization
        
    Returns:
        numpy.ndarray: Adjacency matrix
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract filename without extension for naming outputs
    base_filename = os.path.basename(image_path)
    filename_no_ext = os.path.splitext(base_filename)[0]
    
    # Define transform for edge classification
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    # Class names for edge classification
    class_names = ['0', '1']  # No edge, Edge
    
    try:
        # Run detection on the image
        results = node_model(image_path)
        boxes = results[0].boxes.xyxy

        bbox_coordinates = sorted(boxes, key=lambda x: x[0])
        bbox_dict = {cnt + 1: coord for cnt, coord in enumerate(bbox_coordinates)}
        
        n = len(boxes)
        res_matrix = np.zeros((n, n))
        
        if len(boxes) < 2:  # Skip images with less than 2 boxes
            print(f"⚠️ Not enough boxes detected in {base_filename}, skipping.")
            return res_matrix
        
        # Load the image
        base_image = Image.open(image_path).convert("RGB")
        image = np.array(base_image)
        width, height = base_image.size
        
        scale_factor = max(width, height) / 1000
        font_scale = min(1.0, 0.7 * scale_factor)
        font_thickness = max(1, int(2 * scale_factor))

        # Draw node numbers on the image
        for key, box in bbox_dict.items():
            x_min, y_min, x_max, y_max = map(int, box)
            label_text = str(key)
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            label_x, label_y = x_min, y_min - 10

            if label_y - text_height < 0:
                label_y = y_min + text_height + 10
            if label_x + text_width > width:
                label_x = width - text_width - 10
            if label_x < 0:
                label_x = 10

            # Draw background rectangle
            cv2.rectangle(
                image,
                (label_x, label_y - text_height - baseline),
                (label_x + text_width, label_y + baseline // 2),
                (255, 255, 255),
                thickness=cv2.FILLED
            )

            # Draw label text
            cv2.putText(
                image,
                label_text,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 0, 0),
                font_thickness
            )

        # Optionally show the node visualization
        if show_visualization:
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Detected Nodes")
            plt.show()
            
        # Generate corridors for pairs of boxes
        corridor_count = 0
        
        for i, j in itertools.combinations(range(len(boxes)), 2):
            # Get the boxes
            box1 = bbox_coordinates[i]
            box2 = bbox_coordinates[j]

            # Extract coordinates
            x1_min, y1_min, x1_max, y1_max = box1.tolist()
            x2_min, y2_min, x2_max, y2_max = box2.tolist()
            
            # Calculate centers of the original boxes
            center1_x = (x1_min + x1_max) / 2
            center1_y = (y1_min + y1_max) / 2
            center2_x = (x2_min + x2_max) / 2
            center2_y = (y2_min + y2_max) / 2
            
            # Calculate dimensions
            width1 = x1_max - x1_min
            height1 = y1_max - y1_min
            width2 = x2_max - x2_min
            height2 = y2_max - y2_min
            
            # Expand the boxes while keeping their centers the same
            x1_min_expanded = center1_x - (width1 * box_expansion) / 2
            x1_max_expanded = center1_x + (width1 * box_expansion) / 2
            y1_min_expanded = center1_y - (height1 * box_expansion) / 2
            y1_max_expanded = center1_y + (height1 * box_expansion) / 2
            
            x2_min_expanded = center2_x - (width2 * box_expansion) / 2
            x2_max_expanded = center2_x + (width2 * box_expansion) / 2
            y2_min_expanded = center2_y - (height2 * box_expansion) / 2
            y2_max_expanded = center2_y + (height2 * box_expansion) / 2
            
            # Use the larger of: minimum corridor width or node size for corridor width
            box1_radius = max(width1, height1) * box_expansion / 2
            box2_radius = max(width2, height2) * box_expansion / 2
            corridor_width = max(min_corridor_width, box1_radius, box2_radius)
            
            # Calculate angle of the line connecting the centers
            angle = math.atan2(center2_y - center1_y, center2_x - center1_x)
            
            # Calculate perpendicular offset vectors for corridor width
            dx = corridor_width * math.sin(angle)
            dy = -corridor_width * math.cos(angle)
            
            # Calculate vectors along the line direction
            along_x = math.cos(angle)
            along_y = math.sin(angle)
            
            # Distance to extend beyond box centers to ensure boxes are fully included
            extension1 = max(width1, height1) * box_expansion / 1.8
            extension2 = max(width2, height2) * box_expansion / 1.8
            
            # Create the polygon for the corridor that includes the entire expanded boxes
            polygon = [
                # Box 1 side with extension
                (center1_x + dx - along_x * extension1, center1_y + dy - along_y * extension1),
                (center1_x - dx - along_x * extension1, center1_y - dy - along_y * extension1),
                
                # Box 2 side with extension
                (center2_x - dx + along_x * extension2, center2_y - dy + along_y * extension2),
                (center2_x + dx + along_x * extension2, center2_y + dy + along_y * extension2)
            ]
            # Create a mask image for the corridor
            mask_img = Image.new('L', (width, height), 0)
            draw_mask = ImageDraw.Draw(mask_img)
            draw_mask.polygon(polygon, fill=255)

            polygon = [(max(0, x), max(0, y)) for (x, y) in polygon]
            
            # Apply the mask to create the masked image
            masked_img = Image.new('RGB', (width, height), (0, 0, 0))
            masked_img.paste(base_image, (0, 0), mask_img)
            
            # Find the bounding box of the non-zero region in the mask
            mask_array = np.array(mask_img)
            non_zero_indices = np.where(mask_array > 0)
            if len(non_zero_indices[0]) > 0:  # Check if mask contains any non-zero values
                min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
                min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
                
                # Add a small padding to the crop
                crop_padding = 5
                crop_box = (
                    max(0, min_x - crop_padding),
                    max(0, min_y - crop_padding),
                    min(width, max_x + crop_padding),
                    min(height, max_y + crop_padding)
                )
                
                # Crop the masked image to get just the corridor
                final_img = masked_img.crop(crop_box)
                
                # Count how many vertices (bounding boxes) are in the corridor
                vertex_count = 0
                for box in boxes:
                    box_x_min, box_y_min, box_x_max, box_y_max = box.tolist()
                    box_center_x = (box_x_min + box_x_max) / 2
                    box_center_y = (box_y_min + box_y_max) / 2

                    point = (box_center_x, box_center_y)
                    inside = is_point_in_rectangle(point, polygon)
                    if inside:
                        vertex_count += 1
                        
                # Only process corridors with exactly 2 vertices
                if vertex_count == 2:
                    # Save the cropped corridor image
                    corridor_filename = f"{filename_no_ext}_corridor_{i}_{j}.png"
                    corridor_path = os.path.join(output_dir, corridor_filename)
                    final_img.save(corridor_path)
                    corridor_count += 1
                    
                    # Classify the corridor
                    res = predict_image(final_img, edge_model, transform, class_names, device)
                    
                    # Update adjacency matrix if edge exists
                    if int(res):
                        res_matrix[i][j] = 1
                        res_matrix[j][i] = 1
            
        print(f"✅ Processed {base_filename}: extracted {corridor_count} corridors")
        
        # Save adjacency matrix visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(res_matrix, cmap='Blues')
        plt.colorbar(label='Edge Present')
        plt.title(f"Adjacency Matrix for {filename_no_ext}")
        
        # Add node numbers
        for i in range(n):
            for j in range(n):
                text_color = 'white' if res_matrix[i][j] > 0.5 else 'black'
                plt.text(j, i, f"{int(res_matrix[i][j])}", 
                         ha="center", va="center", color=text_color)
        
        plt.tight_layout()
        matrix_path = os.path.join(output_dir, f"{filename_no_ext}_matrix.png")
        plt.savefig(matrix_path)
        
        if show_visualization:
            plt.figure(figsize=(10, 8))
            plt.imshow(res_matrix, cmap='Blues')
            plt.colorbar(label='Edge Present')
            plt.title(f"Adjacency Matrix for {filename_no_ext}")
            
            for i in range(n):
                for j in range(n):
                    text_color = 'white' if res_matrix[i][j] > 0.5 else 'black'
                    plt.text(j, i, f"{int(res_matrix[i][j])}", 
                             ha="center", va="center", color=text_color)
            
            plt.tight_layout()
            plt.show()
        
        return res_matrix
        
    except Exception as e:
        print(f"❌ Error processing {base_filename}: {str(e)}")
        return np.zeros((0, 0))


def main():
    parser = argparse.ArgumentParser(description='Extract graph structure from an image')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input graph image')
    parser.add_argument('--node_model', type=str, required=True,
                        help='Path to YOLO node detection model')
    parser.add_argument('--edge_model', type=str, required=True,
                        help='Path to edge classification model')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save results (default: inference_results)')
    parser.add_argument('--box_expansion', type=float, default=1.5,
                        help='Factor to expand node boxes (default: 1.5)')
    parser.add_argument('--min_corridor_width', type=int, default=15,
                        help='Minimum corridor width in pixels (default: 15)')
    parser.add_argument('--show', action='store_true',
                        help='Show visualizations')
    
    args = parser.parse_args()
    
    # Load models
    node_model, edge_model, device = load_models(args.node_model, args.edge_model)
    
    if node_model is None or edge_model is None:
        print("Failed to load required models. Exiting.")
        return
    
    # Process the image
    adjacency_matrix = extract_corridors(
        args.image_path, 
        node_model, 
        edge_model, 
        device,
        output_dir=args.output_dir,
        box_expansion=args.box_expansion,
        min_corridor_width=args.min_corridor_width,
        show_visualization=args.show
    )
    
    # Print adjacency matrix
    print("\nAdjacency Matrix:")
    print(adjacency_matrix)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()