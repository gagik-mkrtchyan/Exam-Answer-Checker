import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import itertools
import os
import numpy as np
import math
import glob
from tqdm import tqdm  # Progress bar for the processing
import argparse

# Function to load YOLO model
def load_yolo_model(model_path):
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print(f"Model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_images(input_dir, output_base_dir, model_path, box_expansion=1.2):
    # Load YOLOv8 model
    model = load_yolo_model(model_path)
    if model is None:
        return
    
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Find all images in the input directory
    image_paths = glob.glob(os.path.join(input_dir, "*.png")) + \
                 glob.glob(os.path.join(input_dir, "*.jpg")) + \
                 glob.glob(os.path.join(input_dir, "*.jpeg"))

    print(f"Found {len(image_paths)} images to process.")

    # Process each image
    for img_index, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
        # Extract filename without extension for organizing outputs
        base_filename = os.path.basename(image_path)
        filename_no_ext = os.path.splitext(base_filename)[0]
        
        # Create output directory for this image
        output_dir = os.path.join(output_base_dir, filename_no_ext)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Run detection on the image
            results = model(image_path)
            boxes = results[0].boxes.xyxy
            
            if len(boxes) == 0:
                print(f"⚠️ No boxes detected in {base_filename}, skipping.")
                continue
            
            # Load the image
            base_image = Image.open(image_path).convert("RGB")
            width, height = base_image.size
            
            # Optional: Save a visualization of all detected boxes
            viz_image = base_image.copy()
            draw_viz = ImageDraw.Draw(viz_image)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.tolist()
                draw_viz.rectangle([x1, y1, x2, y2], outline="red", width=2)
                # Add box number for reference
                draw_viz.text((x1, y1), f"{i}", fill="white")
            
            viz_path = os.path.join(output_dir, f"{filename_no_ext}_boxes.png")
            viz_image.save(viz_path)
            
            # Generate corridors for all pairs of boxes
            corridor_count = 0
            
            for idx, (i, j) in enumerate(itertools.combinations(range(len(boxes)), 2)):
                # Get the boxes
                box1 = boxes[i]
                box2 = boxes[j]

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
                # Calculating enlarged bounds
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
                min_corridor_width = max(15, box1_radius, box2_radius)
                
                # Calculate angle of the line connecting the centers
                angle = math.atan2(center2_y - center1_y, center2_x - center1_x)
                
                # Calculate perpendicular offset vectors for corridor width
                dx = min_corridor_width * math.sin(angle)
                dy = -min_corridor_width * math.cos(angle)
                
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
                    
                    # Save the cropped corridor image
                    corridor_filename = f"corridor_{i}_{j}.png"
                    corridor_path = os.path.join(output_dir, corridor_filename)
                    final_img.save(corridor_path)
                    corridor_count += 1
                    
                    # Optional: Save a visualization showing the corridor
                    if idx % 10 == 0:  # Save visualization only for some corridors to save space
                        img_with_corridor = base_image.copy()
                        draw = ImageDraw.Draw(img_with_corridor)
                        
                        # Draw original boxes
                        draw.rectangle(box1.tolist(), outline="red", width=2)
                        draw.rectangle(box2.tolist(), outline="red", width=2)
                        
                        # Draw expanded boxes
                        expanded_box1 = [x1_min_expanded, y1_min_expanded, x1_max_expanded, y1_max_expanded]
                        expanded_box2 = [x2_min_expanded, y2_min_expanded, x2_max_expanded, y2_max_expanded]
                        draw.rectangle(expanded_box1, outline="yellow", width=2)
                        draw.rectangle(expanded_box2, outline="yellow", width=2)
                        
                        # Draw corridor and centerline
                        draw.polygon(polygon, outline="blue", width=1)
                        draw.line([(center1_x, center1_y), (center2_x, center2_y)], fill="green", width=2)
                        
                        viz_corridor_path = os.path.join(output_dir, f"viz_corridor_{i}_{j}.png")
                        img_with_corridor.save(viz_corridor_path)
                
            print(f"✅ Processed {base_filename}: extracted {corridor_count} corridors")
            
        except Exception as e:
            print(f"❌ Error processing {base_filename}: {str(e)}")

    print(f"Completed processing {len(image_paths)} images.")
    print(f"Results saved to {output_base_dir}")

def main():
    parser = argparse.ArgumentParser(description='Extract corridors between nodes in graph images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to YOLO model weights')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='line_corridors_batch', help='Directory to save output')
    parser.add_argument('--box_expansion', type=float, default=1.2, help='Box expansion factor (default: 1.2)')
    
    args = parser.parse_args()
    
    # Process images with parameters from command line
    process_images(
        input_dir=args.input_dir,
        output_base_dir=args.output_dir,
        model_path=args.model_path,
        box_expansion=args.box_expansion
    )

if __name__ == "__main__":
    main()