from ultralytics import YOLO
import argparse
import os

def evaluate_paragraph_identification_model(model_path, data_path, 
                                            imgsz=640, 
                                            conf=0.001, 
                                            iou=0.6,
                                            device=None,
                                            save_json=True,
                                            save_hybrid=False,
                                            verbose=True):
    """
    Evaluate trained YOLOv9s-DocLayNet model for paragraph identification task
    Part of the exam answer checker system
    
    Args:
        model_path (str): Path to trained model (.pt file)
        data_path (str): Path to data.yaml file
        imgsz (int): Image size for evaluation
        conf (float): Confidence threshold
        iou (float): IoU threshold for NMS
        device (str): Device for evaluation ('cpu', 'cuda', or None for auto)
        save_json (bool): Save results in JSON format for further analysis
        save_hybrid (bool): Save hybrid version of labels
        verbose (bool): Print detailed evaluation results
    
    Returns:
        dict: Evaluation metrics
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Evaluate the model
    print(f"Evaluating paragraph identification model: {model_path}")
    results = model.val(
        data=data_path,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save_json=save_json,
        save_hybrid=save_hybrid,
        verbose=verbose,
        split='val',  # Use validation split
        half=False,   # Use full precision
        plots=True    # Generate plots
    )
    
    # Print evaluation results
    if verbose:
        print("\n=== Paragraph Identification Model Evaluation Results ===")
        print(f"Model: {model_path}")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall: {results.box.mr:.4f}")
        print(f"F1-Score: {(2 * results.box.mp * results.box.mr) / (results.box.mp + results.box.mr):.4f}")
        print(f"Results saved to: {results.save_dir}")
    
    return results

def evaluate_on_test_images(model_path, test_images_dir, 
                          output_dir="runs/detect/predict",
                          conf=0.25,
                          iou=0.45,
                          device=None,
                          save_txt=True,
                          save_conf=True,
                          save_crop=False,
                          show_labels=True,
                          show_conf=True,
                          visualize=True):
    """
    Run inference on test images and save predictions
    
    Args:
        model_path (str): Path to trained model
        test_images_dir (str): Directory containing test images
        output_dir (str): Directory to save predictions
        conf (float): Confidence threshold for predictions
        iou (float): IoU threshold for NMS
        device (str): Device for inference
        save_txt (bool): Save prediction results in txt format
        save_conf (bool): Save confidence values
        save_crop (bool): Save cropped detection regions
        show_labels (bool): Show labels in visualizations
        show_conf (bool): Show confidence scores in visualizations
        visualize (bool): Create visualizations of predictions
    
    Returns:
        list: Prediction results
    """
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    print(f"Running inference on images in: {test_images_dir}")
    results = model.predict(
        source=test_images_dir,
        save=True,
        save_txt=save_txt,
        save_conf=save_conf,
        save_crop=save_crop,
        conf=conf,
        iou=iou,
        device=device,
        show_labels=show_labels,
        show_conf=show_conf,
        project=output_dir,
        name="paragraph_identification_inference",
        exist_ok=True
    )
    
    print(f"Inference completed. Results saved to: {output_dir}/paragraph_identification_inference")
    return results

def analyze_document_layout(model_path, document_image, 
                          show_layout=True,
                          save_layout=True,
                          conf=0.3,
                          iou=0.5):
    """
    Analyze document layout and identify paragraphs in a single document
    Useful for document layout analysis and exam answer checking
    
    Args:
        model_path (str): Path to trained model
        document_image (str): Path to document image
        show_layout (bool): Display the layout analysis
        save_layout (bool): Save the annotated layout
        conf (float): Confidence threshold
        iou (float): IoU threshold
    
    Returns:
        list: Detected paragraphs with their coordinates
    """
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference on single document
    results = model.predict(
        source=document_image,
        save=save_layout,
        show=show_layout,
        conf=conf,
        iou=iou,
        project="layout_analysis",
        name="document_layout",
        exist_ok=True
    )
    
    # Extract paragraph information
    paragraphs = []
    if results and len(results) > 0:
        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    # Extract box coordinates and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    paragraphs.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': cls,
                        'class_name': model.names[cls] if hasattr(model, 'names') else f'class_{cls}'
                    })
    
    return paragraphs

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Evaluate Paragraph Identification Model")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to trained model (.pt file)")
    parser.add_argument("--data", type=str, 
                       default="/home/gagik/Documents/Exam answer checker/My First Project.v2i.yolov8/data.yaml",
                       help="Path to dataset configuration file")
    parser.add_argument("--task", type=str, choices=["val", "test", "analyze", "both"], 
                       default="val", help="Evaluation task")
    parser.add_argument("--test-images", type=str, 
                       help="Directory with test images (for test task)")
    parser.add_argument("--document", type=str, 
                       help="Single document image for analysis (for analyze task)")
    parser.add_argument("--conf", type=float, default=0.001, 
                       help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, 
                       help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, 
                       help="Image size")
    parser.add_argument("--device", type=str, default=None, 
                       help="Device (cpu, cuda, etc.)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Print detailed results")
    
    args = parser.parse_args()
    
    if args.task in ["val", "both"]:
        # Run validation evaluation
        print("Running validation evaluation...")
        evaluate_paragraph_identification_model(
            model_path=args.model,
            data_path=args.data,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=args.verbose
        )
    
    if args.task in ["test", "both"]:
        # Run test inference
        if not args.test_images:
            print("Error: --test-images must be specified for test task")
            return
        
        print("Running test inference...")
        evaluate_on_test_images(
            model_path=args.model,
            test_images_dir=args.test_images,
            conf=0.25,  # Higher confidence for final predictions
            iou=0.45,
            device=args.device
        )
    
    if args.task == "analyze":
        # Analyze single document
        if not args.document:
            print("Error: --document must be specified for analyze task")
            return
        
        print("Analyzing document layout...")
        paragraphs = analyze_document_layout(
            model_path=args.model,
            document_image=args.document,
            conf=0.3,
            iou=0.5
        )
        
        print(f"Found {len(paragraphs)} text regions:")
        for i, para in enumerate(paragraphs):
            print(f"  {i+1}. {para['class_name']}: confidence={para['confidence']:.3f}")

if __name__ == "__main__":
    # Example usage if running directly
    # main()
    
    # Or run programmatically
    model_path = "doc_pretrained/train31/weights/best.pt"  # Path to your trained paragraph identification model
    
    # Evaluate on validation set
    results = evaluate_paragraph_identification_model(
        model_path=model_path,
        data_path="/home/gagik/Documents/Exam answer checker/My First Project.v2i.yolov8/data.yaml",
        verbose=True
    )
    
    # Analyze a single document (if you have a document to analyze)
    # paragraphs = analyze_document_layout(
    #     model_path=model_path,
    #     document_image="path/to/exam_answer.jpg"
    # )
