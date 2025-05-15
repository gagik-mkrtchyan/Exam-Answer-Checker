from ultralytics import YOLO

def train_node_detection_model():
    """
    Train YOLOv8s model for node detection task
    This is part of a larger exam answer checker system
    """
    # Initialize the node detection model (using YOLOv8s)
    model = YOLO('yolov8s.pt')

    # Train the model with specified hyperparameters
    results = model.train(
        # Data and model configuration
        data='/home/gagik/Documents/Exam answer checker/graph_verticies.v5i.yolov8/data.yaml',
        
        # Training configuration
        epochs=60,
        patience=100,
        batch=16,
        imgsz=640,
        
        # Output configuration
        save=True,
        save_period=-1,
        project=None,
        name='train90',
        exist_ok=False,
        
        # Training settings
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=0,
        deterministic=True,
        
        # Model settings
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=None,
        multi_scale=False,
        
        # Mask settings
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        
        # Validation settings
        val=True,
        split='val',
        save_json=False,
        save_hybrid=False,
        conf=None,
        iou=0.7,
        max_det=300,
        half=False,
        dnn=False,
        plots=True,
        
        # Data loading settings
        cache=False,
        device=None,
        workers=1,
        
        # Learning rate and optimizer settings
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Loss function weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        nbs=64,
        
        # Data augmentation settings
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        copy_paste_mode='flip',
        auto_augment='randaugment',
        erasing=0.4,
        crop_fraction=1.0,
    )
    
    # Print training results
    print("Node Detection Model training completed!")
    print(f"Results saved to: {results.save_dir}")
    return results

if __name__ == "__main__":
    # Execute node detection model training
    train_results = train_node_detection_model()
    print(f"Node detection model training finished. Best model saved to: {train_results.save_dir}")
