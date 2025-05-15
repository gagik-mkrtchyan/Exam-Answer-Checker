import torch
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, num_classes=2):
    # Load the model
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model_second = convnext_tiny(weights=weights)

    # Set number of output classes
    model_second.classifier[2] = torch.nn.Linear(model_second.classifier[2].in_features, num_classes)

    # Load your saved best model weights
    model_second.load_state_dict(torch.load(model_path, map_location=device))
    model_second = model_second.to(device)
    model_second.eval()
    
    return model_second

# Define transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Inference function
def predict_image(image, model, transform, class_names):
    # Transform
    input_tensor = transform(image).unsqueeze(0)  # add batch dimension
    input_tensor = input_tensor.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
    
    predicted_class = class_names[preds.item()]
    return predicted_class

def main():
    parser = argparse.ArgumentParser(description='Edge classification using ConvNeXt model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--image_path', type=str, help='Path to image for prediction')
    parser.add_argument('--show_result', action='store_true', help='Show prediction visualization')
    args = parser.parse_args()

    # Load model
    model = load_model(args.model_path)
    
    # Set class names
    class_names = ['0', '1']  # Replace with your real class names
    
    # Check if image path is provided
    if args.image_path:
        # Load image
        image = Image.open(args.image_path).convert('RGB')
        
        # Predict
        predicted_class = predict_image(image, model, transform, class_names)
        print(f"Prediction: {predicted_class}")
        
        # Show image and prediction if requested
        if args.show_result:
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Prediction: {predicted_class}')
            plt.show()
    else:
        print("No image path provided. Use --image_path to specify an image.")

if __name__ == "__main__":
    main()