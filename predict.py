import argparse
import json
import numpy as np
import torch
from torchvision import models
from PIL import Image
import torch.nn.functional as F

# Function to load model checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')  # Ensure compatibility with CPU if GPU not available
    model = getattr(models, checkpoint['arch'])(pretrained=True)  # Dynamically load the architecture used during training
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

# Function to process an image
def process_image(image_path):
    image = Image.open(image_path)
    
    # Resize and crop image to 224x224 as required by most models
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    
    # Convert image to a numpy array and normalize
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))  # PyTorch expects (C, H, W)
    
    return torch.tensor(np_image).float()

# Function to predict top K classes
def predict(image_path, model, top_k, device):
    model.to(device)
    model.eval()
    
    # Process image and move to appropriate device
    image = process_image(image_path).unsqueeze_(0).to(device)
    
    # No gradient tracking needed for inference
    with torch.no_grad():
        output = model.forward(image)
    
    # Get probabilities using softmax
    probabilities = torch.exp(output)
    top_p, top_class = probabilities.topk(top_k, dim=1)
    
    return top_p.cpu().numpy(), top_class.cpu().numpy()

# Main function to handle command-line arguments and output
def main():
    # Command Line Arguments
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with probability.')
    
    parser.add_argument('input_img', type=str, help='Path to the input image.')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint.')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes.')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to JSON file mapping categories to names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available.')
    
    args = parser.parse_args()

    # Set device to GPU or CPU
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load the trained model from the checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # Load the category-to-name mapping from the JSON file
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Predict top K classes for the input image
    top_p, top_classes = predict(args.input_img, model, args.top_k, device)
    
    # Convert class indices back to the original labels
    class_to_idx_inverted = {v: k for k, v in model.class_to_idx.items()}
    top_class_names = [cat_to_name[class_to_idx_inverted[c]] for c in top_classes[0]]
    
    # Display the predictions
    for i in range(args.top_k):
        print(f"Class: {top_class_names[i]}, Probability: {top_p[0][i]:.3f}")

if __name__ == "__main__":
    main()

    
    # Map class indices to actual flower names
    class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    top_class_names = [cat_to_name[class_to_idx[str(c)]] for c in top_classes[0]]
    
    # Print out the results
    for i in range(args.top_k):
        print(f"Class: {top_class_names[i]}, Probability: {top_p[0][i]:.3f}")

if __name__ == "__main__":
    main()
