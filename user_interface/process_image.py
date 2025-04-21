import torch
import torchvision.transforms.v2 as v2
from PIL import Image
import torchvision.transforms.functional as F
import cv2
import numpy as np

# Default target size
target_size = (224, 224)

def preprocess_for_classification(image):
    """
    Process an uploaded image for classification with the RGB model.
    
    Args:
        image: PIL Image
        
    Returns:
        torch.Tensor: Preprocessed image tensor ready for the model
    """
    # Ensure image is in RGB format
    image = image.convert('RGB')
    
    # Resize
    image = image.resize((224, 224))
    
    # Convert to tensor and normalize
    tensor = F.to_tensor(image)
    
    # Normalize with ImageNet mean and std
    tensor = F.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor

def process_uploaded_image(image):
    """
    Basic processing for display purposes only.
    
    Args:
        image: PIL Image
        
    Returns:
        PIL Image: Processed image for display
    """
    # Ensure image is in RGB format
    image = image.convert('RGB')
    
    # Resize for display (keeping aspect ratio)
    image.thumbnail((500, 500))
    
    return image