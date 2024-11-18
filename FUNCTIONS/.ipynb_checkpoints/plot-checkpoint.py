# Making Predictions on a set of images from test set
import torch
from typing import Dict, List, Tuple
from PIL import Image 
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
# take a trained model, class_names, image_path, image_size, transform, and target device
device = "cuda" if torch.cuda.is_available() else "cpu"

def pred_plot_image(model: nn.Module,
                   class_names: List[str],
                   image_path: str,
                   image_size: Tuple[int, int] = (224,224),
                   transform: torchvision.transforms = None,
                   device: torch.device= device):
    # open an image:
    img = Image.open(image_path)
    
    # create transformation for the image 
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
                                        transforms.Resize(size=(224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],   
                                                             std = [0.229, 0.224, 0.225])
                                       ])   # Same transformation as on pretrained model EfficientNet_b0
    
    # Predict on image 
    model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)    # Add an extra dimension for batch
        #make predictions on image 
        target_image_pred = model(transformed_image.to(device))
    # Convert logits into prediction probabilities 
    target_image_probs = torch.softmax(target_image_pred, dim=1)
    # Convert probabilities into prediction labels 
    target_label = torch.argmax(target_image_probs, dim=1).item()
    
    
    # Plot the image 
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_label]} | Prob: {target_image_probs.max():.3f}")
    plt.axis(False)
        
    
                                                    
