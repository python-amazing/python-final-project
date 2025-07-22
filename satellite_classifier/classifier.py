# classifier.py
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO
from typing import Dict, List, Optional

from torchgeo.trainers import ClassificationTask
from torchgeo.models import get_model_weights, ResNet18_Weights, ResNet50_Weights, ViTSmall16_Weights # Import specific weights for input channel info

from .constants import DEFAULT_CLASS_NAMES, PREPROCESSORS, IMAGENET_EARTH_OBS_INDICES # Assuming these exist

def _load_image(image_path_or_url):
    """
    Helper function to load an image from a local path or URL.
    """
    try:
        if image_path_or_url.startswith('http'):
            response = requests.get(image_path_or_url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(image_path_or_url).convert('RGB')
        print(f"✓ Loaded image: {img.size}")
        return img
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return None

def classify_satellite_image(sensor: str, models: Dict[str, torch.nn.Module], image_path_or_url: str, class_names: Optional[Dict[str, List[str]]] = None):
    """
    Classify a satellite image using multiple pretrained models.
    
    Args:
        sensor (str): The sensor type (e.g., 'Sentinel-2', 'LandSat').
        models (dict): Dictionary of loaded models (torch.nn.Module instances).
        image_path_or_url (str): Path to local image or URL to satellite image
        class_names (dict, optional): Dictionary mapping model names to their class names.
                                     Defaults to DEFAULT_CLASS_NAMES.
        
    Returns:
        dict: Dictionary with model names as keys and prediction results as values
    """
    
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES
    
    img = _load_image(image_path_or_url)
    if img is None:
        return {}
    
    results = {}
    
    with torch.no_grad():
        for model_name, model_instance in models.items():
            try:
                # Determine in_channels and num_classes for ClassificationTask
                # This part is crucial and might need more robust logic
                # depending on how your model names map to specific TorchGeo weights.
                # For simplicity, we'll assume standard RGB (3 channels) or try to
                # infer from weights if specific weight enums are used.
                
                # A more robust way would be to pass in_channels and num_classes
                # when loading models, or derive them from TorchGeo's weight metadata.
                in_channels = 3 # Default to RGB
                num_classes = None # Will try to infer from model or class_names
                
                # Try to infer in_channels from common TorchGeo weights if possible
                if "resnet18" in model_name:
                    weights_enum = ResNet18_Weights.SENTINEL2_RGB_MOCO
                    in_channels = weights_enum.meta["in_chans"]
                    num_classes = len(DEFAULT_CLASS_NAMES['ResNet18_ImageNet']) # Or a more dynamic way if possible
                    preprocess = weights_enum.transforms()
                    model_classes = class_names.get(model_name, DEFAULT_CLASS_NAMES['ResNet18_ImageNet'])
                    use_imagenet_filtering = True
                elif "resnet50" in model_name:
                    weights_enum = ResNet50_Weights.SENTINEL2_ALL_MOCO
                    in_channels = weights_enum.meta["in_chans"]
                    num_classes = len(DEFAULT_CLASS_NAMES['ResNet50_ImageNet'])
                    preprocess = weights_enum.transforms()
                    model_classes = class_names.get(model_name, ['Unknown'] * 10) # Fallback
                    use_imagenet_filtering = False
                elif "vit" in model_name:
                    if sensor == 'Sentinel-2':
                        weights_enum = ViTSmall16_Weights.SENTINEL2_ALL_MOCO
                    elif sensor == 'LandSat':
                        weights_enum = ViTSmall16_Weights.LANDSAT_ETM_SR_MOCO
                    else:
                        weights_enum = ViTSmall16_Weights.IMAGENET1K_V1
                    in_channels = weights_enum.meta["in_chans"]
                    preprocess = weights_enum.transforms()
                    model_classes = class_names.get(model_name, DEFAULT_CLASS_NAMES['ResNet18_ImageNet']) # Or a ViT specific default
                    use_imagenet_filtering = True
                else:
                    preprocess = PREPROCESSORS['satellite']
                    model_classes = class_names.get(model_name, ['Unknown'] * 10) # Fallback if no specific classes
                    use_imagenet_filtering = False
                    # If num_classes is still None, try to get it from the model's output layer
                    if num_classes is None:
                        try:
                            # This is a heuristic and might not work for all models.
                            # It tries to find a final linear layer and get its out_features.
                            if hasattr(model_instance, 'fc') and isinstance(model_instance.fc, torch.nn.Linear):
                                num_classes = model_instance.fc.out_features
                            elif hasattr(model_instance, 'head') and isinstance(model_instance.head, torch.nn.Linear):
                                num_classes = model_instance.head.out_features
                            elif hasattr(model_instance, 'classifier') and isinstance(model_instance.classifier, torch.nn.Linear):
                                num_classes = model_instance.classifier.out_features
                            # Add more checks for common last layers in different architectures if needed
                            if num_classes is None: # If not found, use length of provided class names
                                num_classes = len(model_classes)
                        except Exception:
                            print(f"Warning: Could not infer num_classes for {model_name}. Using len(model_classes).")
                            num_classes = len(model_classes)


                if num_classes is None or num_classes == 0:
                    print(f"Skipping {model_name}: Could not determine number of classes.")
                    results[model_name] = {'error': 'Could not determine number of classes'}
                    continue
                
                # Instantiate ClassificationTask for the specific model
                # Note: 'loss' and 'lr' are not strictly necessary for pure inference,
                # but ClassificationTask expects them. 'ce' (CrossEntropy) is a common default.
                classification_task = ClassificationTask(
                    model=model_instance,
                    in_channels=in_channels,
                    num_classes=num_classes,
                    loss='ce' # Dummy loss for inference
                )
                classification_task.eval() # Ensure the task is in evaluation mode

                # Preprocess image
                img_tensor = preprocess(img).unsqueeze(0)
                
                # Forward pass using the ClassificationTask
                # Directly use the encapsulated model (simpler for inference)
                outputs = classification_task.model(img_tensor)
                
                # Handle different output formats (e.g., timm models sometimes have 'logits')
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=1)
                
                predictions = []
                
                if use_imagenet_filtering:
                    # For ImageNet models, filter to relevant earth observation classes only
                    for idx, class_name in IMAGENET_EARTH_OBS_INDICES.items():
                        if idx < probs.shape[1]:
                            prob = probs[0][idx].item()
                            predictions.append({
                                'class': class_name,
                                'confidence': prob,
                                'index': idx
                            })
                    predictions.sort(key=lambda x: x['confidence'], reverse=True)
                    predictions = predictions[:10] # Top 10 of the filtered classes
                else:
                    # For custom models, use all outputs and their corresponding class names
                    for i in range(min(len(model_classes), probs.shape[1])):
                        prob = probs[0][i].item()
                        predictions.append({
                            'class': model_classes[i],
                            'confidence': prob,
                            'index': i
                        })
                    predictions.sort(key=lambda x: x['confidence'], reverse=True)
                
                results[model_name] = {
                    'predictions': predictions,
                    'raw_output': logits.cpu().numpy(),
                    'probabilities': probs.cpu().numpy(),
                    'num_classes': len(model_classes) # Use the class_names length for reporting
                }
                
                print(f"✓ Classification completed for {model_name} ({len(model_classes)} classes)")
                
            except Exception as e:
                print(f"✗ Error with model {model_name}: {e}")
                results[model_name] = {'error': str(e)}
    
    return results