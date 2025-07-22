""" 
Provides functions to load pre-trained models from the TorchGeo library.
It includes error handling for model loading and provides a list of available models.
"""

import torch
from typing import Dict, List, Optional
from torchgeo.models import get_model, list_models

def load_pretrained_models(args) -> Dict[str, torch.nn.Module]:
    """
    Load pre-trained models from torchgeo library based on args.models.
    
    Args:
        args: Argument namespace containing models list (args.models)
        
    Returns:
        dict: Dictionary containing loaded models with their names as keys
    """
    models = {}
    
    if not hasattr(args, 'models') or not args.models:
        print("! No models specified in args.models")
        return models
    
    available_models = list_models()
    print(f"Available TorchGeo models: {available_models}")
    
    for model_name in args.models:
        try:
            # Load the pre-trained model from torchgeo
            model = get_model(model_name, pretrained=True)
            model.eval()
            models[model_name] = model
            print(f"✓ Loaded {model_name} from TorchGeo")
            
        except ValueError as e:
            print(f"! Model '{model_name}' not found in TorchGeo. Available models: {available_models}")
        except Exception as e:
            print(f"✗ Could not load {model_name}: {e}")
    
    print(f"\nLoaded {len(models)} models successfully")
    return models

def get_available_models() -> List[str]:
    """
    Get list of available pre-trained models from torchgeo.
    
    Returns:
        List of available model names
    """
    try:
        from torchgeo.models import list_models
        return list_models()
    except ImportError:
        print("! TorchGeo library not found. Please install with: pip install torchgeo")
        return []
    except Exception as e:
        print(f"! Error getting available models: {e}")
        return []