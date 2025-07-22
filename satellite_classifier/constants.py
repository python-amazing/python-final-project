import torch
from torchvision import transforms

# Task definitions with their corresponding models
from typing import Dict, List, Any

TASK_MODELS: Dict[str, Any] = {
    "Classification": {
        "description": "Image classification tasks.",
        "datasets": ["EuroSAT", "BigEarthNet"],
        "sensors": {
            "Sensor-Agnostic": {
                "description": "Models pre-trained on diverse remote sensing data.",
                "models": {
                    "ResNet18": ["random", "imagenet", "ssl4eo_s12_rgb_moco", "ssl4eo_s12_all_moco"],
                    "ResNet50": ["random", "imagenet", "ssl4eo_s12_rgb_moco", "ssl4eo_s12_all_dino"],
                    "ViT_Small_Patch16": ["random", "imagenet", "ssl4eo_s12_rgb_moco"],
                }
            },
            "Sentinel-1 (SAR)": {
                "description": "Models pre-trained on Sentinel-1 Synthetic Aperture Radar data.",
                "models": {
                    "ResNet50": ["ssl4eo_s1_all_moco"],
                }
            },
            "Sentinel-2 (Optical)": {
                "description": "Models pre-trained on Sentinel-2 optical imagery.",
                "models": {
                    "ResNet18": ["ssl4eo_s2_rgb_moco", "ssl4eo_s2_all_moco", "ssl4eo_s2_rgb_seco"],
                    "ResNet50": ["ssl4eo_s2_rgb_moco", "ssl4eo_s2_all_dino", "ssl4eo_s2_rgb_seco"],
                    "ViT_Small_Patch16": ["ssl4eo_s2_all_dino", "ssl4eo_s2_all_moco", "satlas_rgb"],
                }
            },
            "Landsat": {
                "description": "Models pre-trained on Landsat optical imagery.",
                "models": {
                    "ResNet50": ["ssl4eo_landsat7_l2_all_moco"],
                }
            },
            "NAIP": {
                "description": "Models pre-trained on NAIP imagery.",
                "models": {
                    "Swin_V2_B": ["satlas_rgb"],
                }
            }
        }
    },
}

# Define class names for different model types
DEFAULT_CLASS_NAMES = {
    # EuroSAT classes (10 land cover types)
    'ResNet50_EuroSAT': [
        'Annual_Crop', 'Forest', 'Herbaceous_Vegetation', 'Highway',
        'Industrial', 'Pasture', 'Permanent_Crop', 'Residential',
        'River', 'Sea_Lake'
    ],
    
    # Custom land cover classes (8 types)
    'ResNet18_LandCover': [
        'Urban', 'Forest', 'Water', 'Agriculture', 
        'Grassland', 'Barren', 'Wetland', 'Infrastructure'
    ],
    
    # Aerial imagery classes (12 types)
    'ResNet18_Aerial': [
        'Residential', 'Commercial', 'Industrial', 'Transportation',
        'Forest', 'Water', 'Agriculture', 'Recreation',
        'Vacant_Land', 'Mixed_Development', 'Infrastructure', 'Natural_Areas'
    ],
    
    # For ImageNet models, use selected relevant classes
    'ResNet18_ImageNet': [
        'Lake', 'Coastline', 'Water_Body', 'Beach_Sand', 'Shallow_Water',
        'Port_Harbor', 'Marina', 'Mountain_Peak', 'Rocky_Terrain', 'Desert_Sand',
        'Dense_Forest', 'Grassland_Pasture', 'Agricultural_Area', 'Rural_Roads',
        'Airport_Runway', 'Railway', 'Urban_Traffic', 'Urban_Plaza',
        'Commercial_District', 'Recreation_Area'
    ],
    
    'ResNet50_ImageNet': [
        'Lake', 'Coastline', 'Water_Body', 'Beach_Sand', 'Shallow_Water',
        'Port_Harbor', 'Marina', 'Mountain_Peak', 'Rocky_Terrain', 'Desert_Sand',
        'Dense_Forest', 'Grassland_Pasture', 'Agricultural_Area', 'Rural_Roads',
        'Airport_Runway', 'Railway', 'Urban_Traffic', 'Urban_Plaza',
        'Commercial_District', 'Recreation_Area'
    ]
}

# Define different preprocessing for different model types
PREPROCESSORS = {
    'standard': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]),
    'satellite': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Different normalization for satellite imagery
        transforms.Normalize(mean=[0.45, 0.45, 0.40], 
                           std=[0.22, 0.22, 0.25])
    ])
}

# For ImageNet models, filter to relevant classes only
IMAGENET_EARTH_OBS_INDICES = {
    948: 'Farmland', 949: 'Coastline', 988: 'Water_Body', 992: 'Beach_Sand',
    976: 'Shallow_Water', 511: 'Port_Harbor', 895: 'Marina',
    970: 'Mountain_Peak', 975: 'Rocky_Terrain', 977: 'Desert_Sand',
    980: 'Dense_Forest', 984: 'Grassland_Pasture', 867: 'Agricultural_Area',
    717: 'Rural_Roads', 402: 'Airport_Runway', 466: 'Railway',
    468: 'Urban_Traffic', 547: 'Urban_Plaza', 757: 'Commercial_District',
    575: 'Recreation_Area'
}
