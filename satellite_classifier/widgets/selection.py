# torchgeo_model_selector.py
"""
Widgets for selecting and loading models for different tasks.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Dict, List, Any, Optional

from satellite_classifier.constants import TASK_MODELS
from satellite_classifier.models import get_available_models, load_pretrained_models
import time

class TorchGeoModelSelector:
    """Interactive widget for selecting and loading TorchGeo models based on tasks."""
    
    def __init__(self):
        self.selected_task = None
        self.selected_sensor = None
        self.selected_model_base = None # e.g., ResNet18, ViT_Small_Patch16
        self.selected_weights = None
        
        self.loaded_models = {}
        
        # Create widgets
        self.create_widgets()
        self.setup_interactions()
        
        # Initial update of options
        self._update_sensor_options()
        self._update_model_base_options()
        self._update_weights_options()
        self.update_model_info() # Initial info display
    
    def create_widgets(self):
        """Create all UI widgets."""
        # Task selection dropdown
        self.task_dropdown = widgets.Dropdown(
            options=list(TASK_MODELS.keys()),
            value=list(TASK_MODELS.keys())[0] if TASK_MODELS else None,
            description='Task:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        # Task description and datasets (updated dynamically)
        self.task_description = widgets.HTML(value="")
        self.update_task_description()

        # Sensor selection dropdown
        self.sensor_dropdown = widgets.Dropdown(
            options=[],
            description='Sensor:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )

        # Model base selection (e.g., ResNet18, ViT_Small_Patch16)
        self.model_base_dropdown = widgets.Dropdown(
            options=[],
            description='Model Architecture:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )

        # Weights selection dropdown
        self.weights_dropdown = widgets.Dropdown(
            options=[],
            description='Pre-trained Weights:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        # Buttons
        self.load_button = widgets.Button(
            description='Load Selected Model', # Changed to singular
            button_style='primary',
            tooltip='Click to load the selected model',
            icon='download',
            layout=widgets.Layout(width='180px')
        )
        
        self.available_button = widgets.Button(
            description='Show All Available Models (TorchGeo)',
            button_style='info',
            tooltip='Show all available TorchGeo models from list_models()',
            icon='list',
            layout=widgets.Layout(width='auto')
        )
        
        self.clear_button = widgets.Button(
            description='Clear Output',
            button_style='warning',
            tooltip='Clear the output area',
            icon='eraser',
            layout=widgets.Layout(width='120px')
        )
        
        # Output area
        self.output = widgets.Output(layout=widgets.Layout(height='300px', overflow='auto'))
        
        # Model info display
        self.model_info = widgets.HTML(value="")
        
        # Progress bar
        self.progress = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='Loading:',
            bar_style='',
            style={'bar_color': 'blue'},
            layout=widgets.Layout(visibility='hidden')
        )
    
    def setup_interactions(self):
        """Set up widget event handlers."""
        self.task_dropdown.observe(self.on_task_change, names='value')
        self.sensor_dropdown.observe(self.on_sensor_change, names='value')
        self.model_base_dropdown.observe(self.on_model_base_change, names='value')
        self.weights_dropdown.observe(self.on_weights_change, names='value')
        
        self.load_button.on_click(self.load_model) # Changed to singular
        self.available_button.on_click(self.show_available_models)
        self.clear_button.on_click(self.clear_output)
    
    def update_task_description(self):
        """Update the task description HTML based on the selected task."""
        task_info = TASK_MODELS.get(self.task_dropdown.value)
        if task_info:
            self.task_description.value = (
                f"<b>Description:</b> {task_info['description']}<br>"
                f"<b>Common Datasets:</b> {', '.join(task_info['datasets'])}"
            )
        else:
            self.task_description.value = ""

    def _update_sensor_options(self):
        """Update sensor dropdown options based on selected task."""
        current_task = self.task_dropdown.value
        if current_task and current_task in TASK_MODELS:
            sensors_info = TASK_MODELS[current_task].get('sensors', {})
            sensor_options = list(sensors_info.keys())
            self.sensor_dropdown.options = sensor_options
            if sensor_options:
                self.sensor_dropdown.value = sensor_options[0]
            else:
                self.sensor_dropdown.value = None
        else:
            self.sensor_dropdown.options = []
            self.sensor_dropdown.value = None

    def _update_model_base_options(self):
        """Update model base dropdown options based on selected task and sensor."""
        current_task = self.task_dropdown.value
        current_sensor = self.sensor_dropdown.value
        
        model_base_options = []
        if current_task and current_sensor:
            sensor_info = TASK_MODELS[current_task].get('sensors', {}).get(current_sensor, {})
            model_base_options = list(sensor_info.get('models', {}).keys())
            
        self.model_base_dropdown.options = model_base_options
        if model_base_options:
            self.model_base_dropdown.value = model_base_options[0]
        else:
            self.model_base_dropdown.value = None

    def _update_weights_options(self):
        """Update weights dropdown options based on selected task, sensor, and model base."""
        current_task = self.task_dropdown.value
        current_sensor = self.sensor_dropdown.value
        current_model_base = self.model_base_dropdown.value
        
        weights_options = []
        if current_task and current_sensor and current_model_base:
            sensor_info = TASK_MODELS[current_task].get('sensors', {}).get(current_sensor, {})
            weights_options = sensor_info.get('models', {}).get(current_model_base, [])
            
        self.weights_dropdown.options = weights_options
        if weights_options:
            self.weights_dropdown.value = weights_options[0]
        else:
            self.weights_dropdown.value = None

    def on_task_change(self, change):
        """Handle task dropdown change."""
        self.update_task_description()
        self._update_sensor_options()
        self._update_model_base_options() # This will be triggered by sensor change
        self._update_weights_options()    # This will be triggered by model base change
        self.update_model_info()

    def on_sensor_change(self, change):
        """Handle sensor dropdown change."""
        self._update_model_base_options()
        self._update_weights_options() # This will be triggered by model base change
        self.update_model_info()
    
    def on_model_base_change(self, change):
        """Handle model base dropdown change."""
        self._update_weights_options()
        self.update_model_info()

    def on_weights_change(self, change):
        """Handle weights dropdown change."""
        self.update_model_info()

    def _get_full_model_name(self) -> Optional[str]:
        """Construct the full model name string for TorchGeo's get_model."""
        base = self.model_base_dropdown.value
        weights = self.weights_dropdown.value

        if not base or not weights:
            return None

        # Logic to combine base and weights into a TorchGeo model name string
        # This is where your naming convention comes into play.
        # Examples: 'resnet18_imagenet', 'resnet50_sentinel2_all_moco'
        
        # A simple approach: if weights is not 'random' or 'imagenet', append it.
        # This needs to match TorchGeo's internal naming
        full_name = base.lower() # e.g., resnet18, vit_small_patch16
        
        # Special handling for standard torchvision weights often not prefixed
        if weights == "imagenet":
            full_name = f"{full_name}_imagenet"
        elif weights == "random":
            # 'random' implies no pretrained weights, get_model handles this
            pass
        else:
            # For TorchGeo-specific weights, they are often part of the model name
            # e.g., resnet18_ssl4eo_s12_rgb_moco
            full_name = f"{full_name}_{weights}"
            
        # Standardize ViT naming
        if "ViT_Small_Patch16" in base:
            full_name = full_name.replace("vit_small_patch16", "vit_small_patch16_224")
        elif "Swin_V2_B" in base:
            full_name = full_name.replace("swin_v2_b", "swin_v2_b_256") # Example, check actual TorchGeo names

        return full_name
        
    def load_model(self, button): # Changed to singular
        """Load selected model using the module function."""
        full_model_name = self._get_full_model_name()
        if not full_model_name:
            with self.output:
                print("❌ No complete model and weights selection made.")
            return
        
        # Show progress bar
        self.progress.layout.visibility = 'visible'
        self.progress.value = 0
        
        with self.output:
            clear_output(wait=True)
            print(f"🎯 Task: {self.task_dropdown.value}")
            print(f"🤖 Loading model: {self.model_base_dropdown.value} with weights: {self.weights_dropdown.value}")
            print("=" * 50)
            
            # Create args object for the module function
            class Args:
                def __init__(self, models):
                    self.models = models
            
            args = Args([full_model_name]) # Pass a list with the single full model name
            
            try:
                # Update progress
                self.progress.value = 20
                
                # Call the actual module function
                self.loaded_models = load_pretrained_models(args)
                
                # Update progress
                self.progress.value = 100
                
                print("\n✅ Model Loading Complete!")
                print(f"📊 Successfully loaded: {len(self.loaded_models)} models")
                
                if self.loaded_models:
                    print("\n📋 Loaded Model Summary:")
                    for name, model in self.loaded_models.items():
                        num_params = sum(p.numel() for p in model.parameters())
                        print(f"   • {name}: {type(model).__name__} ({num_params:,} parameters)")
                
                # Update model info
                self.update_model_info()
                
            except ImportError as e:
                print(f"❌ Import Error: {e}")
                print("💡 Make sure TorchGeo is installed: pip install torchgeo")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
            finally:
                # Hide progress bar after a delay
                time.sleep(1)
                self.progress.layout.visibility = 'hidden'
    
    def show_available_models(self, button):
        """Show all available TorchGeo models using list_models()."""
        with self.output:
            clear_output(wait=True)
            print("🔍 All Available TorchGeo Models (from list_models())")
            print("=" * 50)
            
            try:
                available_models = get_available_models()
                
                if available_models:
                    # Group models by type
                    model_types = {}
                    for model in available_models:
                        model_type = model.split('_')[0]  # Get base architecture
                        if model_type not in model_types:
                            model_types[model_type] = []
                        model_types[model_type].append(model)
                    
                    for i, (model_type, models) in enumerate(model_types.items(), 1):
                        print(f"\n{i}. {model_type.upper()} Models:")
                        for model in sorted(models): # Sort for better readability
                            print(f"   • {model}")
                    
                    print(f"\n📊 Total: {len(available_models)} models available via list_models()")
                else:
                    print("❌ No models found or TorchGeo not installed")
                
            except Exception as e:
                print(f"❌ Error getting available models: {e}")
    
    def clear_output(self, button):
        """Clear the output area."""
        with self.output:
            clear_output(wait=True)
    
    def update_model_info(self):
        """Update model information display."""
        task_info = TASK_MODELS.get(self.task_dropdown.value)
        sensor_info_html = ""
        model_info_html = ""
        weights_info_html = ""

        if self.task_dropdown.value and task_info:
            task_description = task_info.get('description', 'N/A')
            common_datasets = ', '.join(task_info.get('datasets', []))

            if self.sensor_dropdown.value:
                sensor_data = task_info.get('sensors', {}).get(self.sensor_dropdown.value, {})
                sensor_description = sensor_data.get('description', 'N/A')
                sensor_info_html = f"<p><b>Sensor:</b> {self.sensor_dropdown.value}<br><i>{sensor_description}</i></p>"

                if self.model_base_dropdown.value:
                    model_base_name = self.model_base_dropdown.value
                    model_info_html = f"<p><b>Model Architecture:</b> <code>{model_base_name}</code></p>"
                    
                    if self.weights_dropdown.value:
                        weights_name = self.weights_dropdown.value
                        weights_info_html = f"<p><b>Pre-trained Weights:</b> <code>{weights_name}</code></p>"
                        
        
        info_html = f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff;'>
            <h4>🎯 Selected Configuration</h4>
            <p><b>Task:</b> {self.task_dropdown.value}</p>
            <p><b>Description:</b> {task_description}</p>
            <p><b>Relevant Datasets:</b> {common_datasets}</p>
            {sensor_info_html}
            {model_info_html}
            {weights_info_html}
            <p><b>Full Model Name for Loading:</b> <code>{self._get_full_model_name() or 'N/A'}</code></p>
        </div>
        """
        
        self.model_info.value = info_html
    
    def display(self):
        """Display the complete widget interface."""
        # Main title
        title = widgets.HTML(
            "<div style='text-align: center; padding: 20px;'>"
            "<h2>🛰️ TorchGeo Model Selector</h2>"
            "<p>Interactive tool for selecting and loading pre-trained geospatial models</p>"
            "</div>"
        )
        
        # Create sections
        task_section = widgets.VBox([
            widgets.HTML("<h3>1️⃣ Select Task</h3>"),
            self.task_dropdown,
            self.task_description
        ], layout=widgets.Layout(margin='10px 0'))
        
        selection_section = widgets.VBox([
            widgets.HTML("<h3>2️⃣ Select Model Details</h3>"),
            self.sensor_dropdown,
            self.model_base_dropdown,
            self.weights_dropdown,
            widgets.HBox([self.load_button, self.available_button, self.clear_button]),
            self.progress
        ], layout=widgets.Layout(margin='10px 0'))
        
        info_section = widgets.VBox([
            widgets.HTML("<h3>3️⃣ Configuration Summary</h3>"),
            self.model_info
        ], layout=widgets.Layout(margin='10px 0'))
        
        output_section = widgets.VBox([
            widgets.HTML("<h3>4️⃣ Output</h3>"),
            self.output
        ], layout=widgets.Layout(margin='10px 0'))
        
        # Main container
        main_container = widgets.VBox([
            title,
            task_section,
            selection_section,
            info_section,
            output_section
        ], layout=widgets.Layout(padding='20px'))
        
        display(main_container)
    
    def get_loaded_models(self) -> Dict:
        """Return the loaded models dictionary."""
        return self.loaded_models

def quick_load_models(task_name: str, model_base: str, weights: str) -> Dict:
    """
    Quick function to load a single model for a specific task with specified sensor and weights.
    
    Args:
        task_name: Name of the task (must be in TASK_MODELS keys)
        model_base: Base model architecture (e.g., "ResNet18")
        weights: Specific pre-trained weights (e.g., "imagenet", "ssl4eo_s2_all_moco")
    
    Returns:
        Dictionary of loaded models
    """
    if task_name not in TASK_MODELS:
        available_tasks = list(TASK_MODELS.keys())
        raise ValueError(f"Task '{task_name}' not found. Available tasks: {available_tasks}")
    
    # Construct the full model name using the same logic as in the widget
    full_model_name = model_base.lower()
    if "ViT_Small_Patch16" in model_base:
        full_model_name = full_model_name.replace("vit_small_patch16", "vit_small_patch16_224")
    elif "Swin_V2_B" in model_base:
        full_model_name = full_model_name.replace("swin_v2_b", "swin_v2_b_256")

    if weights == "imagenet":
        full_model_name = f"{full_model_name}_imagenet"
    elif weights == "random":
        pass # 'random' implies no pretrained weights, get_model handles this
    else:
        full_model_name = f"{full_model_name}_{weights}"

    class Args:
        def __init__(self, models):
            self.models = models
    
    args = Args([full_model_name])
    
    print(f"🎯 Loading model for task: {task_name}")
    print(f"🤖 Model: {model_base}, Weights: {weights}")
    print("=" * 40)
    
    return load_pretrained_models(args)

def create_model_selector():
    """
    Factory function to create and display a TorchGeoModelSelector widget.
    
    Returns:
        TorchGeoModelSelector instance
    """
    selector = TorchGeoModelSelector()
    selector.display()
    return selector