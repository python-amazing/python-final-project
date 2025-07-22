"""
Widgets for running inference using module functions.
"""

import ipywidgets as widgets
from IPython.display import clear_output
from satellite_classifier.inference import run_inference

def create_inference_widget():
    # File upload widget
    file_upload = widgets.FileUpload(
        accept='.jpg,.png,.tif,.tiff',
        multiple=False,
        description='Upload Image'
    )

    # Button to trigger inference
    run_button = widgets.Button(
        description='Run Inference',
        button_style='success'
    )

    # Output area
    output = widgets.Output()

    def on_run_clicked(b):
        with output:
            clear_output()
            if not file_upload.value:
                print("Please upload an image file.")
                return
            uploaded_file = next(iter(file_upload.value.values()))
            image_bytes = uploaded_file['content']
            # Run inference (assuming run_inference accepts bytes)
            result = run_inference(image_bytes)
            print("Inference Result:", result)

    run_button.on_click(on_run_clicked)

    # Return the widget container
    return widgets.VBox([file_upload, run_button, output])