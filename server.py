import json
import os
import warnings
from typing import TypedDict

from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    BatchFileInput,
    FileInput,
    FileResponse,
    FileType,
    InputSchema,
    InputType,
    NewFileInputType,
    ResponseBody,
    TaskSchema,
)

from helper import GeoLocatorModel

# Suppress warnings
warnings.filterwarnings("ignore")

# Define typed dictionaries for inputs and parameters
class ImageInputs(TypedDict):
    image_input: BatchFileInput
    output_path: FileInput

class ImageParameters(TypedDict):
    pass

# Initialize the ML server and model
server = MLServer(__name__)
# Add application metadata
server.add_app_metadata(
    name="GeoLocator-ONNX",
    author="Manasa K",
    version="0.1.0",
    info=load_file_as_string("./README.md"),
)
model = GeoLocatorModel("GeoLocator.onnx")

def initialize_task_schema() -> TaskSchema:
    """Initialize task schema with image input and output path"""
    inputs = [
        InputSchema(
            key="image_input", 
            label="Upload images", 
            input_type=InputType.BATCHFILE
        ),
        InputSchema(
            key="output_path",
            label="Output JSON Path",
            input_type=NewFileInputType(
                default_name="output.json",
                default_extension=".json",
                allowed_extensions=[".json"]
            ),
        ),
    ]
    return TaskSchema(inputs=inputs, parameters=[])

def clean_output_path(output_path: str):
    """Ensure the output path is clean before writing"""
    if os.path.exists(output_path):
        os.remove(output_path)

def save_results(results: list, output_path: str):
    """Save processed results to a JSON file"""
    with open(output_path, "w") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

def process_single_image(image_path: str) -> dict:
    """Process a single image and return the result"""
    return model.predict(image_path)

@server.route(
    "/process_images", task_schema_func=initialize_task_schema, short_title="Result"
)
def process_images(inputs: ImageInputs, parameters: ImageParameters) -> ResponseBody:
    """Handle image processing and save results to a JSON"""
    output_path = inputs["output_path"].path
    input_images = inputs["image_input"].files

    clean_output_path(output_path)

    results = [process_single_image(image_file.path) for image_file in input_images]

    save_results(results, output_path)

    print(f"Results saved at: {output_path}")

    return ResponseBody(
        root=FileResponse(
            file_type=FileType.JSON,
            path=output_path
        )
    )

# Start the server
if __name__ == "__main__":
    server.run()
