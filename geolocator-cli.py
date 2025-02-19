import onnxruntime as ort
import numpy as np
from helper import GeoLocatorModel
import argparse
from pathlib import Path
import json
from pprint import pprint

# Arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--input_image", type=str, required=True)
parser.add_argument("--output_directory", type=str, required=True)
args = parser.parse_args()
input_image = Path(args.input_image)
output_directory = Path(args.output_directory)
output_directory.mkdir(parents=True, exist_ok=True)

model = GeoLocatorModel("GeoLocator.onnx")
outputs = model.predict(str(input_image))
pprint(outputs)
with open(output_directory / "output.json", "w") as f:
    json.dump(outputs, f)