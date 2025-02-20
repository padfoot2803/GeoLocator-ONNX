from torchvision import transforms as trn
from PIL import Image
from torch.autograd import Variable as V
import os
import onnxruntime as ort
import numpy as np


class GeoLocatorProcessor:
    def __init__(self):
        self.valid_extension = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

    def returnTF(self):
        # Load the image transformer
        tf = trn.Compose(
            [
                trn.Resize((224, 224)),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return tf

    def load_labels(self):
        # Prepare all the labels
        # Scene category relevant
        file_name_category = "categories_places365.txt"
        classes = list()
        with open(file_name_category) as class_file:
            for line in class_file:
                classes.append(line.strip().split(" ")[0][3:])
        classes = tuple(classes)

        # Indoor and outdoor relevant
        file_name_IO = "IO_places365.txt"
        with open(file_name_IO) as f:
            lines = f.readlines()
            labels_IO = []
            for line in lines:
                items = line.rstrip().split()
                labels_IO.append(int(items[-1]) - 1)  # 0 is indoor, 1 is outdoor
        labels_IO = np.array(labels_IO)

        # Scene attribute relevant
        # file_name_attribute = "labels_sunattribute.txt"
        # with open(file_name_attribute) as f:
        #     lines = f.readlines()
        #     labels_attribute = [item.rstrip() for item in lines]
        # file_name_W = "W_sceneattribute_wideresnet18.npy"
        # W_attribute = np.load(file_name_W)

        return (
            classes,
            labels_IO,
        )  # labels_attribute, W_attribute

    def run_IO_Detector(self, img_file):
        # Load the transformer
        tf = self.returnTF()  # Image transformer

        img = Image.open(img_file)
        if img.mode != "RGB":
            img = img.convert("RGB")
        input_img = V(tf(img).unsqueeze(0))
        input_numpy = input_img.numpy()
        # print(input_numpy.shape())
        return input_numpy

    def postProcessing(self, output_probs, image_path, top_k=5, io_threshold=0.5):
        # Sort category indices by descending probability
        sorted_indices = np.argsort(output_probs)[::-1]

        # Compute Indoor/Outdoor classification using top 10 categories
        classes, labels_IO = self.load_labels()
        io_score = np.average(
            labels_IO[sorted_indices[:10]], weights=output_probs[sorted_indices[:10]]
        )
        if io_score < io_threshold:
            environment_type = "Indoor"
        else:
            environment_type = "Outdoor"

        # Get top-k scene categories
        scene_categories = [
            {"Description": classes[idx], "Confidence": f"{output_probs[idx]:.3f}"}
            for idx in sorted_indices[:top_k]
            if output_probs[idx] > 0.01
        ]

        # Construct and return result dictionary
        return {
            "Image": image_path,
            "Environment Type": environment_type,
            "Scene Category": scene_categories,
        }


class GeoLocatorModel:
    def __init__(self, model_path):
        self.glp = GeoLocatorProcessor()
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def predict(self, image_path):
        input = self.glp.run_IO_Detector(image_path)
        output = self.session.run(None, {"input": input})
        probs = self._softmax(output[0][0])
        result = self.glp.postProcessing(probs, image_path)
        return result

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
