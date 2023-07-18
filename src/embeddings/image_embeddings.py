import tensorflow as tf
from PIL import Image


import sys
import os
import json
sys.path.append(os.getcwd())

from src.utils.utilities import load_vision_model

# Load the ViT model and feature extractor
model_name = 'google/vit-base-patch16-224-in21k'
model,feature_extractor = load_vision_model(model_name)

# Load and preprocess the input image
image_path = '36979.jpg'
image = Image.open(image_path)

# Process the image
inputs = feature_extractor(images=image, return_tensors='pt')

# Extract features from the image
features = model(**inputs)
embeddings = features.last_hidden_state

# Print the shape of the generated embeddings
print(embeddings.shape)
