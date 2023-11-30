import tensorflow as tf
from PIL import Image
from tqdm.auto import tqdm

import sys
import os
import json
sys.path.append(os.getcwd())

from src.utils.utilities import load_vision_model

# Load the ViT model and feature extractor
model_name = 'google/vit-base-patch16-224-in21k'
model,feature_extractor = load_vision_model(model_name)

IMAGES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'images')

def process_image(image_path, feature_extractor, model):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors='pt')
    features = model(**inputs)
    embeddings = features.last_hidden_state
    embeddings_detached = embeddings.detach()
    embeddings_tf = tf.convert_to_tensor(embeddings_detached)
    return embeddings_tf

def get_image_embeddings(images_names, model, feature_extractor):
    batch_embeddings = []
    # Create a tqdm progress bar for images
    with tqdm(total=len(images_names), desc="Image embeddings", dynamic_ncols=True) as progress_bar:
        for image_name in images_names:
            image_path = os.path.join(IMAGES_PATH, image_name)
            embeddings_tf = process_image(image_path, feature_extractor, model)
            batch_embeddings.append(embeddings_tf)
            progress_bar.update(1)


    batch_embeddings = tf.concat(batch_embeddings, axis=0)
    print(f"shape after concat: {batch_embeddings.shape}")
    return batch_embeddings


if __name__ == "__main__":
    # images_names = ['1000092795.jpg', '10002456.jpg', '1000268201.jpg', '1000344755.jpg', '1000366164.jpg', '1000523639.jpg', '1000919630.jpg', '10010052.jpg', '1001465944.jpg', '1001545525.jpg', '1001573224.jpg', '1001633352.jpg', '1001773457.jpg', '1001896054.jpg', '100197432.jpg', '100207720.jpg', '1002674143.jpg', '1003163366.jpg', '1003420127.jpg', '1003428081.jpg', '100444898.jpg', '1005216151.jpg', '100577935.jpg', '1006452823.jpg', '100652400.jpg', '1007129816.jpg', '100716317.jpg', '1007205537.jpg', '1007320043.jpg', '100759042.jpg', '10082347.jpg', '10082348.jpg', '100845130.jpg', '10090841.jpg', '1009434119.jpg', '1009692167.jpg', '101001624.jpg', '1010031975.jpg', '1010087179.jpg', '1010087623.jpg', '10101477.jpg', '1010470346.jpg', '1010673430.jpg', '101093029.jpg', '101093045.jpg', '1011572216.jpg', '1012150929.jpg', '1012212859.jpg', '1012328893.jpg', '101262930.jpg']
    images_names = ['1000092795.jpg', '10002456.jpg', '1000268201.jpg']
    embeddings = get_image_embeddings(images_names,model,feature_extractor)
    print(embeddings.shape)