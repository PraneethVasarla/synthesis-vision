import sys
import os
import json
import numpy as np
sys.path.append(os.getcwd())
from tqdm.auto import tqdm

from src.utils.utilities import load_text_model,load_vision_model,normalize_batch_vectors
from src.core.joint_embedding import JointEmbedding
from pymilvus import connections,Collection
from sentence_transformers import SentenceTransformer,util

connections.connect("default", host="localhost", port="19530")
collection = Collection("images")   # Get an existing collection.


text_model_name = 'distilbert-base-uncased'
vision_model_name = 'google/vit-base-patch16-224-in21k'

text_model,tokenizer = load_text_model(text_model_name)
vision_model,feature_extractor = load_vision_model(vision_model_name)

model = SentenceTransformer('clip-ViT-B-32')

# joint_embedding = JointEmbedding(text_model,tokenizer,vision_model,feature_extractor)

joint_embedding = JointEmbedding(text_model,tokenizer,vision_model,feature_extractor,embedding_model = model)

file_path = os.path.join(os.path.dirname(__file__), 'data', 'descriptions_cleaned.json')
with open(file_path,"r") as file:
    captions_dict = json.load(file)

images = list(captions_dict.keys())
captions = list(captions_dict.values())


batch_embeddings = joint_embedding.get_embeddings(images=images,texts=captions)
print(f"len: {len(batch_embeddings)}")

if len(batch_embeddings.shape) == 3:
    batch_count = 0
    with tqdm(total=len(batch_embeddings), desc="Inserting into Milvus", dynamic_ncols=True) as progress_bar:
        for batch in batch_embeddings:
            batch_size = len(batch)
            print(f"batch size: {batch_size}")
            batch_count += 1
            batch_start = batch_count*batch_size - batch_size
            batch_end = batch_count*batch_size
            print("normalizing vectors")
            normalized_embeddings = normalize_batch_vectors(batch)
            image_names_to_insert = images[batch_start:batch_end]
            print(len(normalized_embeddings),len(image_names_to_insert))

            data = [
                image_names_to_insert,
                normalized_embeddings
            ]

            mr = collection.insert(data)
            print(mr)
            collection.flush()
            progress_bar.update(1)

else:
    normalized_embeddings = normalize_batch_vectors(batch_embeddings)
    image_names_to_insert = images[:len(normalized_embeddings)]

    data = [
        image_names_to_insert,
        normalized_embeddings
    ]

    mr = collection.insert(data)
    print(mr)
    collection.flush()