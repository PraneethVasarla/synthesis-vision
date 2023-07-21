import sys
import os
import json
sys.path.append(os.getcwd())

from src.utils.utilities import load_text_model,load_vision_model
from src.core.joint_embedding import JointEmbedding
from pymilvus import connections,Collection

connections.connect("default", host="localhost", port="19530")
collection = Collection("synthesis_vision")   # Get an existing collection.


text_model_name = 'distilbert-base-uncased'
vision_model_name = 'google/vit-base-patch16-224-in21k'

text_model,tokenizer = load_text_model(text_model_name)
vision_model,feature_extractor = load_vision_model(vision_model_name)

joint_embedding = JointEmbedding(text_model,tokenizer,vision_model,feature_extractor)

file_path = os.path.join(os.path.dirname(__file__), 'data', 'descriptions.json')
with open(file_path,"r") as file:
    captions_dict = json.load(file)

images = list(captions_dict.keys())
captions = list(captions_dict.values())

joint_embedding.get_pair_embeddings(images=images[:100],captions=captions[:100])
fused_embeddings = joint_embedding.fuse_embeddings_with_attention()



print(f"Text embedding shape: {joint_embedding.text_embeddings.shape}")
# print(f"Image embedding shape: {joint_embedding.image_embeddings.shape}")
print(f"Fused embedding shape: {fused_embeddings[0][0].shape}")


embeddings_to_insert = [fused_embeddings[i][0] for i in range(len(fused_embeddings))]
# embeddings_to_insert = [joint_embedding.text_embeddings[i][0] for i in range(len(joint_embedding.text_embeddings))]
image_names_to_insert = images[:100]

data = [
    image_names_to_insert,
    embeddings_to_insert
]

print(image_names_to_insert)
print(captions[:100])
mr = collection.insert(data)
print(mr)
collection.flush()