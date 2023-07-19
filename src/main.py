import sys
import os
import json
sys.path.append(os.getcwd())

from src.utils.utilities import load_text_model,load_vision_model
from core.joint_embedding import JointEmbedding

text_model_name = 'distilbert-base-uncased'
vision_model_name = 'google/vit-base-patch16-224-in21k'

text_model,tokenizer = load_text_model(text_model_name)
vision_model,feature_extractor = load_vision_model(vision_model_name)

joint_embedding = JointEmbedding(text_model,tokenizer,vision_model,feature_extractor)

file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'descriptions.json')
with open(file_path,"r") as file:
    captions_dict = json.load(file)

images = list(captions_dict.keys())
captions = list(captions_dict.values())

joint_embedding.get_pair_embeddings(images=images[:100],captions=captions[:100])
fused_embeddings = joint_embedding.fuse_embeddings_with_attention()

print(f"Text embedding shape: {joint_embedding.text_embeddings.shape}")
print(f"Image embedding shape: {joint_embedding.image_embeddings.shape}")

print(f"Fused embedding shape: {fused_embeddings.shape}")