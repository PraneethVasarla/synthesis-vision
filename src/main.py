import sys
import os
import json
sys.path.append(os.getcwd())

from src.utils.utilities import load_text_model
from core.joint_embedding import JointEmbedding

text_model_name = 'distilbert-base-uncased'
vision_model_name = 'google/vit-base-patch16-224-in21k'

text_model,text_tokenizer = load_text_model(text_model_name)

joint_embedding = JointEmbedding(text_model,text_tokenizer)

file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'descriptions.json')
with open(file_path,"r") as file:
    captions_dict = json.load(file)

images = list(captions_dict.keys())
captions = list(captions_dict.values())

joint_embedding.get_pair_embeddings(images=None,captions=captions[:50])

print(joint_embedding.text_embeddings.shape)