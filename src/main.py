import sys
import os
import json
sys.path.append(os.getcwd())

from src.utils.utilities import load_model
from core.joint_embedding import JointEmbedding

model_name = 'distilbert-base-uncased'
models_dir = 'models'

model,tokenizer = load_model(model_name,models_dir)

joint_embedding = JointEmbedding(model,tokenizer)

file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'descriptions.json')
with open(file_path,"r") as file:
    captions_dict = json.load(file)

images = list(captions_dict.keys())
captions = list(captions_dict.values())

joint_embedding.get_pair_embeddings(images=None,captions=captions[:100])

print(joint_embedding.text_embeddings.shape)