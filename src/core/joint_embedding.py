import sys
import os

sys.path.append(os.getcwd())

from src.embeddings.text_embeddings import get_text_embedding
class JointEmbedding:
    def __init__(self,model,tokenizer):
        self.image_embeddings = None
        self.text_embeddings = None
        self.model = model
        self.tokenizer = tokenizer

    def get_pair_embeddings(self,images,captions):
        self.text_embeddings = get_text_embedding(captions,model=self.model,tokenizer=self.tokenizer)
        pass
