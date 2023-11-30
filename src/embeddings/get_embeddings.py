from sentence_transformers import SentenceTransformer,util
from PIL import Image
import os,sys

sys.path.append(os.getcwd())
model = SentenceTransformer('clip-ViT-B-32')

IMAGES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'images')

def get_embeddings(model,image):
    embedding = model.encode(Image.open(image))
    return embedding

