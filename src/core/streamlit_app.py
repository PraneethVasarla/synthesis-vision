import streamlit as st
import os

import sys
from pymilvus import connections,Collection
from sentence_transformers import SentenceTransformer,util
from PIL import Image

sys.path.append(os.getcwd())

from src.utils.utilities import load_text_model,get_input_text_embedding,load_images_as_html,normalize_vector

IMAGES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'images')

connections.connect("default", host="localhost", port="19530")
collection = Collection("images")      # Get an existing collection.
print(collection.num_entities)
collection.load()

model = SentenceTransformer('clip-ViT-B-32')

search_params = {
    "nprobe": 16,
    "metric": "L2",
    "efSearch": 250,
    "efConstruction": 250,
}


def search_with_text(text_input,model,limit):
    # Image.open(image_path)
    embedding = model.encode(text_input,show_progress_bar=False)
    normalized_embedding = normalize_vector(embedding)

    results = collection.search(
	data=[normalized_embedding],
	anns_field="embedding",
	param=search_params,
	limit=limit,
	expr=None,
	# set the names of the fields you want to retrieve from the search result.
	output_fields=['product_id'],
	consistency_level="Strong")

    image_results = results[0].ids
    distances = results[0].distances

    image_results = [os.path.join(IMAGES_PATH, image_name) for image_name in image_results]
    print("results found")

    return image_results

def search_with_image(image_input,model,limit):
    # Image.open(image_path)
    embedding = model.encode(Image.open(image_input),show_progress_bar=False)
    normalized_embedding = normalize_vector(embedding)

    results = collection.search(
	data=[normalized_embedding],
	anns_field="embedding",
	param=search_params,
	limit=limit,
	expr=None,
	# set the names of the fields you want to retrieve from the search result.
	output_fields=['product_id'],
	consistency_level="Strong")

    image_results = results[0].ids
    distances = results[0].distances

    image_results = [os.path.join(IMAGES_PATH, image_name) for image_name in image_results]
    print("results found")

    return image_results

st.header("Welcome to Synthesis Vision!")
st.caption("Our multimodal image retrieval system")

col1, col2 = st.columns(2)
# Create a text input widget
text_input = col1.text_input("Enter a text input:")
uploaded_file = col1.file_uploader("Upload an image:")

limit_slider = col2.slider("Number of results: ",1,50,5)

# Run the function in the background

image_paths = []

if text_input:
   image_paths = search_with_text(text_input,model,limit_slider)
   st.session_state.uploaded_file = None
elif uploaded_file:
   st.session_state.text_input = ""
   image_paths = search_with_image(uploaded_file,model,limit_slider)

# Show the images
for image_path in image_paths:
  try:
      st.image(image_path)
  except:
     st.text("Search for results")
