import sys
import os
from pymilvus import connections,Collection

sys.path.append(os.getcwd())

from src.utils.utilities import load_text_model,get_input_text_embedding,load_images_as_html

IMAGES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'images')

connections.connect("default", host="localhost", port="19530")
collection = Collection("images")      # Get an existing collection.
print(collection.num_entities)
collection.load()

text_model_name = 'distilbert-base-uncased'

text_model,tokenizer = load_text_model(text_model_name)

text = "dog running on streets"

embedding = get_input_text_embedding(text,text_model,tokenizer)
embedding = list(embedding)

search_params = {
    "nprobe": 16,
    "metric": "L2",
    "efSearch": 250,
    "efConstruction": 250,
}

results = collection.search(
	data=[embedding],
	anns_field="embedding",
	param=search_params,
	limit=20,
	expr=None,
	# set the names of the fields you want to retrieve from the search result.
	output_fields=['product_id'],
	consistency_level="Strong"
)

image_results = results[0].ids
distances = results[0].distances

image_results = [os.path.join(IMAGES_PATH, image_name) for image_name in image_results]


load_images_as_html(image_results,distances,text)