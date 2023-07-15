import os
import sys
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer

sys.path.append(os.getcwd())

from src.utils.utilities import get_subfolders,check_tf_gpu

project_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_directory = os.path.join(project_directory, "models")
os.makedirs(models_directory, exist_ok=True)

subfolders = get_subfolders(models_directory)
existing_models = [os.path.basename(path) for path in subfolders]

model_name = 'bert-large-uncased'

model_directory = os.path.join(models_directory, model_name)
os.makedirs(model_directory, exist_ok=True)

gpu = check_tf_gpu()
print(f"Using GPU: {gpu}")

if model_name not in existing_models:
    model = TFAutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save the model and tokenizer to the model-specific directory
    model.save_pretrained(model_directory)
    tokenizer.save_pretrained(model_directory)

else:
    print("Model already exists. Loading from disk...")
    model = TFAutoModel.from_pretrained(model_directory)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)


# Example input sentence
text = "a man with yellow hat sitting with two people and a dog walking on the road while another helicopter is landing in background"

# Tokenize the input
tokens = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='tf')

# Get the model's output
outputs = model(tokens)

# Get the final hidden state (last layer) from the output
last_hidden_state = outputs.last_hidden_state
sentence_embeddings = tf.reduce_mean(last_hidden_state, axis=1)
# Print the shape of the last hidden state
print("Shape of last hidden state:", sentence_embeddings.shape)
print(sentence_embeddings)