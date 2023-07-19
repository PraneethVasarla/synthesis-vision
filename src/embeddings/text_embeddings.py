import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from tqdm.auto import tqdm

def get_individual_embeddings(sentences,model,tokenizer):
    # Tokenize the input
    encoded_inputs = tokenizer.batch_encode_plus(sentences, padding=True,truncation=True,return_tensors='tf')

    # Get the model's output
    outputs = model(**encoded_inputs)

    # Get the final hidden state (last layer) from the output
    last_hidden_state = outputs.last_hidden_state
    sentence_embeddings = tf.reduce_mean(last_hidden_state, axis=1)

    return sentence_embeddings

def fuse_with_self_attention(embeddings):
    # Convert the list of embeddings to a tensor
    embeddings_tensor = tf.concat(embeddings, axis=0)

    # Compute the similarity matrix using dot product
    similarity_matrix = tf.matmul(embeddings_tensor, embeddings_tensor, transpose_b=True)

    # Apply softmax to obtain attention weights
    attention_weights = tf.nn.softmax(similarity_matrix, axis=1)

    # Perform weighted sum of the embeddings using attention weights
    fused_embedding = tf.matmul(attention_weights, embeddings_tensor)

    # Reduce the fused embedding to a single tensor
    fused_embedding = tf.reduce_mean(fused_embedding, axis=0, keepdims=True)

    return fused_embedding

def get_text_embedding(sentences, model, tokenizer):
    batch_fused_embeddings = []
    # Create a tqdm progress bar for texts
    with tqdm(total=len(sentences), desc="Text embeddings", dynamic_ncols=True) as progress_bar:
        for batch in sentences:
            individual_embeddings = get_individual_embeddings(batch, model, tokenizer)
            fused_embedding = fuse_with_self_attention(individual_embeddings)
            batch_fused_embeddings.append(fused_embedding)
            del individual_embeddings
            progress_bar.update(1)

    batch_fused_embeddings = np.array(batch_fused_embeddings)
    return batch_fused_embeddings



# Example usage

# from src.utils.utilities import load_model
# model_name = 'distilbert-base-uncased'
# models_dir = 'models'
#
# model,tokenizer = load_model(model_name,models_dir)

# captions = [[
#         "Two young guys with shaggy hair look at their hands while hanging out in the yard.",
#         "Two young, White males are outside near many bushes.",
#         "Two men in green shirts are standing in a yard.",
#         "A man in a blue shirt standing in a garden.",
#         "Two friends enjoy time spent together."
#     ],
#     [
#         "Several men in hard hats are operating a giant pulley system.",
#         "Workers look down from up above on a piece of equipment.",
#         "Two men working on a machine wearing hard hats.",
#         "Four men on top of a tall structure.",
#         "Three men on a large rig."
#     ],
# [
#         "A child in a pink dress is climbing up a set of stairs in an entry way.",
#         "A little girl in a pink dress going into a wooden cabin.",
#         "A little girl climbing the stairs to her playhouse.",
#         "A little girl climbing into a wooden playhouse.",
#         "A girl going into a wooden building."
#     ]
# ]
# embeddings = get_text_embedding(captions,model=model,tokenizer=tokenizer)
# print("Shape of embedding:", embeddings.shape)
# print(embeddings)
