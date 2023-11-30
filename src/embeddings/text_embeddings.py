import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from tqdm.auto import tqdm

def reshape_embedding(embedding):
    truncated_embeddings = embedding[:, :1] #taking first element from 2nd dimension, keeping 1st n 3rd dims unchanged
    concatenated_embeddings = tf.concat(truncated_embeddings, axis=1)
    return concatenated_embeddings

def get_individual_embeddings(sentences,model,tokenizer):
    # Tokenize the input
    encoded_inputs = tokenizer.batch_encode_plus(sentences, padding=True,truncation=True,return_tensors='tf')
    # Get the model's output
    outputs = model(**encoded_inputs)

    # Get the final hidden state (last layer) from the output
    last_hidden_state = outputs.last_hidden_state
    sentence_embeddings = reshape_embedding(last_hidden_state)
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
    fused_embedding = tf.reduce_mean(fused_embedding,axis=0)
    return fused_embedding

def get_text_embedding(sentences, model, tokenizer):
    batch_fused_embeddings = []
    # Create a tqdm progress bar for texts
    with tqdm(total=len(sentences), desc="Text embeddings", dynamic_ncols=True) as progress_bar:
        for batch in range(len(sentences)):
            individual_embeddings = get_individual_embeddings(sentences[batch], model, tokenizer)
            fused_embedding = fuse_with_self_attention(individual_embeddings)
            batch_fused_embeddings.append(fused_embedding)
            del individual_embeddings
            progress_bar.update(1)

    batch_fused_embeddings = np.array(batch_fused_embeddings)
    return batch_fused_embeddings

# Example usage
if __name__ == "__main__":
    from src.utils.utilities import load_text_model
    model_name = 'distilbert-base-uncased'
    models_dir = 'models'

    model,tokenizer = load_text_model(model_name,models_dir)

    captions = [[
        "two young guy shaggy hair look hand hanging yard",
        "two young white male outside near many bush",
        "two men green shirt standing yard",
        "man blue shirt standing garden",
        "two friend enjoy time spent together"
      ],
        [
            "several men hard hat operating giant pulley system",
            "worker look piece equipment",
            "two men working machine wearing hard hat",
            "four men top tall structure",
            "three men large rig"
        ],
        [
        "child pink dress climbing set stair entry way",
        "little girl pink dress going wooden cabin",
        "little girl climbing stair playhouse",
        "little girl climbing wooden playhouse",
        "girl going wooden building"
      ],
        [
            "two young men young lady walk field near water",
            "three people walking beautiful meadow towards ocean",
            "two men woman walking field",
            "two male one female walking on path",
            "three people walking on path meadow"
        ]
    ]

    embeddings = get_text_embedding(captions,model,tokenizer)
    print(embeddings.shape)