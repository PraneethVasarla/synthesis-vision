import os
import torch
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer


def get_subfolders(directory):
    subfolders = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            subfolders.append(os.path.join(root, dir))
    return subfolders

models_directory = "../models"
subfolders = get_subfolders(models_directory)

existing_models = [os.path.basename(path) for path in subfolders]

model_name = 'bert-base-uncased'

if model_name not in existing_models:
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Save the model and tokenizer to a local directory
    model.save_pretrained(f'../models/{model_name}')
    tokenizer.save_pretrained(f'../models/{model_name}')

else:
    print("Model already existing. Loading from disk...")
    model = BertModel.from_pretrained(f'../models/{model_name}')
    tokenizer = BertTokenizer.from_pretrained(f'../models/{model_name}')


# Example input sentence
text = "a man with yellow hat sitting with two people"

# Tokenize the input
tokens = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')

# Get the model's output
outputs = model(**tokens)

# Get the final hidden state (last layer) from the output
last_hidden_state = outputs.last_hidden_state
sentence_embeddings = torch.mean(last_hidden_state, dim=1)
# Print the shape of the last hidden state
print("Shape of last hidden state:", sentence_embeddings.shape)
print(sentence_embeddings.detach().numpy())