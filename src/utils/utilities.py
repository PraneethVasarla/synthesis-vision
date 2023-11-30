import subprocess
import os
import yaml
import tensorflow as tf
import numpy as np
from transformers import TFAutoModel,AutoTokenizer, ViTModel, ViTImageProcessor

# Load the existing YAML file
def add_attu_block():
    compose_path = os.path.join("src","milvus", "docker-compose.yml")
    with open(compose_path, "r") as file:
        docker_compose_data = yaml.safe_load(file)

    # Define the new 'attu' service block
    attu_service = {
        "container_name": "attu",
        "image": "zilliz/attu:v2.2.6",
        "environment": {
            "MILVUS_URL": "milvus-standalone:19530"
        },
        "ports": [
            "8000:3000"
        ],
        "depends_on": [
            "standalone"
        ]
    }

    # Add the 'attu' service block to the services
    docker_compose_data["services"]["attu"] = attu_service

    # Write the updated YAML data back to the file
    with open(compose_path, "w") as file:
        yaml.dump(docker_compose_data, file)
    print("added attu block to docker-compose")

def check_container_exists(container_name):
    command = f"docker ps -aqf name={container_name}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return bool(result.stdout.strip())  # Return True if the container exists, False otherwise

def check_container_running(container_name):
    command = f"docker ps -qf name={container_name}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout.strip()
    if output:
        return True
    else:
        return False

def up_docker_compose(container_name):
    command = f"docker-compose --project-name {container_name} up -d"
    execute_path = os.path.join("src","milvus")
    result = subprocess.run(command, shell=True, capture_output=True, cwd=execute_path)
    if result.returncode == 0:
        print("Docker container up and running!")
        return True
    else:
        error_message = result.stderr.decode("utf-8").strip() if result.stderr else "Docker compose failed to run the container"
        raise Exception(error_message)

def start_container(container_name):
    command = f"docker-compose start {container_name}"
    result = subprocess.run(command, shell=True, capture_output=True)
    if result.returncode == 0:
        return True
    else:
        raise Exception("Docker compose failed to start the container")

def download_file(url, destination_path):
    subprocess.run(["wget", url, "-O", destination_path])

def get_subfolders(directory):
    subfolders = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            subfolders.append(os.path.join(root, dir))
    return subfolders

def check_tf_gpu():
    if tf.config.list_physical_devices('GPU'):
        # Use CUDA for GPU acceleration
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        return True
    else:
        # Run on CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return False

def load_text_model(model_name,models_directory='models',use_cache=True):
    project_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_directory = os.path.join(project_directory, models_directory)
    os.makedirs(models_directory, exist_ok=True)

    subfolders = get_subfolders(models_directory)
    existing_models = [os.path.basename(path) for path in subfolders]


    gpu = check_tf_gpu()
    print(f"Using GPU: {gpu}")

    if "models--"+model_name in existing_models:
        print("Model already exists. Loading from disk...")

    model = TFAutoModel.from_pretrained(model_name,cache_dir=models_directory if use_cache else None)
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=models_directory if use_cache else None)

    return model,tokenizer

def load_vision_model(model_name,models_directory='models',use_cache=True):
    project_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_directory = os.path.join(project_directory, models_directory)
    os.makedirs(models_directory, exist_ok=True)

    subfolders = get_subfolders(models_directory)
    existing_models = [os.path.basename(path) for path in subfolders]

    # gpu = check_tf_gpu()
    # print(f"Using GPU: {gpu}")

    if "models--"+model_name in existing_models:
        print("Model already exists. Loading from disk...")
    model = ViTModel.from_pretrained(model_name,cache_dir=models_directory if use_cache else None)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name,cache_dir=models_directory if use_cache else None)

    return model,feature_extractor

def get_input_text_embedding(text,model,tokenizer):
    encoded_input = tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors='tf')
    outputs = model(**encoded_input)

    last_hidden_state = outputs.last_hidden_state
    sentence_embeddings = tf.reduce_mean(last_hidden_state, axis=1)
    sentence_embeddings = sentence_embeddings.numpy()

    return sentence_embeddings[0]

def load_images_as_html(image_paths,distances,text_input):
    """Loads a list of image paths as a normal HTML file in the current directory and opens it in a web browser. The images are aligned linearly in a table with the first column as serial numbers and table lines. The table will fit 60% of the screen and the images will be centered in the image column.

    Args:
        image_paths (list): A list of image paths.
    """

    html = ""
    html += "<table border='1' style='width: 60%'>"
    html += f"<h3>Search results for: <b>{text_input}</b></h3>"
    html += "<tr><th>S.No.</th><th style='text-align: center'>Image</th><th>Distance</th></tr>"
    for i, image in enumerate(zip(image_paths,distances)):
        html += f'<tr><td>{i + 1}</td><td align="center" padding="15px"><img src="{image[0]}"/></td><td>{image[1]}</td></tr>'
    html += "</table>"

    with open("images.html", "w") as f:
        f.write(html)

def normalize_batch_vectors(vectors_list):
    normalized_vectors = []
    for vector in vectors_list:
        magnitude = np.linalg.norm(vector)
        if magnitude == 0:
            normalized_vectors.append(vector)
        else:
            normalized_vector = vector / magnitude
            normalized_vectors.append(normalized_vector)
    return normalized_vectors

def normalize_vector(vector):
    magnitude = np.linalg.norm(vector)  # Calculate the magnitude of the vector
    if magnitude == 0:  # Avoid division by zero
        return vector
    normalized_vector = vector / magnitude  # Normalize the vector
    return normalized_vector