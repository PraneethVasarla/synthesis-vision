import subprocess
import os
import yaml
import tensorflow as tf

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
        raise "Docker compose failed to start the container"

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