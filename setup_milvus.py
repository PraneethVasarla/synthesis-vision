import subprocess
import argparse
import os

from utils import check_container_running,check_container_exists,up_docker_compose,start_container,download_file,add_attu_block

parser = argparse.ArgumentParser(description='Take arguments to run the setup_milvus.py file')
parser.add_argument('--collection_name',help='Name of the milvus collection/database')

args = parser.parse_args()

container_name = 'milvus'

if args.collection_name:
    collection_name = args.collection_name
else:
    collection_name = "synthesis_vision"

if not check_container_exists(container_name):
    compose_path = os.path.join("milvus","docker-compose.yml")
    compose_url = "https://github.com/milvus-io/milvus/releases/download/v2.2.11/milvus-standalone-docker-compose.yml"
    download_file(compose_url,compose_path)
    add_attu_block()
    up_docker_compose(container_name)

elif not check_container_running(container_name):
    start_container(container_name)
    print("Docker container up and running")

else:
    print(f"A container with the name '{container_name}' already exists and is running. Skipping container creation.")

# Step 3: Wait for Milvus to start
print("Waiting for Milvus to start...")
subprocess.run(["sleep", "30"])

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

connections.connect("default", host="localhost", port="19530")

has = utility.has_collection(collection_name)
print(f"Does collection {collection_name} exist in Milvus: {has}")


# Define the collection schema
fields = [
    FieldSchema(name="image_name", dtype=DataType.VARCHAR,is_primary=True,max_length=200),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    # Add more fields as needed
]
schema = CollectionSchema(fields=fields, description="synthesis-vision vector database")
collection = Collection(collection_name, schema, consistency_level="Strong")

print(f"{collection_name} collection created!")
print("Attu server running...")
print("Attu interface available at http://localhost:8000")