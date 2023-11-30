import subprocess
import argparse
import os
import sys

from pymilvus.client.types import MetricType

sys.path.append(os.getcwd())

from src.utils import check_container_running,check_container_exists,up_docker_compose,start_container,download_file,add_attu_block

parser = argparse.ArgumentParser(description='Take arguments to run the setup_milvus.py file')
parser.add_argument('--database_name',help='Name of the milvus collection/database')

args = parser.parse_args()

container_name = 'milvus'

if not args.database_name:
    raise Exception("Database name mandatory. Run from command line and use arguments to pass database name")

database_name = args.database_name

if not check_container_exists(container_name):
    compose_path = os.path.join("src", "milvus", "docker-compose.yml")
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
    FieldSchema, CollectionSchema, DataType,
    Collection, IndexType,db
)

connections.connect(host="localhost", port="19530")

databases = db.list_database()

has = database_name in databases
if has:
    print(f"Database {database_name} already exists.")
    pass
else:
    database = db.create_database(database_name)

db.using_database(database_name)
print(f"Does database {database_name} exist in Milvus: {has}")


# Define the collection schema
image_fields = [
    FieldSchema(name="product_id", dtype=DataType.VARCHAR,is_primary=True,max_length=200),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    # Add more fields as needed
]

image_index_params = {
    "index_type": IndexType.HNSW,
    "metric_type": MetricType.IP,  # Use MetricType.IP for inner product similarity
    "params": {
        "M": 16,  # The number of bidirectional links
        "efConstruction": 500  # The size of the dynamic candidate list during index building
    }
}

text_fields = [
    FieldSchema(name="product_id", dtype=DataType.VARCHAR,is_primary=True,max_length=200),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    # Add more fields as needed
]

text_index_params = {
    "index_type": IndexType.HNSW,
    "metric_type": MetricType.IP,  # Use MetricType.IP for inner product similarity
    "params": {
        "M": 16,  # The number of bidirectional links
        "efConstruction": 500  # The size of the dynamic candidate list during index building
    }
}

image_collection_info = {"fields":image_fields,"index":image_index_params}
text_collection_info = {"fields":text_fields,"index":text_index_params}

# Image collection
image_schema = CollectionSchema(fields=image_collection_info['fields'], description="Image collection")
image_collection = Collection("images", image_schema, consistency_level="Strong")
image_collection.create_index(field_name="embedding", index_params=image_collection_info['index'])
print("Image collection created")

#Text collection
text_schema = CollectionSchema(fields=text_collection_info['fields'], description="Text collection")
text_collection = Collection("texts", text_schema, consistency_level="Strong")
text_collection.create_index(field_name="embedding", index_params=text_collection_info['index'])
print("Text collection created")

print(f"{database_name} database created!")
print("Attu server running...")
print("Attu interface available at http://localhost:8000")