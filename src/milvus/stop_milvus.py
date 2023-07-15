import subprocess
import os
import argparse
import shutil

container_name = "milvus"

parser = argparse.ArgumentParser(description='Take arguments to run the setup_milvus.py file')
parser.add_argument('--volumes',action="store_true",help='Delete volumes along with removing the milvus container')

args = parser.parse_args()

command = f"docker-compose --project-name {container_name} down"
execute_path = os.path.join("src","milvus")
result = subprocess.run(command, shell=True, capture_output=True, cwd=execute_path)
if result.returncode == 0:
    print("Docker container stopped successfully!")
else:
    error_message = result.stderr.decode("utf-8").strip() if result.stderr else "Failed to stop Docker compose"

if args.volumes:
    response = input("WARNING: THIS ACTION WOULD DELETE ALL THE COLLECTIONS AND THEIR CONTENTS WITHIN MILVUS. DO YOU WISH TO CONTINUE? Y/N:")
    if response.lower() == "y":
        script_directory = os.path.dirname(os.path.abspath(__file__))

        # Set the current working directory
        os.chdir(script_directory)

        # Remove the directory
        directory_path = "volumes"
        shutil.rmtree(directory_path)
        print("Successfully deleted all Milvus collections and their contents!")
    else:
        print("Skipping Milvus collection deletion.")