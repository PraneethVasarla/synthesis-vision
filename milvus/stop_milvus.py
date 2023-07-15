import subprocess
import os


container_name = "milvus"

command = f"docker-compose --project-name {container_name} down"
execute_path = os.path.join("")
result = subprocess.run(command, shell=True, capture_output=True, cwd=execute_path)
if result.returncode == 0:
    print("Docker container stopped successfully!")
else:
    error_message = result.stderr.decode("utf-8").strip() if result.stderr else "Failed to stop Docker compose"