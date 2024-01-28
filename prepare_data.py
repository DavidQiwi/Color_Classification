import os
import requests
import zipfile
from pathlib import Path

# Path to the data
data_path = Path("data/")
image_path = data_path / Path("dataset/")

# Check if the directory exists, if not create one 
if image_path.is_dir():
    print(f"{image_path} is a directory.")
else:
    print(f"{image_path} does not exist. Creating this directory.")
    image_path.mkdir(parents=True, exist_ok=True)                   # Preventing error if exists

# Download images 

with open(image_path / "blue", "wb") as f:
    request = requests.get("https://git.rwth-aachen.de/justin.pratt/ki-demonstrator/-/tree/main/images/blue")
    print("Downloading file")
    f.write(request.content)
