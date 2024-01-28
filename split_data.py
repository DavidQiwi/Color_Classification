import splitfolders
from pathlib import Path
import shutil

# \parts contains 3 directories named blue, red, yellow 
input_folder = Path("parts")
output_folder = Path("dataset")

# Check if the directory exists, if not create one 
if output_folder.is_dir():
    print(f"{output_folder} is a directory.")
else:
    print(f"{output_folder} does not exist. Creating this directory.")
    output_folder.mkdir(parents=True, exist_ok=True)

# Split the folders into the training and testing sets using 'splitfolders' package
splitfolders.ratio(input_folder, output=output_folder,
                   seed=42, ratio=(.8, .0 , .2),
                   group_prefix=None)        # 80% training set, 0% validation set, 20% testing set

# Delete val directory, which contains empty directories
val_folder = output_folder / "val"
if val_folder.is_dir():
    shutil.rmtree(val_folder)
