import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
    
    # Create model save path
    if model_name.endswith(".pth") or model_name.endswith(".pt"):
        model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    torch.save(obj=model.state_dict(),
               f=model_save_path)
    