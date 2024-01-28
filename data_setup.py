from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def image_dataloaders(train_dir: str, test_dir: str, transform: transforms, batch_size: int, NUM_WORKERS=os.cpu_count()):
    
    # Creating datasets
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=transform)
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=transform)
    
    # Get names for classes (blue, red, yellow)
    class_names = train_data.classes

    # Turning train and test datasets into Dataloaders
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size,
                                  num_workers=NUM_WORKERS, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size,
                                 num_workers=NUM_WORKERS, shuffle=False)
    
    return train_dataloader, test_dataloader, class_names
