import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt

# Define a color map for different keypoint labels
class_to_idx = {
    'Adult Male': 0,
    'Adult Female': 1,
}

# Dataset
class EiderDuckDataset(Dataset):
    def __init__(self, processed_data_dir, transform=transforms.ToTensor()):
        self.image_files = [f for f in os.listdir(os.path.join(processed_data_dir, 'images')) if f.endswith('.png')]
        self.processed_data_dir = processed_data_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        base_filename = os.path.splitext(image_file)[0]
        
        density_map_path = os.path.join(self.processed_data_dir, 'gt-density', f"{base_filename}.npy")
        
        # Load density map
        density_maps = np.load(density_map_path)
        
        # Load and transform image
        image_path = os.path.join(self.processed_data_dir, 'images', image_file)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        density_maps = torch.from_numpy(density_maps)
        segmentation_maps = (density_maps > 1e-3).float()

        return image, density_maps, segmentation_maps
    
def plot_sample(sample):
    image, density_maps, segmentation_maps = sample
    
    # Plot the image
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title('Image')

    # Plot the density maps
    for i, label in enumerate(class_to_idx.keys()):
        plt.subplot(2, 3, i + 2)
        plt.imshow(density_maps[i], cmap='hot')
        plt.title(f'Density Map: {label}')

    # Plot the segmentation maps
    for i, label in enumerate(class_to_idx.keys()):
        plt.subplot(2, 3, i + 5)
        plt.imshow(segmentation_maps[i], cmap='gray')
        plt.title(f'Segmentation Map: {label}')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    processed_data_dir = 'processed-data'
    dataset = EiderDuckDataset(processed_data_dir)
    random_idx = random.randint(0, len(dataset) - 1)
    plot_sample(dataset[random_idx])
    

