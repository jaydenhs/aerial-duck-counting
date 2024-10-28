import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from dataset import EiderDuckDataset  
from models.unet import UNet
from utils import class_to_idx

# Load the model
model = UNet(in_channels=3, num_classes=2, base_channels=128)
model.load_state_dict(torch.load('unet_epoch_50_loss_0.000156.pth', weights_only=True))
model.eval()

# Initialize the dataset and dataloader
dataset = train_dataset = EiderDuckDataset('processed-data')
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Verify training by visualizing a sample image, its ground truth, and the predicted density map
sample_image, sample_density_map, _ = next(iter(train_loader))
with torch.no_grad():
    predicted_density_map = model(sample_image)

# Plot the image
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(sample_image[0].permute(1, 2, 0).cpu().numpy())
plt.title('Image')

# Determine the common colorbar limits
vmin = min(sample_density_map.min().item(), predicted_density_map.min().item())
vmax = max(sample_density_map.max().item(), predicted_density_map.max().item())

for i, label in enumerate(class_to_idx.keys()):
    plt.subplot(2, 3, i + 2)
    ground_truth_density = sample_density_map[0][i].cpu().numpy()
    plt.imshow(ground_truth_density, cmap='hot', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(f'Ground Truth Density Map: {label}\nCount: {ground_truth_density.sum()}')

# Plot the segmentation maps and calculate the area underneath
for i, label in enumerate(class_to_idx.keys()):
    plt.subplot(2, 3, i + 5)
    predicted_density = predicted_density_map[0][i].cpu().numpy()
    plt.imshow(predicted_density, cmap='hot', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(f'Predicted Density Map: {label}\nCount: {predicted_density.sum()}')

plt.show()