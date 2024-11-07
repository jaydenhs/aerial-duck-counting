import os
import torch
import numpy as np
from PIL import Image
import json
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import shutil
from utils import class_to_idx

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Paths
image_dir = "labelstudio-export/images"
annotations_path = "labelstudio-export/annotations.json"
output_dir = "processed-data"
image_output_dir = os.path.join(output_dir, "images")
density_output_dir = os.path.join(output_dir, "gt-density")
points_output_dir = os.path.join(output_dir, "gt-points")

# Delete processed-data directory if it exists and create directories
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(density_output_dir, exist_ok=True)
os.makedirs(points_output_dir, exist_ok=True)

# Parameters
min_annotations = 200  # Minimum annotations threshold
sigma = 3  # For Gaussian blur, based on size of ducks

# Load annotations
with open(annotations_path, 'r') as f:
    annotations = json.load(f)

# Gaussian kernel to generate density maps
def generate_density_map(height, width, points, sigma):
    density_map = np.zeros((height, width), dtype=np.float32)
    
    for point in points:
        x = int(point['x'] * width / 100)
        y = int(point['y'] * height / 100)
        if 0 <= x < width and 0 <= y < height:
            density_map[y, x] += 1
            
    return gaussian_filter(density_map, sigma=sigma)

# Filter annotations
filtered_annotations = []
for annotation in tqdm(annotations, desc="Filtering annotations"):
    valid_annotations = [ann for ann in annotation['annotations'] if not ann['was_cancelled']]
    if valid_annotations and len(valid_annotations[0]['result']) >= min_annotations:
        filtered_annotations.append(annotation)

# Process each annotation
for idx, annotation in enumerate(tqdm(filtered_annotations, desc="Processing annotations")):
    # Image
    image_file = annotation['file_upload']
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size  # Get original dimensions

    # Create density maps (one per class)
    density_maps = np.zeros((2, height, width), dtype=np.float32)
    points = {'Adult Male': [], 'Adult Female': []}

    # Extract annotations
    for result in annotation['annotations'][0]['result']:
        if result['type'] == 'keypointlabels':
            label = result['value']['keypointlabels'][0]
            if label in class_to_idx:
                points[label].append({'x': result['value']['x'], 'y': result['value']['y']})

    # Generate density maps for each class
    for label, point_list in points.items():
        # Check if the label exists in the mapping
        class_idx = class_to_idx[label]
        density_maps[class_idx] = generate_density_map(height, width, point_list, sigma=sigma)

    # Patch the data into 512x512 patches and save
    patch_size = 512
    num_patches_x = width // patch_size
    num_patches_y = height // patch_size

    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Extract points patch first
            points_patch = {label: [] for label in points}
            for label, point_list in points.items():
                for point in point_list:
                    x = int(point['x'] * width / 100)
                    y = int(point['y'] * height / 100)
                    if i * patch_size <= y < (i + 1) * patch_size and j * patch_size <= x < (j + 1) * patch_size:
                        points_patch[label].append({'x': (x - j * patch_size) * 100 / patch_size, 'y': (y - i * patch_size) * 100 / patch_size})

            # Check if there are annotations over the minimum threshold
            total_points = sum(len(points_patch[label]) for label in points_patch)
            if total_points >= min_annotations:
                save_name = f"{image_file}_{i}_{j}"

                if save_name not in ["09dfba85-IMG_4660.JPG_2_3", "d8d096d5-overhead_2.png_2_2"]:
                    continue
                
                # Extract image patch
                image_patch = image.crop((j * patch_size, i * patch_size, (j + 1) * patch_size, (i + 1) * patch_size))
                image_patch.save(os.path.join(image_output_dir, f"{save_name}.png"))

                # Extract density map patches
                density_patch = density_maps[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                np.save(os.path.join(density_output_dir, f"{save_name}.npy"), density_patch)

                # Save points patch
                np.save(os.path.join(points_output_dir, f"{save_name}.npy"), points_patch)
