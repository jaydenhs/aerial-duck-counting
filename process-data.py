import os
import torch
import numpy as np
from PIL import Image
import json
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import h5py

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

# Create directories if they don't exist
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(density_output_dir, exist_ok=True)
os.makedirs(points_output_dir, exist_ok=True)

# Parameters
min_annotations = 200  # Minimum annotations threshold
sigma = 5  # For Gaussian blur

# Load annotations
with open(annotations_path, 'r') as f:
    annotations = json.load(f)

# Define a color map for different keypoint labels
class_to_idx = {
    'Adult Male': 0,
    'Adult Female': 1,
    'Juvenile': 2
}

# Gaussian kernel to generate density maps
def generate_density_map(height, width, points, sigma=10):
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
    image = image.resize((1024, 1024))  # Resize all images to 1024x1024

    # Create density maps (one per class)
    density_maps = np.zeros((3, 1024, 1024), dtype=np.float32)
    points = {'Adult Male': [], 'Adult Female': [], 'Juvenile': []}

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
        density_maps[class_idx] = generate_density_map(1024, 1024, point_list, sigma=sigma)

    # Save the processed data
    image.save(os.path.join(image_output_dir, f"{image_file}.png"))
    with h5py.File(os.path.join(density_output_dir, f"{image_file}.h5"), 'w') as hf:
        hf.create_dataset('density_maps', data=density_maps)
    np.save(os.path.join(points_output_dir, f"{image_file}.npy"), points)
