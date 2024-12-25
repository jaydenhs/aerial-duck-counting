import numpy as np
from numpy.random import randint, normal
from skimage.draw import ellipse
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import os
import shutil
import json
from tqdm import tqdm

def add_ducks_circular(image, num_ducks, male_color, female_color, center, radius, min_distance=5):
    positions = []
    bounding_boxes = []
    for _ in range(num_ducks):
        while True:
            r = np.sqrt(np.random.uniform(0, 1)) * radius
            theta = np.random.uniform(0, 2 * np.pi)
            center_x = int(center[0] + r * np.cos(theta))
            center_y = int(center[1] + r * np.sin(theta))

            if 0 <= center_x < image.shape[1] and 0 <= center_y < image.shape[0]:
                if all(np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2) >= min_distance for x, y in positions):
                    positions.append((center_x, center_y))
                    break

        color = male_color if len(positions) % 2 == 0 else female_color
        bounding_boxes.extend(draw_duck(image, center_x, center_y, color))
    
    return positions, bounding_boxes

def add_ducks_clustered(image, num_clusters, ducks_per_cluster, cluster_std_dev, male_color, female_color, min_distance=5):
    positions = []
    bounding_boxes = []
    height, width = image.shape[:2]

    cluster_centers = [(randint(100, width - 100), randint(100, height - 100)) for _ in range(num_clusters)]
    for center in cluster_centers:
        cluster_positions = []
        for _ in range(ducks_per_cluster):
            while True:
                x = int(normal(center[0], cluster_std_dev))
                y = int(normal(center[1], cluster_std_dev))

                if 0 <= x < width and 0 <= y < height:
                    if all(np.sqrt((x - px) ** 2 + (y - py) ** 2) >= min_distance for px, py in cluster_positions):
                        cluster_positions.append((x, y))
                        break

        for i, (center_x, center_y) in enumerate(cluster_positions):
            color = male_color if i % 2 == 0 else female_color
            bounding_boxes.extend(draw_duck(image, center_x, center_y, color))
    
    return positions, bounding_boxes

def draw_duck(image, x, y, color):
    radius_x = 5
    radius_y = randint(4, 8)
    angle = randint(0, 180)
    rr, cc = ellipse(y, x, radius_x, radius_y, rotation=np.deg2rad(angle), shape=image.shape)

    mask = np.zeros(image.shape[:2], dtype=np.float32)
    mask[rr, cc] = 1
    mask = gaussian(mask, sigma=(radius_x / 2, radius_y / 2)) * 1.1

    for i in range(3):
        image[rr, cc, i] = np.clip(image[rr, cc, i] + mask[rr, cc] * color[i], 0, 255)

    min_r, max_r = rr.min(), rr.max()
    min_c, max_c = cc.min(), cc.max()
    return [(min_c, min_r, max_c, max_r)]

def synthesize_dataset(output_dir, n_images=2, mode="circular"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "male_duck"},
            {"id": 2, "name": "female_duck"}
        ]
    }
    annotation_id = 1

    for image_id in tqdm(range(n_images), desc=f"Generating {n_images} images"):
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        image[:, :] = [26, 32, 37]  # Background color

        male_color = [242, 242, 242]  # Light gray
        female_color = [169, 169, 169]  # Dark gray

        bounding_boxes = []

        if mode == "circular":
            num_ducks = randint(800, 1000)
            _, bounding_boxes = add_ducks_circular(image, num_ducks, male_color, female_color, center=(512, 512), radius=400)
        elif mode == "clustered":
            num_clusters = randint(2, 4)
            ducks_per_cluster = randint(400, 600)
            cluster_std_dev = 50
            _, bounding_boxes = add_ducks_clustered(image, num_clusters, ducks_per_cluster, cluster_std_dev, male_color, female_color)
        elif mode == "mixed":
            # Combine circular and clustered distributions
            num_ducks_circular = randint(400, 500)
            _, circular_bboxes = add_ducks_circular(image, num_ducks_circular, male_color, female_color, center=(512, 512), radius=300)
            num_clusters = randint(2, 3)
            ducks_per_cluster = randint(200, 300)
            cluster_std_dev = 40
            _, clustered_bboxes = add_ducks_clustered(image, num_clusters, ducks_per_cluster, cluster_std_dev, male_color, female_color)
            bounding_boxes = circular_bboxes + clustered_bboxes

        output_path = os.path.join(output_dir, f'synth_{image_id}.png')
        plt.imsave(output_path, image)

        coco_annotations["images"].append({
            "id": image_id,
            "file_name": f'synth_{image_id}.png',
            "height": 1024,
            "width": 1024
        })

        for i, bbox in enumerate(bounding_boxes):
            coco_annotations["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1 if i % 2 == 0 else 2,  # Alternate between male and female
                "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                "iscrowd": 0
            })
            annotation_id += 1

    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump(coco_annotations, f, indent=4, default=int)

if __name__ == "__main__":
    mode = input("Enter mode (circular/clustered/mixed): ").strip().lower()
    synthesize_dataset('synthesized_combined/circular', n_images=100, mode='circular')
    synthesize_dataset('synthesized_combined/clustered', n_images=100, mode='clustered')
    synthesize_dataset('synthesized_combined/mixed', n_images=100, mode='mixed')