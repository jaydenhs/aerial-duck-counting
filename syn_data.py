import numpy as np
from numpy.random import randint, choice
from skimage.draw import ellipse
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import os
import shutil
import json
from tqdm import tqdm

def add_ducks(image, num_ducks, color, min_distance=0):
    positions = []
    bounding_boxes = []
    for _ in range(num_ducks):
        while True:
            center_x = randint(0, 1024)
            center_y = randint(0, 1024)
            if all(np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2) >= min_distance for x, y in positions):
                positions.append((center_x, center_y))
                break
        
        radius_x = 4
        radius_y = choice([3, 6])
        angle = randint(0, 180)
        rr, cc = ellipse(center_y, center_x, radius_x, radius_y, rotation=np.deg2rad(angle), shape=image.shape)
        
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        mask[rr, cc] = 1
        
        mask = gaussian(mask, sigma=(radius_x / 2, radius_y / 2)) * 1.1
        
        for i in range(3):
            image[rr, cc, i] = np.clip(image[rr, cc, i] + mask[rr, cc] * color[i], 0, 255)
        
        min_r, max_r = rr.min(), rr.max()
        min_c, max_c = cc.min(), cc.max()
        bounding_boxes.append((min_c, min_r, max_c, max_r))
            
    return positions, bounding_boxes

def synthesize_dataset(output_dir, n_images=100):
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
        n_males = randint(125, 175)
        n_females = randint(125, 175)

        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        image[:, :] = [26, 32, 37]
        _, male_bboxes = add_ducks(image, n_males, [242, 242, 242])
        _, female_bboxes = add_ducks(image, n_females, [169, 169, 169])

        output_path = os.path.join(output_dir, f'synth_{image_id}_M{n_males}_F{n_females}.png')
        plt.imsave(output_path, image)

        coco_annotations["images"].append({
            "id": image_id,
            "file_name": f'synth_{image_id}_M{n_males}_F{n_females}.png',
            "height": 1024,
            "width": 1024
        })

        for bbox in male_bboxes:
            coco_annotations["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                "iscrowd": 0
            })
            annotation_id += 1

        for bbox in female_bboxes:
            coco_annotations["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 2,
                "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                "iscrowd": 0
            })
            annotation_id += 1

    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump(coco_annotations, f, indent=4, default=int)


synthesize_dataset('synthesized_combined/sparse')