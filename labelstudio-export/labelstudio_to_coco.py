import json
import os
import re
import argparse
import random
from icecream import ic
from PIL import Image
'''
USAGE

python tools/active_learning/task_to_coco_with_no_split.py /home/m32patel/scratch/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/Uncompleted_tasks/dfo_uncompleted_task_iter5.json /home/m32patel/scratch/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/Uncompleted_tasks/all_data.json --image_filename_prefix='' --image_width=8256 --image_height=5504 --bbox_width=40 --bbox_height=40

python tools/active_learning/task_to_coco_with_no_split.py /home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/Uncompleted_tasks/VIP_uncompleted_task_iter7.json /home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/Uncompleted_tasks/VIP_uncompleted_task_iter7_coco.json --image_filename_prefix='' --image_width=8256 --image_height=5504 --bbox_width=40 --bbox_height=40

python labelstudio-export/labelstudio_to_coco.py /Users/aqeelpatel/Downloads/aerial-duck-counting/labelstudio-export/annotations.json /Users/aqeelpatel/Downloads/aerial-duck-counting/labelstudio-export/coco_annotations.json  --image_filename_prefix='' --image_width=8256 --image_height=5504 --bbox_width=40 --bbox_height=40

'''

def main(input_json_file, output_json_file, image_filename_prefix,image_foldername , image_width, image_height, bbox_width, bbox_height):
    with open(input_json_file, 'r') as file:
        tasks = json.load(file)
    # ic(image_foldername)
    images = []
    annotations = []
    categories = [    {
      "id": 0,
      "name": "Adult Female"
    },
    {
      "id": 1,
      "name": "Adult Male"
    },
    {
      "id": 2,
      "name": "Ice"
    },
    {
      "id": 3,
      "name": "Juvenile"
    },
    {
      "id": 4,
      "name": "Unknown"
    }]  # Adjust as per your dataset
    category_name_to_id = {'Adult Female': 0, 'Adult Male': 1, 'Ice': 2, 'Juvenile': 3, 'Unknown': 4}

    annotation_id = 0

    replacement_mappings = {
        "Plane1": "HACS2023DSLR_Plane1_yellow",
        "5C20230811_cam1": "5C20230811_cam1_25mm_nofoolography",
        "5C20230812_cam1": "5C20230812_cam1_25mm_nofoolography",
        "5C20230812_cam2": "5C20230812_cam2_25mm_foolography",
        "5C20230813_cam2": "5C20230813_cam2_25mm_foolography",
        "5C20230813_cam1": "5C20230813_cam1_25mm_nofoolography",
        "5C20230815_cam1": "5C20230815_cam1_25mm",
        "5C20230818_cam1": "5C20230818_cam1_25mm",
        "5C20230820_cam1": "5C20230820_cam1_25mm",
        "5C20230823_cam1": "5C20230823_cam1_25mm",
        "5C20230824_cam1": "5C20230824_cam1_25mm",
        "5C20230825_cam1": "5C20230825_cam1_25mm",
        "5C20230828_cam1": "5C20230828_cam1_25mm",
        "5C20230829_cam1": "5C20230829_cam1_25mm",
    }
    def filename_to_bbox_height_width(filename):
        
        filename_to_bbox = {
        "06de88f9-IMG_4610.JPG": (20,20,-4),
        "086ed3c4-cluster_2.jpg": (15,15,0),
        "08763981-oblique_1.png": (20,20,-4),
        "09dfba85-IMG_4660.JPG": (15,15,-4),
        "3348606c-IMG_8483.jpg": (20,20,-4),
        "3c4f336c-IMG_4638.jpg": (20,20,-4),
        "44ad2b66-oblique_1.png": (20,20,-4),
        "48d8aa80-IMG_4108.jpg": (20,20,-4),
        "4956fc0c-IMG_4405.JPG": (20,20,-4),
        "55c2d4ae-IMG_8499.jpg": (20,20,-4),
        "70df6d9b-cluster_1.jpg": (15,15,-4),
        "8569b068-IMG_4621.JPG": (20,20,-4),
        "a5af68aa-cluster_3.jpg": (20,20,-4),
        "a5ef6dc5-overhead_1.png": (10,10,0),
        "aed014f4-IMG_4641.JPG": (20,20,-4),
        "b61226a1-IMG_4605.JPG": (20,20,-4),
        "b67048da-IMG_4406.JPG": (20,20,-4),
        "c09605b7-IMG_4648.JPG": (10,10,0),
        "c84aa17f-IMG_8612.JPG": (20,20,-4),
        "c8517700-IMG_4232.JPG": (20,20,-4),
        "d088d9a7-IMG_4591.JPG": (20,20,-4),
        "d75d83e1-oblique_2.jpg": (10,20,0),
        "d8d096d5-overhead_2.png": (10,10,0),
        "dfd4fed6-IMG_4413.JPG": (20,20,-4),
        "e7d7aebf-IMG_4655.JPG": (13,13,0),
        "f0b8cb04-IMG_4243.JPG": (20,20,-4),
        "f1566244-IMG_4561.jpg": (20,20,-4),
        "ff23f97b-IMG_4405.JPG": (20,20,-4),      
         }
        
        return filename_to_bbox[filename]

    def reverse_replace_string(input_str, replacement_mappings):
        reversed_mappings = {v: k for k, v in replacement_mappings.items()}
        pattern = re.compile('|'.join(re.escape(key) for key in reversed_mappings.keys()))
        result = pattern.sub(lambda x: reversed_mappings[x.group()], input_str)
        return result

    def convert_percentage_to_pixels(value, original_size):
        return int(value * original_size/100)

    def convert_rectangles_to_bbox(x_percent,y_percent,w_percent,h_percent, original_width, original_height):
        x = convert_percentage_to_pixels(x_percent, original_width)
        y = convert_percentage_to_pixels(y_percent, original_height)
        w = convert_percentage_to_pixels(w_percent, original_width)
        h = convert_percentage_to_pixels(h_percent, original_width)

        return x,y,w,h

    def convert_keypoints_to_bbox(x_percent,y_percent,bbox_width,bbox_height, original_width, original_height):
        x = convert_percentage_to_pixels(x_percent, original_width)
        y = convert_percentage_to_pixels(y_percent, original_height)
        w = bbox_width
        h = bbox_height

        return x-w/2, y-h/2, w, h

    def get_annotation_item(annotation, id, image_height, image_width, annotation_id, bbox_height, bbox_width, offset):
        if annotation['type'] == 'keypointlabels':
            category_name = annotation['value']['keypointlabels'][0]
            x, y, w, h = convert_keypoints_to_bbox(annotation['value']['x'], annotation['value']['y'], bbox_width, bbox_height, image_width, image_height)
        if annotation['type'] == 'rectanglelabels':
            category_name = annotation['value']['rectanglelabels'][0]
            x, y, w, h = convert_rectangles_to_bbox(annotation['value']['x'], annotation['value']['y'], annotation['value']['width'], annotation['value']['height'], image_width, image_height)

        annotation_item = {
            "image_id": id,
            "category_id": category_name_to_id[category_name],
            "bbox": [x+ offset, y+offset, w, h],
            "iscrowd": 0,
            "area": h * w,
            "id": annotation_id
        }
        return annotation_item

    def get_image_height_width(filename, img_foldername):
        """
        Get the height and width of an image.

        Args:
            filename (str): The name of the image file.
            img_foldername (str): The path to the folder containing the image.

        Returns:
            tuple: A tuple containing the height and width of the image (height, width).
        """
        # Join the folder path and filename to get the full path of the image
        img_full_path = os.path.join(img_foldername, filename)

        # Open the image using PIL
        with Image.open(img_full_path) as img:
            width, height = img.size
        return height, width
        
    for task in tasks:
        filename = task['file_upload']
        filename = os.path.join(image_filename_prefix, filename)
        filename = reverse_replace_string(filename, replacement_mappings)
        filename = filename.replace("%5C", "/")
        image_height, image_width = get_image_height_width(filename,image_foldername)
        bbox_height, bbox_width, offset = filename_to_bbox_height_width(filename)
        image_item = {
            "id": task['id'],
            "file_name": filename,
            "width": image_width,
            "height": image_height
        }
        images.append(image_item)

        for annotator in task['annotations']:
            for annotation in annotator['result']:
                annotation_item = get_annotation_item(annotation, task['id'], image_height, image_width, annotation_id, bbox_height, bbox_width, offset)
                annotations.append(annotation_item)
                annotation_id += 1

    mscoco_data = {
        "annotations": annotations,
        "images": images,
        "categories": categories
    }

    with open(output_json_file, 'w') as json_file:
        json.dump(mscoco_data, json_file, indent=2)

    print(f'File successfully dumped at {output_json_file}')
    ic(output_json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DFO tasks to MS COCO format")
    parser.add_argument("input_json_file", type=str, help="Path to input JSON file")
    parser.add_argument("output_json_file", type=str, help="Path to output JSON file")
    parser.add_argument("--image_filename_prefix", type=str, default='/media/pc2041/Elements/High_Arctic_Cetacean_Survey_25mm', help="Prefix for image filenames")
    parser.add_argument("--image_foldername", type=str, default='', help="The folder path which contians the images")
    parser.add_argument("--image_width", type=int, default=8256, help="Image width")
    parser.add_argument("--image_height", type=int, default=5504, help="Image height")
    parser.add_argument("--bbox_width", type=int, default=40, help="Bounding box width")
    parser.add_argument("--bbox_height", type=int, default=40, help="Bounding box height")
    args = parser.parse_args()
    main(args.input_json_file, args.output_json_file, args.image_filename_prefix,args.image_foldername,args.image_width, args.image_height, args.bbox_width, args.bbox_height)