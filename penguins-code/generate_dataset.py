import os.path
import pandas as pd
import json
import glob
import tqdm
import numpy as np
import shutil
from PIL import Image
import scipy.ndimage as ndimage
import h5py


def process_json_file(filename, imgFolder, outFolder):
    """
    Convert json file to npy file
    :param filename: The json file
    :param imgFolder: The corresponding image folder
    :param outFolder: The output folder
    :return: two folders one of which is the original image dataset,
    another is the corresponding annotation info for each individual image saved in .npy
    """
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    out_img_Folder = os.path.join(outFolder, 'imgs')
    if not os.path.exists(out_img_Folder):
        os.makedirs(out_img_Folder)
    out_points_Folder = os.path.join(outFolder, 'points')
    if not os.path.exists(out_points_Folder):
        os.makedirs(out_points_Folder)
    # COCO2017/annotations/instances_val2017.json
    s = json.load(open(filename, 'r'))

    str1 = "Bas"
    str2 = "png"

    for i in tqdm.tqdm(s):
        image_id = i['Labeled Data'][i['Labeled Data'].index(str1): i['Labeled Data'].index(str2) + 3]
        cx_list = []
        cy_list = []
        height_list = []
        width_list = []
        if 'objects' in i['Label'].keys():
            for j in i['Label']['objects']:
                label = j['value']
                if label != 'penguin':
                    continue
                x1 = j['bbox']['top']
                x2 = j['bbox']['top'] + j['bbox']['height']
                y1 = j['bbox']['left']
                y2 = j['bbox']['left'] + j['bbox']['width']
                x = (x1 + x2) / 2
                cx_list.append(x)
                y = (y1 + y2) / 2
                cy_list.append(y)
                width = x2 - x1
                width_list.append(width)
                height = y2 - y1
                height_list.append(height)
        i_cx_list = np.array(cx_list)
        i_cy_list = np.array(cy_list)
        i_width_list = np.array(width_list)
        i_height_list = np.array(height_list)
        i_info = np.hstack((i_cx_list.reshape(-1, 1), i_cy_list.reshape(-1, 1), i_height_list.reshape(-1, 1),
                            i_width_list.reshape(-1, 1)))
        save_path = os.path.join(out_points_Folder, image_id.split('.png')[0] + '.npy')
        if os.path.exists(os.path.join(imgFolder, image_id)):
            np.save(save_path, i_info)
            shutil.copyfile(os.path.join(imgFolder, image_id), os.path.join(out_img_Folder, image_id))


if __name__ == '__main__':
    process_json_file(
        'JSON/JACK_export-2021-08-02T07_51_56.733Z.json',
        'Jack',
        'data')
    process_json_file(
        'JSON/LUKE_export-2021-08-02T07_53_50.740Z.json',
        'Luke',
        'data')
    process_json_file(
        'JSON/MAISIE_export-2021-08-02T07_54_34.637Z.json',
        'Maisie',
        'data')
    process_json_file(
        'JSON/THOMAS_export-2021-08-02T07_52_31.079Z.json',
        'Thomas',
        'data')
    # Generate learning targets
    p_list = glob.glob(os.path.join("data/points", "*.npy"))
    if not os.path.exists(os.path.join('data', 'gt_den')):
        os.makedirs(os.path.join('data', 'gt_den'))

    for i in tqdm.tqdm(p_list):
        points = np.load(i)[:, :2].astype('int')
        im_name = i.replace('points', 'imgs').replace('npy', 'png')
        img = Image.open(im_name)
        w, h = img.size
        d = np.zeros((h, w))
        for j in range(len(points)):
            point_y, point_x = points[j][:2]
            if point_y >= h or point_x >= w:
                continue
            d[point_y - 1, point_x - 1] = 1
        d = ndimage.filters.gaussian_filter(d, 4)

        with h5py.File(i.replace('points', 'gt_den').replace('Bas', 'GT_Bas').replace('npy', 'h5'), 'w') as hf:
            hf['density_map'] = d
