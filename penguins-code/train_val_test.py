import glob
import os
import random
import shutil


def generate_dir():
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    home = os.path.join('processed_data')
    train_home = os.path.join(home, 'train')
    if not os.path.exists(train_home):
        os.makedirs(train_home)
    val_home = os.path.join(home, 'val')
    if not os.path.exists(val_home):
        os.makedirs(val_home)
    test_home = os.path.join(home, 'test')
    if not os.path.exists(test_home):
        os.makedirs(test_home)
    train_imgs = os.path.join(train_home, 'imgs')
    train_points = os.path.join(train_home, 'points')
    train_den = os.path.join(train_home, 'gt_den')
    if not os.path.exists(train_imgs):
        os.makedirs(train_imgs)
    if not os.path.exists(train_points):
        os.makedirs(train_points)
    if not os.path.exists(train_den):
        os.makedirs(train_den)

    val_imgs = os.path.join(val_home, 'imgs')
    val_points = os.path.join(val_home, 'points')
    val_den = os.path.join(val_home, 'gt_den')

    if not os.path.exists(val_imgs):
        os.makedirs(val_imgs)
    if not os.path.exists(val_points):
        os.makedirs(val_points)
    if not os.path.exists(val_den):
        os.makedirs(val_den)

    test_imgs = os.path.join(test_home, 'imgs')
    test_points = os.path.join(test_home, 'points')
    test_den = os.path.join(test_home, 'gt_den')

    if not os.path.exists(test_imgs):
        os.makedirs(test_imgs)
    if not os.path.exists(test_points):
        os.makedirs(test_points)
    if not os.path.exists(test_den):
        os.makedirs(test_den)


def get_sample_list(img_dir):
    im_list = glob.glob(os.path.join(img_dir, '*.png'))
    num_train = round(len(im_list) * 0.604)
    num_val = round((len(im_list) - num_train) / 2)
    train_img_list = random.sample(im_list, num_train)
    sub_list = list(set(im_list) - set(train_img_list))
    val_img_list = random.sample(sub_list, num_val)
    test_img_list = list(set(sub_list) - set(val_img_list))
    return train_img_list, val_img_list, test_img_list


def move_files(img_list, destination_home_dir):
    for i in img_list:
        name = os.path.basename(i).split(".png")[0]
        den_file = i.replace('imgs', 'gt_den').replace('Bas', 'GT_Bas').replace('.png', '.h5')
        point_file = i.replace('.png', '.npy').replace('imgs', 'points')
        new_img_path = os.path.join(destination_home_dir, 'imgs', name + '.png')
        new_den_path = os.path.join(destination_home_dir, 'gt_den', 'GT_' + name + '.h5')
        new_point_path = os.path.join(destination_home_dir, 'points', name + '.npy')
        shutil.copyfile(i, new_img_path)
        shutil.copyfile(den_file, new_den_path)
        shutil.copyfile(point_file, new_point_path)


if __name__ == '__main__':
    generate_dir()
    train_img_list, val_img_list, test_img_list = get_sample_list('data/imgs')
    move_files(train_img_list,
               'processed_data/train')
    move_files(val_img_list,
               'processed_data/val')
    move_files(test_img_list,
               'processed_data/test')
