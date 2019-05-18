'''
May 2019
This script performs dataset splitting and superpixel segmentation for the wildfire
localization project. It takes as input a directory of whole wildfire images, splits
them into train/val/test sets, then segments each image into superpixels. It saves
lists of x (filenames) and y (ground truth labels) for each set as numpy arrays,
as well as .txt files in the form required by the model h5py.

The script is intended to be called from within the fire-detection-cnn directory, which is also
where the data directory should be stored.
'''
import argparse
import random
import os

from tqdm import tqdm

import cv2
import os
import sys
import math
import numpy as np
from collections import defaultdict

################################################################################
random.seed(230)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='all_db_fire', help="Directory with the Corsican dataset")
parser.add_argument('--all_output_dir', default='all_db_fire/isolated_superpixels', help="Where to write the new data")

args = parser.parse_args()

assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

# Initialize dictionaries for holding the train/val/test filenames
x_all_splits = defaultdict(list)
y_all_splits = defaultdict(list)

def split_train_test_first(data_dir, all_output_dir):
    '''
    Function to split isolated superpixel images into train/val/test sets using 7:2:1 split.

    Inputs:
        all_output_dir = directory name for all processed files

    Returns:
        None

    Directory structure created:

    data_dir
    --all_output_dir
    ----train
    ----val
    ----test
    '''

    # Create output directory
    if not os.path.exists(all_output_dir):
        os.mkdir(all_output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.all_output_dir))

    # Create list of filenames in dataset. Use only the RGB images and no image sequences.
    filenames = os.listdir(data_dir)
    filenames = [i for i in filenames if i.endswith('_rgb.png')]

    # Make train/val/test splits
    filenames.sort()
    random.shuffle(filenames)

    train_proportion = 0.7
    val_proportion = 0.2
    test_proportion = 0.1

    split_one = int(train_proportion * len(filenames))
    split_two = int(val_proportion * len(filenames)) + split_one

    train_filenames = filenames[ : split_one]
    val_filenames = filenames[split_one : split_two]
    test_filenames = filenames[split_two : ]

    filenames = {'train': train_filenames,
                    'val': val_filenames,
                    'test': test_filenames}

    # Create new directories for split data sets
    for split in ['train', 'val', 'test']:

        output_dir_split = os.path.join(all_output_dir, '{}'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))
            
        print("Saving {} data".format(split))

        # Segment each image and save isolated superpixel images
        for filename in tqdm(filenames[split]):
            path_to_split = os.path.join(all_output_dir, split)
            isolate_superpixels_second(filename, data_dir, path_to_split)  

    print("Done splitting and segmenting dataset")         

    
def isolate_superpixels_second(file, data_dir, path_to_split, threshold=0.25):

    '''
    Function to isolate superpixels and sort resulting images into 'fire' or 'nonfire' 
    directories based on percent overlap of superpixel with ground truth image provided 
    by Corsican Fire Database (*_gt.png).

    Inputs:
        file = file name without path for a single rgb image
        data_dir = directory containing original image files (rgb and gt)
        path_to_split = path to specific train/val/test dataset
        threshold = (optional) threshold of pixel overlap at which to call an image 'fire'

    Returnss:
        None
    '''

    # network input sizes
    rows = 224
    cols = 224

    # Get full filename including path
    filename = os.path.join(data_dir, file)
    gt_filename = os.path.join(data_dir, file.split('rgb')[0] + 'gt.png')

    # Open each image and its paired ground truth image
    gt_img = cv2.imread(gt_filename)
    img = cv2.imread(filename)

    # Resize images, split RGB image into superpixels
    gt_small_frame = cv2.resize(gt_img, (rows, cols), cv2.INTER_AREA)
    small_frame = cv2.resize(img, (rows, cols), cv2.INTER_AREA)
    slic = cv2.ximgproc.createSuperpixelSLIC(small_frame, region_size=22)
    slic.iterate(10)
    segments = slic.getLabels()

    # loop over the unique segment values
    for (i, segVal) in enumerate(np.unique(segments)):

        # Construct a mask for the segment
        mask = np.zeros(small_frame.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255

        # get contours (first checking if OPENCV >= 4.x)
        if (int(cv2.__version__.split(".")[0]) >= 4):
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # create the superpixel by applying the mask
        superpixel = cv2.bitwise_and(small_frame, small_frame, mask = mask)
        bw = superpixel.copy()
        cv2.drawContours(bw, contours, -1, (255,255,255), thickness=cv2.FILLED)
        
        # Save the isolated superpixel image and add filename to x for its split.
        suffix = file.split('.')[0] + '_sp' + str(i) + '.png'
        new_name = os.path.join(path_to_split, suffix)
        cv2.imwrite(new_name, superpixel)
        x_all_splits[path_to_split.split('/')[-1]].append(new_name)

        # Find intersection of superpixel and ground truth image
        intersection = cv2.bitwise_and(bw, gt_small_frame)
        sum_intersection = np.sum(np.sum((intersection[:,:,0] > 0)))
        sum_superpixel = np.sum(np.sum((bw[:,:,0] > 0)))

        # Save ground truth label. Call 'fire' if threshold% of the
        # superpixel overlaps with ground truth fire (default is 25%).
        if (sum_intersection / sum_superpixel) > threshold:
            y_all_splits[path_to_split.split('/')[-1]].append(1)            
        else:
            # Call 'nonfire'
            y_all_splits[path_to_split.split('/')[-1]].append(0)

        # Save binary version superpixels for test images only (for computing IoU)
        if path_to_split.split('/')[-1] == 'test':
            gt_new_name = os.path.join(path_to_split, file.split('rgb')[0] + 'gt_sp' + str(i) + '.png')
            cv2.imwrite(gt_new_name, bw)

################################################################################

if __name__ == '__main__':

    # Split the data and isolate superpixels
    split_train_test_first(args.data_dir, args.all_output_dir)

    for split in ['train', 'val', 'test']:

        # Save a .txt file for each set that can be used by h5py in the future
        with open(os.path.join(args.data_dir, split) + '.txt', 'w') as write_to:
            write_list = [i[0] + ' ' + str(i[1]) for i in zip(x_all_splits[split], y_all_splits[split])]
            for i in range(len(write_list) - 1):
                write_to.write(write_list[i] + '\n')
            write_to.write(write_list[-1])

        # Save all x and y as numpy arrays
        np.save(os.path.join(args.all_output_dir, 'x_' + split), x_all_splits[split])
        np.save(os.path.join(args.all_output_dir, 'y_' + split), y_all_splits[split])
