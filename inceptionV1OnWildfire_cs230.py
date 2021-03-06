'''
May 2019

This script fine tunes the SP-InceptionV1-OnFire model for classifying wildfire images
from the Corsican Fire Database (Toulouse et al., 2017).

The SP-InceptionV1-OnFire model consists of the InceptionV1-OnFire model developed by
Dunnings and Breckon (2018) loaded with pre-trained SP-InceptionV1-OnFire weights, which
can be cloned from: https://github.com/tobybreckon/fire-detection-cnn.git 
'''

################################################################################

import cv2
import os
import sys
import math
import numpy as np
import argparse

################################################################################

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, activation
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader
from tflearn.data_augmentation import ImageAugmentation

################################################################################

def construct_inceptionv1onwildfire (x, y, training=False):

    # Build network as per architecture in [Dunnings/Breckon, 2018] with some modifications:
    # Set trainable=False for first two inception modules and restore=False for weights
    # in fully connected layer

    # # Add image augmentation (optional)
    # img_aug = ImageAugmentation()
    # img_aug.add_random_blur()
    # img_aug.add_random_90degrees_rotation (rotations=[0, 1, 2, 3])

    network = input_data(shape=[None, y, x, 3], data_augmentation=None) # Set data_augmentation=img_aug if using augmentation

    conv1_7_7 = conv_2d(network, 64, 5, strides=2, activation='relu', trainable=False, name = 'conv1_7_7_s2')

    pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
    pool1_3_3 = local_response_normalization(pool1_3_3)

    conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu', trainable=False, name = 'conv2_3_3_reduce')
    conv2_3_3 = conv_2d(conv2_3_3_reduce, 128,3, activation='relu', trainable=False, name='conv2_3_3')

    conv2_3_3 = local_response_normalization(conv2_3_3)
    pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

    inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', trainable=False, name='inception_3a_1_1')

    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', trainable=False, name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', trainable=False, name = 'inception_3a_3_3')
    inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', trainable=False, name ='inception_3a_5_5_reduce' )
    inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', trainable=False, name= 'inception_3a_5_5')
    inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', trainable=False, name='inception_3a_pool_1_1')

    # merge the inception_3a__
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

    inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', trainable=False, name= 'inception_3b_1_1' )
    inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', trainable=False, name='inception_3b_3_3_reduce')
    inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu', trainable=False, name='inception_3b_3_3')
    inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', trainable=False, name = 'inception_3b_5_5_reduce')
    inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5, trainable=False,  name = 'inception_3b_5_5')
    inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
    inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', trainable=False, name='inception_3b_pool_1_1')

    #merge the inception_3b_*
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

    pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
    inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
    inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
    inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
    inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
    inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
    inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

    pool5_7_7 = avg_pool_2d(inception_4a_output, kernel_size=5, strides=1)
    pool5_7_7 = dropout(pool5_7_7, 0.4)

    # create new layer called 'logits' for constructing saliency map
    # 'loss' is simply softmax activation applied to logits

    logits = fully_connected(pool5_7_7, 2, name='logits', restore=False) # Change restore=True when predicting
    loss = activation(logits, activation='softmax')

    # if training then add training hyperparameters

    # # if using optimizer='momentum' with learning rate decay add the following:
    # momentum = Momentum(learning_rate=0.01, lr_decay=0.993, decay_step=100, staircase=False)

    if(training):
        network = regression(loss, optimizer='adam',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)

    else:
        network = loss;

    # set a checkpoint file prefix

    model = tflearn.DNN(network, checkpoint_path='inceptiononv1onwildfire',
                        max_checkpoints=2, tensorboard_verbose=0)

    return model

################################################################################

if __name__ == '__main__':

################################################################################
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='all_db_fire', help="Data directory")
    parser.add_argument('--output_dir', default='all_db_fire/isolated_superpixels', help="Location of preloaders and isolated superpixels")
    args = parser.parse_args()

    assert os.path.isdir(args.output_dir), "Couldn't find the directory {}".format(args.output_dir)
    
    #   construct and display model

    model = construct_inceptionv1onwildfire (224, 224, training=True)
    print("Constructed SP-InceptionV1-OnWildfire ...")

    # load pre-trained weights; these are stored in directory called 'models'

    model.load(os.path.join("models/SP-InceptionV1-OnFire", "sp-inceptiononv1onfire"),weights_only=True)
    print("Loaded CNN network weights ...")

################################################################################

    # network input sizes
    rows = 224
    cols = 224

    # Make directory for storing model weights and checkpoints if doesn't already exist
    if not os.path.exists('models/SP-InceptionV1-OnWildfire'):
        os.makedirs('models/SP-InceptionV1-OnWildfire')
    
    # Build dataset preloader
    print('Building dataset')

    # Note that the authors of the original model did not normalize their input
    # They also trained their network on BGR rather than RGB color arrays
    # because BGR is the format loaded by OpenCV
    # so images loaded by the preloader (stored as RGB) must be flipped

    # Build the preloader array - ONLY NEED TO DO ONCE IF SAVE

    x_train, y_train = image_preloader(os.path.join(args.data_dir, 'train.txt'), 
           image_shape=(224, 224), mode='file', categorical_labels=True, normalize=False)
    x_train = np.flip(x_train, 3)

    # optional: save for faster loading later, if space
    # np.save(os.path.join(args.data_dir, 'x_train_preloader'), x_train)
    # np.save(os.path.join(args.data_dir, 'y_train_preloader'), y_train)

    x_val, y_val = image_preloader(os.path.join(args.data_dir, 'val.txt'), image_shape=(224, 224), 
           mode='file', categorical_labels=True, normalize=False)
    x_val = np.flip(x_val, 3)

    # optional: save for faster loading later, if space
    # np.save(os.path.join(args.data_dir, 'x_val_preloader'), x_val)
    # np.save(os.path.join(args.data_dir, 'y_val_preloader'), y_val)

    # Load preloader arrays if saved as above, if desired
    # x_train = np.load(os.path.join(args.data_dir, 'x_train_preloader.npy'))
    # y_train = np.load(os.path.join(args.data_dir, 'y_train_preloader.npy'))
    # x_val = np.load(os.path.join(args.data_dir, 'x_val_preloader.npy'))
    # y_val = np.load(os.path.join(args.data_dir, 'y_val_preloader.npy'))

    # For visualizing with cv2, imshow multiplies all pixel values by 255 when dtype=float32 
    # instead of uint8 so convert to [0,255] scale and unit8 before visualizing:
    #x_train_0 = cv2.convertScaleAbs(x_train[0])
    #cv2.imshow('example', x_train_0)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    print('Finished loading dataset...')
    
    # Start fine-tuning
    
    model.fit(x_train, y_train, n_epoch=100, validation_set=(x_val, y_val), shuffle=True,
          show_metric=True, batch_size=256,
          snapshot_epoch=True, run_id='inceptiononv1onwildfire')
    
    # Save model for restoring weights
    # Alternatively, can restore model weights from saved checkpoint if terminate early
    model.save('models/inceptiononv1onwildfire')

    # To view tensorboard (call from terminal):
    # tensorboard --logdir='/tmp/tflearn_logs'

################################################################################
