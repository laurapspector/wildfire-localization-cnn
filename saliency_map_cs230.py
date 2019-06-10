'''
June 2019

Script to generate image-specific class saliency maps using methods presented
in PAIR-code/saliency repo. Clone the repo before running this script.
'''

import tensorflow as tf
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import pickle
import os
import cv2
from inceptionV1OnWildfire_cs230 import construct_inceptionv1onwildfire # make sure restore=True for fully connected layer
import saliency
import tensorpack.utils.viz as viz

################################################################################
# Load the graph by constructing the model and loading weights from a checkpoint
# ONLY DO ONCE (creates a new graph and modifies it each time)
model = construct_inceptionv1onwildfire (224, 224, training=False)
ckpt_file = 'models/SP-InceptionV1-OnWildfire/inceptiononv1onwildfire_adam_256_1-5053'
sess = model.session
graph = sess.graph
################################################################################

with graph.as_default():

    # Extract input tensor
    images = graph.get_tensor_by_name('InputData/X:0')
    
    # Restore the checkpoint
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)

    # Construct the scalar neuron tensor
    logits = graph.get_tensor_by_name('logits/BiasAdd:0')
    
    neuron_selector = tf.placeholder(tf.int32)
    y = logits[0][neuron_selector]

    # Construct tensor for predictions
    prediction = tf.argmax(logits, 1)

# Load an image
im = cv2.imread('all_db_fire/isolated_superpixels/test/310_rgb_sp25.png')

# Cast im as float32
im = np.float32(im)

# Make a prediction
prediction_class = sess.run( prediction, feed_dict = {images:[im]} )[0]
print("Prediction class: " + str(prediction_class))

# Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
gradient_saliency = saliency.GradientSaliency(graph, sess, y, images)

# Compute the vanilla mask and the smoothed mask.
vanilla_mask_3d = gradient_saliency.GetMask(im , feed_dict = {neuron_selector: prediction_class})

# Generate saliency map and save
abs_saliency = np.abs(vanilla_mask_3d).max(axis=-1)
abs_saliency = viz.intensity_to_rgb(abs_saliency, normalize=True)[:, :, ::-1]  # cv2 loads as BGR
cv2.imwrite("abs-saliency.jpg", abs_saliency)
