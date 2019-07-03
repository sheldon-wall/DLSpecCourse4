
# coding: utf-8

# # Face Recognition for the Happy House
# 
# Welcome to the second assignment of week 4! Here you will build a face recognition system. Many of the ideas presented here are from [FaceNet](https://arxiv.org/pdf/1503.03832.pdf). In lecture, we also talked about [DeepFace](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf). 
# 
# Face recognition problems commonly fall into two categories: 
# 
# - **Face Verification** - "is this the claimed person?". For example, at some airports, you can pass through customs by letting a system scan your passport and then verifying that you (the person carrying the passport) are the correct person. A mobile phone that unlocks using your face is also using face verification. This is a 1:1 matching problem. 
# - **Face Recognition** - "who is this person?". For example, the video lecture showed a face recognition video (https://www.youtube.com/watch?v=wr4rx0Spihs) of Baidu employees entering the office without needing to otherwise identify themselves. This is a 1:K matching problem. 
# 
# FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, you can then determine if two pictures are of the same person.
#     
# **In this assignment, you will:**
# - Implement the triplet loss function
# - Use a pretrained model to map face images into 128-dimensional encodings
# - Use these encodings to perform face verification and face recognition
# 
# In this exercise, we will be using a pre-trained model which represents ConvNet activations using a "channels first" convention, as opposed to the "channels last" convention used in lecture and previous programming assignments. In other words, a batch of images will be of shape $(m, n_C, n_H, n_W)$ instead of $(m, n_H, n_W, n_C)$. Both of these conventions have a reasonable amount of traction among open-source implementations; there isn't a uniform standard yet within the deep learning community. 
# 
# Let's load the required packages. 
# 

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

import matplotlib

#get_ipython().magic('matplotlib inline')
#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload 2')

#np.set_printoptions(threshold=np.NAN)

# Naive Face Verification

# In Face Verification, you're given two images and you have to tell if they are of the same person.
# The simplest way to do this is to compare the two images pixel-by-pixel.
# If the distance between the raw images are less than a chosen threshold, it may be the same person!

# 1 - Encoding face images into a 128-dimensional vector
# 
# ### 1.1 - Using an ConvNet  to compute encodings
# 
# The FaceNet model takes a lot of data and a long time to train. So following common practice in applied deep learning settings, let's just load weights that someone else has already trained. The network architecture follows the Inception model from [Szegedy *et al.*](https://arxiv.org/abs/1409.4842). We have provided an inception network implementation. You can look in the file `inception_blocks.py` to see how it is implemented (do so by going to "File->Open..." at the top of the Jupyter notebook).  
# 

# The key things you need to know are:
# 
# - This network uses 96x96 dimensional RGB images as its input. Specifically, inputs a face image
# (or batch of $m$ face images) as a tensor of shape $(m, n_C, n_H, n_W) = (m, 3, 96, 96)$
# - It outputs a matrix of shape $(m, 128)$ that encodes each input face image into a
# 128-dimensional vector
# 


def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss


def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """

    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path, model)

    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding - database[identity])

    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return dist, door_open


def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """

    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path, model)

    ## Step 2: Find the closest encoding ##

    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name) in database:

        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding - database[name])

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity

# Load the FaceNet model
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Total Params:", FRmodel.count_params())

# Test the triplet loss function
with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
    loss = triplet_loss(y_true, y_pred)

    print("loss = " + str(loss.eval()))

# Loading the trained model
#
# FaceNet is trained by minimizing the triplet loss. But since training requires a lot of data
# and a lot of computation, we won't train it from scratch here. Instead, we load a previously
# trained model. Load a model using the following cell; this might take a couple of minutes to run.

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

# ### 3.1 - Face Verification
#
# Let's build a database containing one encoding vector for each person allowed to enter the
# happy house. To generate the encoding we use `img_to_encoding(image_path, model)`
# which basically runs the forward propagation of the model on the specified image.
#
# Run the following code to build the database (represented as a python dictionary).
# This database maps each person's name to a 128-dimensional encoding of their face.
# Setting up a "database" of images and their related
#
database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)

# Test verify function
verify("images/camera_0.jpg", "younes", database, FRmodel)
verify("images/camera_2.jpg", "kian", database, FRmodel)

# Face Recognition
# 
# 1. Compute the target encoding of the image from image_path
# 2. Find the encoding from the database that has smallest distance with the target encoding. 
#     - Initialize the `min_dist` variable to a large enough number (100).
#     It will help you keep track of what is the closest encoding to the input's encoding.
#     - Loop over the database dictionary's names and encodings. To loop use `for (name, db_enc)
#     in database.items()`.
#         - Compute L2 distance between the target "encoding" and the current "encoding"
#         from the database.
#         - If this distance is less than the min_dist, then set min_dist to dist,
#         and identity to name.

who_is_it("images/camera_0.jpg", database, FRmodel)

# You can change "`camera_0.jpg`" (picture of younes) to "`camera_1.jpg`"
# (picture of bertrand) and see the result.
#
# You've now seen how a state-of-the-art face recognition system works.
# 
# Although we won't implement it here, here're some ways to further improve the algorithm:
# - Put more images of each person (under different lighting conditions, taken on different days, etc.)
#   into the database. Then given a new image, compare the new face to multiple pictures of the person.
#   This would increase accuracy.
#
# - Crop the images to just contain the face, and less of the "border" region around the face.
#   This preprocessing removes some of the irrelevant pixels around the face, and also makes the
#   algorithm more robust.
# 

# Main points
# - Face verification solves an easier 1:1 matching problem; face recognition addresses a harder
#   1:K matching problem.
#
# - The triplet loss is an effective loss function for training a neural network to learn an
#   encoding of a face image.
#
# - The same encoding can be used for verification and recognition. Measuring distances between
#   two images' encodings allows you to determine whether they are pictures of the same person.
