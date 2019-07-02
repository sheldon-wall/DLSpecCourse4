# coding: utf-8
# Deep Learning & Art: Neural Style Transfer

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import datetime
import imageio as io

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(tf.transpose(a_C), [n_H, n_W, -1])
    a_G_unrolled = tf.reshape(tf.transpose(a_G), [n_H, n_W, -1])
    
    # compute the cost with tensorflow (≈1 line)
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4 * n_H * n_W * n_C)
    ### END CODE HERE ###
    
    return J_content

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    GA = tf.matmul(A, tf.transpose(A))
    
    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.reshape(tf.transpose(a_S), [n_C, (n_H*n_W)])
    a_G = tf.reshape(tf.transpose(a_G), [n_C, (n_H*n_W)])

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = (1/(4 * (n_H * n_W)**2 * (n_C**2))) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    
    ### END CODE HERE ###
    
    return J_style_layer


def compute_style_cost(sess, model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """

    J = (alpha * J_content) + (beta * J_style)
    
    return J


def model_nn(sess, model, train_step, J, J_content, J_style, input_image, save_iter=False, save_name=None, num_iterations=200):
    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())

    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)

        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Print every 20 iteration.
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            if save_iter is True:
                save_image("output/" + str(i) + ".png", generated_image)

    if save_name is not None:
        save_image(save_name, generated_image)

    return generated_image


def generate_image(content_loc, save_loc, style_loc, content_pic, style_pic, my_date='',
                   alpha=10, beta=40, noise=.6, save_image=True, depth='middle'):
    """
    Generates an image based on content, style and hyperparameters

    Arguments:
    data_loc -- path to the data location
    content_pic -- name of content image
    style_pic -- name of style image
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    noise -- hyperparameter indicating the initial noise on the generated image

    Returns:
    generated_image -- a PIL image generated by nueral style transfer.
    """

#    STYLE_LAYERS = [
#       ('conv1_1', 0.2),
#       ('conv2_1', 0.2),
#       ('conv3_1', 0.2),
#       ('conv4_1', 0.2),
#       ('conv5_1', 0.2)]

    if depth == 'shallow':
        STYLE_LAYERS = [
            ('conv1_1', 0.2),
            ('conv1_2', 0.2),
            ('conv2_1', 0.2),
            ('conv2_2', 0.2),
            ('conv3_1', 0.2)]
    elif depth == 'middle':
        STYLE_LAYERS = [
            ('conv2_1', 0.2),
            ('conv2_2', 0.2),
            ('conv3_1', 0.2),
            ('conv3_2', 0.2),
            ('conv3_3', 0.2)]
    elif depth == 'deep':
        STYLE_LAYERS = [
            ('conv4_1', 0.125),
            ('conv4_2', 0.125),
            ('conv4_3', 0.125),
            ('conv4_4', 0.125),
            ('conv5_1', 0.125),
            ('conv5_2', 0.125),
            ('conv5_3', 0.125),
            ('conv5_4', 0.125)]
    elif depth == 'broad':
        STYLE_LAYERS = [
            ('conv1_2', 0.066),
            ('conv2_1', 0.066),
            ('conv2_2', 0.066),
            ('conv3_1', 0.066),
            ('conv3_2', 0.066),
            ('conv3_3', 0.066),
            ('conv3_4', 0.066),
            ('conv4_1', 0.066),
            ('conv4_2', 0.066),
            ('conv4_3', 0.066),
            ('conv4_4', 0.066),
            ('conv5_1', 0.066),
            ('conv5_2', 0.066),
            ('conv5_3', 0.066),
            ('conv5_4', 0.066)]
    elif depth == 'normal':
        STYLE_LAYERS = [
            ('conv1_1', 0.2),
            ('conv2_1', 0.2),
            ('conv3_1', 0.2),
            ('conv4_1', 0.2),
            ('conv5_1', 0.2)]

    # Reset the graph
    tf.reset_default_graph()

    # Start interactive session
    sess = tf.InteractiveSession()

#    content_image = scipy.misc.imread(os.path.join(content_loc, content_pic))
    content_image = io.imread(os.path.join(content_loc, content_pic))
    content_image = scipy.misc.imresize(content_image, (600, 800))
    content_image = reshape_and_normalize_image(content_image)

    style_image = scipy.misc.imread(os.path.join(style_loc, style_pic))
    style_image = scipy.misc.imresize(style_image, (600, 800))
    style_image = reshape_and_normalize_image(style_image)

    generated_image = generate_noise_image(content_image, noise_ratio=noise)

    # Load VGG model
    model = load_vgg_model("imagenet-vgg-verydeep-19.mat")

    # To get the program to compute the content cost, we will now assign `a_C` and `a_G`
    # to be the appropriate hidden layer activations. We will use layer `conv4_2` to compute the content cost.

    # Assign the content image to be the input of the VGG model.
    sess.run(model['input'].assign(content_image))
    # Select the output tensor of layer conv4_2
    out = model['conv4_2']
    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)
    # Set a_G to be the hidden layer activation from same layer as reference only.
    # a_G is a tensor and hasn't been evaluated. It will be evaluated and updated at each iteration when we run the Tensorflow graph in model_nn() below.
    a_G = out
    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)

    # Assign the input of the model to be the "style" image
    sess.run(model['input'].assign(style_image))
    # Compute the style cost
    J_style = compute_style_cost(sess, model, STYLE_LAYERS)

    # Compute the total cost
    # J = total_cost(J_content, J_style, alpha = 10, beta=40)
    J = total_cost(J_content, J_style, alpha, beta)

    # define optimizer (1 line)
    optimizer = tf.train.AdamOptimizer(2.0)

    # define train_step optimizing for total cost
    train_step = optimizer.minimize(J)

    if save_image is True:
        # generate the save file name
        content_pic, _ = content_pic.split('.')
        style_pic, _ = style_pic.split('.')
        save_name = os.path.join(save_loc, content_pic + ' ' + my_date + ' ' +
                                 style_pic + ' ' + depth +
                                 str(alpha) + str(beta) + str(noise).replace('.','pt')) + '.png'
    else:
        save_name = None

    # Run the following to generate and save the artistic image
    generated_image = model_nn(sess, model, train_step, J, J_content, J_style, generated_image,
                               save_iter=True, save_name=save_name)
    sess.close()

    return generated_image

# Set the possible hyperparameters
alpha_beta_list = ((10, 40))
noise_list = (.3, .7)
#depth_list = ('broad','deep','normal','middle','shallow')
depth_list = ('broad','normal','shallow')

# Let's load, reshape, and normalize our "content" image (the Louvre museum picture):
#content_pic = "P1010340.jpg"  # tree blurring coast
#content_pic = "P1010296.jpg"  # sunset with rocks
#content_pic = "P1010238.jpg"  # trees
#content_pic = "P1010335.jpg"  # lighthouse
#content_pic = "P1010217.jpg"  # cathy portrait
#content_pic = "IMG_5141.jpg"  # sheldon and cathy portrait
#content_pic = "P1010339.jpg"  # trees and ocean
#content_pic = "P1010364.jpg"  # ocean and sun
#content_pic = "IMG_5127.jpg"  # lighthouse at night
#content_pic = "-814DSC_7527.jpg"  # tavis and kristen
content_pic = "P1010420.jpg"   # stream
#content_pic = "P1010430.JPG"   # pond
#content_pic = "20180914_033952327_iOS.jpg"  #sunset

# generate a single style
# Set the data location
content_loc = "C:\\NST\\Input"
content_pic = "IMG_0349.jpg"
style_loc = "styles\\"
style_pic = "Aztec 1.jpg"
save_loc = "C:\\NST\\Completed"
my_date, _ = str(datetime.datetime.now()).split('.')
my_date = my_date.replace(':', '')
generate_image(content_loc, style_loc, save_loc, content_pic, style_pic,
               my_date=my_date, save_image=True,
               alpha=10, beta=80, noise=.3, depth='normal')

# generate the selected content against all number of different styles
# for all possible hyper-parameters

content_dir = "E:\Data Science\Data\\Neural Style Transfer Images\\Input"
content_files_list = os.listdir(content_dir)

style_dir = "E:\Data Science\Data\\Neural Style Transfer Images\\Styles"
style_files_list = os.listdir(style_dir)

save_loc = "E:\Pictures\\NST"

# Set the possible hyperparameters
alpha_beta_list = ((10, 40))
alpha = 10
beta = 40
noise_list = (.3, .7)
#depth_list = ('broad','deep','normal','middle','shallow')
depth_list = ('broad','normal','shallow')

content_files_list = os.listdir(content_dir)
style_files_list = os.listdir(style_dir)

# loop through all the input images to process
for each_content in content_files_list:

    # set the name of the save directory based on the input content
    content_dir_name, _ = each_content.split('.')
    top_dir = os.path.join(save_loc, content_dir_name)

    # set the name of the sub directory based on the date and time of execution
    my_date, _ = str(datetime.datetime.now()).split('.')
    my_date = my_date.replace(':', '')
    top_dir = os.path.join(top_dir, my_date)

    # loop through all the style files to apply
    for each_style in style_files_list:
        alpha = 10
        beta = 40

        style_dir, _ = each_style.split('.')
        save_dir = os.path.join(top_dir, style_dir)
        os.makedirs(save_dir, exist_ok=True)

        # generate a new transfer image for all the parameter combinations
        for noise in noise_list:
            for depth in depth_list:
                generate_image(content_dir, style_dir, save_dir, each_content, each_style,
                               my_date=my_date, save_image=True, alpha=alpha, beta=beta,
                               noise=noise, depth=depth)

# Conclusion
# What you should remember:
# - Neural Style Transfer is an algorithm that given a content image C and a style image S can generate an artistic image
# - It uses representations (hidden layer activations) based on a pretrained ConvNet. 
# - The content cost function is computed using one hidden layer's activations.
# - The style cost function for one layer is computed using the Gram matrix of that layer's activations. The overall style cost function is obtained using several hidden layers.
# - Optimizing the total cost function results in synthesizing new images. 
# 


