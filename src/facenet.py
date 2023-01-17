"""Functions for building the face recognition network.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import Popen, PIPE
import tensorflow as tf
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from tensorflow.python.platform import gfile
import math
from six import iteritems

def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss
  
def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    print('centers',centers)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers

def Neutral_Face_center_FFA_loss(features, label, alfa, nrof_classes):

    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    print('centers',centers)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)

    loss = tf.reduce_mean(tf.square(features - centers_batch))

    return loss

def Neutral_Face_center_FFA_Cosine_loss(features, label, batch_size, nrof_classes):

    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    print('centers',centers)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)

    'cos matrix'
    # print('1 / batch_size',1 / batch_size)
    Num_val = tf.constant(1 / batch_size)
    # Num_val = 1.0

    ones_matrix = tf.constant(np.ones((batch_size, batch_size)), dtype=np.float32)
    centers_features_cos_matrix = get_cos_distance(centers_batch, features)
    centers_features_cos_matrix_subtract = ones_matrix - centers_features_cos_matrix
    centers_features_cos_diag=tf.diag_part(centers_features_cos_matrix_subtract)
    loss = tf.reduce_sum(centers_features_cos_diag)
    loss = tf.multiply(loss, Num_val)
    print('ones_matrix', ones_matrix)
    print('centers_features_cos_matrix', centers_features_cos_matrix)
    print('centers_features_cos_diag', centers_features_cos_diag)

    return loss

def Neutral_Face_IFA_loss(reconstImg, label,image_size, batch_size, nrof_classes):

    centers = tf.get_variable('image_centers', [nrof_classes, image_size,image_size,3], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    print('centers',centers)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    print('centers_batch', centers_batch)

    Num_val = tf.constant(1 / (batch_size*image_size*image_size*3))
    # Num_val = 1.0
    print('(reconstImg - centers_batch)',(reconstImg - centers_batch))
    loss = tf.reduce_mean(tf.norm((reconstImg - centers_batch),ord=1))

    loss = tf.multiply(loss, Num_val)

    return loss

def Neutral_Face_IFA_intensity_texture_loss(reconstImg, label,image_size, batch_size, nrof_classes,delt1=1.0,delt2=1/6):

    centers = tf.get_variable('image_centers', [nrof_classes, image_size,image_size,3], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    print('centers',centers)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    print('centers_batch', centers_batch)

    Num_val = tf.constant(1 / (batch_size*image_size*image_size*3))

    'loss intesity'
    print('(reconstImg - centers_batch)',(reconstImg - centers_batch))
    loss_intesity = tf.reduce_mean(tf.norm((reconstImg - centers_batch),ord=2))
    loss_intesity = tf.multiply(loss_intesity, Num_val)

    'loss texture prewitt'
    'prewitt_y'
    prewitt_y=tf.constant([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=tf.float32)
    prewitt_y_filter=tf.reshape(prewitt_y,[3,3,1,1])

    reconstImg_y_r=tf.nn.conv2d(tf.expand_dims(reconstImg[:,:,:,0],3),prewitt_y_filter,strides=[1,1,1,1],padding='SAME')
    reconstImg_y_g = tf.nn.conv2d(tf.expand_dims(reconstImg[:, :, :, 1],3), prewitt_y_filter, strides=[1, 1, 1, 1], padding='SAME')
    reconstImg_y_b = tf.nn.conv2d(tf.expand_dims(reconstImg[:, :, :, 2],3), prewitt_y_filter, strides=[1, 1, 1, 1], padding='SAME')

    oriImg_y_r=tf.nn.conv2d(tf.expand_dims(centers_batch[:,:,:,0],3),prewitt_y_filter,strides=[1,1,1,1],padding='SAME')
    oriImg_y_g = tf.nn.conv2d(tf.expand_dims(centers_batch[:, :, :, 1],3), prewitt_y_filter, strides=[1, 1, 1, 1], padding='SAME')
    oriImg_y_b = tf.nn.conv2d(tf.expand_dims(centers_batch[:, :, :, 2],3), prewitt_y_filter, strides=[1, 1, 1, 1], padding='SAME')

    print('(reconstImg_y_r - oriImg_y_r)', (reconstImg_y_r - oriImg_y_r))
    loss_texture_y_r = tf.reduce_mean(tf.norm((reconstImg_y_r - oriImg_y_r),ord=2))
    loss_texture_y_r = tf.multiply(loss_texture_y_r, Num_val)

    loss_texture_y_g = tf.reduce_mean(tf.norm((reconstImg_y_g - oriImg_y_g),ord=2))
    loss_texture_y_g = tf.multiply(loss_texture_y_g, Num_val)

    loss_texture_y_b = tf.reduce_mean(tf.norm((reconstImg_y_b - oriImg_y_b),ord=2))
    loss_texture_y_b = tf.multiply(loss_texture_y_b, Num_val)


    'prewitt_x'
    prewitt_x = tf.constant([[1, 1, 1], [0, 0, 0], [-1, -1, -1]],dtype=tf.float32)
    prewitt_x_filter = tf.reshape(prewitt_x, [3, 3, 1, 1])

    reconstImg_x_r = tf.nn.conv2d(tf.expand_dims(reconstImg[:, :, :, 0],3), prewitt_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    reconstImg_x_g = tf.nn.conv2d(tf.expand_dims(reconstImg[:, :, :, 1],3), prewitt_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    reconstImg_x_b = tf.nn.conv2d(tf.expand_dims(reconstImg[:, :, :, 2],3), prewitt_x_filter, strides=[1, 1, 1, 1], padding='SAME')

    oriImg_x_r = tf.nn.conv2d(tf.expand_dims(centers_batch[:, :, :, 0],3), prewitt_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    oriImg_x_g = tf.nn.conv2d(tf.expand_dims(centers_batch[:, :, :, 1],3), prewitt_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    oriImg_x_b = tf.nn.conv2d(tf.expand_dims(centers_batch[:, :, :, 2],3), prewitt_x_filter, strides=[1, 1, 1, 1], padding='SAME')


    print('(reconstImg_x_r - oriImg_x_r)',(reconstImg_x_r - oriImg_x_r))
    loss_texture_x_r = tf.reduce_mean(tf.norm((reconstImg_x_r - oriImg_x_r),ord=2))
    loss_texture_x_r = tf.multiply(loss_texture_x_r, Num_val)

    loss_texture_x_g = tf.reduce_mean(tf.norm((reconstImg_x_g - oriImg_x_g),ord=2))
    loss_texture_x_g = tf.multiply(loss_texture_x_g, Num_val)

    loss_texture_x_b = tf.reduce_mean(tf.norm((reconstImg_x_b - oriImg_x_b),ord=2))
    loss_texture_x_b = tf.multiply(loss_texture_x_b, Num_val)

    loss_intesity=delt1*loss_intesity
    loss_texture=delt2*(loss_texture_y_r + loss_texture_y_g + loss_texture_y_b + loss_texture_x_r + loss_texture_x_g + loss_texture_x_b)
    loss=loss_intesity+loss_texture

    print('oriImg_x_r',oriImg_x_r)
    print('oriImg_x_g', oriImg_x_g)
    print('oriImg_x_b', oriImg_x_b)
    print('oriImg_y_r', oriImg_y_r)
    print('oriImg_y_g', oriImg_y_g)
    print('oriImg_y_b', oriImg_y_b)
    return loss,loss_intesity,loss_texture

def Neutral_Face_IFA_intensity_texture_gray_loss(reconstImg, label,image_size, batch_size, nrof_classes,delt1=1.0,delt2=1/6):

    centers = tf.get_variable('image_centers', [nrof_classes, image_size,image_size,3], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    print('centers',centers)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    print('centers_batch', centers_batch)

    Num_val = tf.constant(1 / (batch_size*image_size*image_size*3))

    'loss intesity'
    print('(reconstImg - centers_batch)',(reconstImg - centers_batch))
    loss_intesity = tf.reduce_mean(tf.norm((reconstImg - centers_batch),ord=1))
    loss_intesity = tf.multiply(loss_intesity, Num_val)

    'loss texture prewitt'
    reconstImg_Gray=tf.image.rgb_to_grayscale(reconstImg)
    centers_batch_Gray = tf.image.rgb_to_grayscale(centers_batch)


    'prewitt_y'
    prewitt_y=tf.constant([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=tf.float32)
    prewitt_y_filter=tf.reshape(prewitt_y,[3,3,1,1])


    reconstImg_y=tf.nn.conv2d(reconstImg_Gray,prewitt_y_filter,strides=[1,1,1,1],padding='SAME')

    oriImg_y=tf.nn.conv2d(centers_batch_Gray,prewitt_y_filter,strides=[1,1,1,1],padding='SAME')

    print('(reconstImg_y - oriImg_y)', (reconstImg_y - oriImg_y))

    loss_texture_y = tf.reduce_mean(tf.norm((reconstImg_y - oriImg_y),ord=1))
    loss_texture_y = tf.multiply(loss_texture_y, Num_val)

    'prewitt_x'
    prewitt_x = tf.constant([[1, 1, 1], [0, 0, 0], [-1, -1, -1]],dtype=tf.float32)
    prewitt_x_filter = tf.reshape(prewitt_x, [3, 3, 1, 1])

    reconstImg_x = tf.nn.conv2d(reconstImg_Gray, prewitt_x_filter, strides=[1, 1, 1, 1], padding='SAME')

    oriImg_x = tf.nn.conv2d(centers_batch_Gray, prewitt_x_filter, strides=[1, 1, 1, 1], padding='SAME')

    print('(reconstImg_x - oriImg_x)', (reconstImg_x - oriImg_x))

    loss_texture_x = tf.reduce_mean(tf.norm((reconstImg_x - oriImg_x), ord=1))
    loss_texture_x = tf.multiply(loss_texture_x, Num_val)

    loss_intesity=delt1*loss_intesity
    loss_texture=delt2*(loss_texture_x+loss_texture_y)
    loss=loss_intesity+loss_texture

    print('reconstImg_Gray',reconstImg_Gray)
    print('centers_batch_Gray', centers_batch_Gray)

    return loss,loss_intesity,loss_texture

def Neutral_Face_IFA_intensity_texture_YCbCr_loss(reconstImg, label,image_size, batch_size, nrof_classes,delt1=1.0,delt2=1/6):

    centers = tf.get_variable('image_centers', [nrof_classes, image_size,image_size,3], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    print('centers',centers)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    print('centers_batch', centers_batch)

    Num_val = tf.constant(1 / (batch_size*image_size*image_size*3))

    'loss intesity'
    print('(reconstImg - centers_batch)',(reconstImg - centers_batch))
    loss_intesity_RGB = tf.reduce_mean(tf.norm((reconstImg - centers_batch),ord=2))
    loss_intesity_RGB = tf.multiply(loss_intesity_RGB, Num_val)

    'loss texture prewitt'
    reconstImg_Gray=0.299*reconstImg[:,:,:,0]+0.587*reconstImg[:,:,:,1]+0.114*reconstImg[:,:,:,2]
    centers_batch_Gray = 0.299*centers_batch[:,:,:,0]+0.587*centers_batch[:,:,:,1]+0.114*centers_batch[:,:,:,2]
    reconstImg_Gray=tf.expand_dims(reconstImg_Gray,-1)
    centers_batch_Gray=tf.expand_dims(centers_batch_Gray,-1)

    loss_intesity_Y = tf.reduce_mean(tf.norm((reconstImg_Gray - centers_batch_Gray),ord=2))
    loss_intesity_Y = tf.multiply(loss_intesity_Y, Num_val)

    loss_intesity=loss_intesity_RGB+loss_intesity_Y

    'prewitt_y'
    prewitt_y=tf.constant([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=tf.float32)
    prewitt_y_filter=tf.reshape(prewitt_y,[3,3,1,1])


    reconstImg_y=tf.nn.conv2d(reconstImg_Gray,prewitt_y_filter,strides=[1,1,1,1],padding='SAME')

    oriImg_y=tf.nn.conv2d(centers_batch_Gray,prewitt_y_filter,strides=[1,1,1,1],padding='SAME')

    print('(reconstImg_y - oriImg_y)', (reconstImg_y - oriImg_y))

    loss_texture_y = tf.reduce_mean(tf.norm((reconstImg_y - oriImg_y),ord=2))
    loss_texture_y = tf.multiply(loss_texture_y, Num_val)

    'prewitt_x'
    prewitt_x = tf.constant([[1, 1, 1], [0, 0, 0], [-1, -1, -1]],dtype=tf.float32)
    prewitt_x_filter = tf.reshape(prewitt_x, [3, 3, 1, 1])

    reconstImg_x = tf.nn.conv2d(reconstImg_Gray, prewitt_x_filter, strides=[1, 1, 1, 1], padding='SAME')

    oriImg_x = tf.nn.conv2d(centers_batch_Gray, prewitt_x_filter, strides=[1, 1, 1, 1], padding='SAME')

    print('(reconstImg_x - oriImg_x)', (reconstImg_x - oriImg_x))

    loss_texture_x = tf.reduce_mean(tf.norm((reconstImg_x - oriImg_x), ord=2))
    loss_texture_x = tf.multiply(loss_texture_x, Num_val)

    loss_intesity=delt1*loss_intesity
    loss_texture=delt2*(loss_texture_x+loss_texture_y)
    loss=loss_intesity+loss_texture

    print('reconstImg_Gray',reconstImg_Gray)
    print('centers_batch_Gray', centers_batch_Gray)

    return loss,loss_intesity,loss_texture


def Neutral_Face_IFA_Pretrain_intensity_texture_gray_loss(reconstImg,img_input, label,image_size, batch_size, nrof_classes,delt1=1.0,delt2=1.0):

    Num_val = tf.constant(1 / (batch_size*image_size*image_size*3))
    'loss intesity'
    print('(reconstImg - img_input)',(reconstImg - img_input))
    loss_intesity = tf.reduce_mean(tf.norm((reconstImg - img_input),ord=1))

    loss_intesity = tf.multiply(loss_intesity, Num_val)

    'loss texture prewitt'
    reconstImg_Gray=tf.image.rgb_to_grayscale(reconstImg)
    centers_batch_Gray = tf.image.rgb_to_grayscale(img_input)
    'prewitt_y'
    prewitt_y=tf.constant([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=tf.float32)
    prewitt_y_filter=tf.reshape(prewitt_y,[3,3,1,1])


    reconstImg_y=tf.nn.conv2d(reconstImg_Gray,prewitt_y_filter,strides=[1,1,1,1],padding='SAME')

    oriImg_y=tf.nn.conv2d(centers_batch_Gray,prewitt_y_filter,strides=[1,1,1,1],padding='SAME')

    print('(reconstImg_y - oriImg_y)', (reconstImg_y - oriImg_y))

    loss_texture_y = tf.reduce_mean(tf.norm((reconstImg_y - oriImg_y),ord=1))
    loss_texture_y = tf.multiply(loss_texture_y, Num_val)

    'prewitt_x'
    prewitt_x = tf.constant([[1, 1, 1], [0, 0, 0], [-1, -1, -1]],dtype=tf.float32)
    prewitt_x_filter = tf.reshape(prewitt_x, [3, 3, 1, 1])

    reconstImg_x = tf.nn.conv2d(reconstImg_Gray, prewitt_x_filter, strides=[1, 1, 1, 1], padding='SAME')

    oriImg_x = tf.nn.conv2d(centers_batch_Gray, prewitt_x_filter, strides=[1, 1, 1, 1], padding='SAME')

    print('(reconstImg_x - oriImg_x)', (reconstImg_x - oriImg_x))

    loss_texture_x = tf.reduce_mean(tf.norm((reconstImg_x - oriImg_x), ord=1))
    loss_texture_x = tf.multiply(loss_texture_x, Num_val)

    loss_intesity=delt1*loss_intesity
    loss_texture=delt2*(loss_texture_x+loss_texture_y)
    loss=loss_intesity+loss_texture

    print('reconstImg_Gray',reconstImg_Gray)
    print('centers_batch_Gray', centers_batch_Gray)

    return loss,loss_intesity,loss_texture

def Neutral_Face_IFA_Pretrain_loss(reconstImg,img_input, label,image_size, batch_size, nrof_classes):

    Num_val = tf.constant(1 / (batch_size*image_size*image_size*3))
    # Num_val = 1.0
    print('(reconstImg - img_input)',(reconstImg - img_input))
    loss = tf.reduce_mean(tf.norm((reconstImg - img_input),ord=1))

    loss = tf.multiply(loss, Num_val)

    return loss

#ASL Loss
def SLloss_Select_Class(embeddings,inter_class_num,intra_class_num,sample_class,embedding_size,Deta=0.5):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    labelsIdx = []
    labelzero=[]
    for idx in range(inter_class_num+intra_class_num):
        for i in range(sample_class):
            labelsIdx.append(idx)
            labelzero.append(0)
    print('labelsIdx', labelsIdx)
    labelsIdx = tf.constant(labelsIdx)

    with tf.variable_scope('RQ_loss'):
        class_mean = tf.segment_mean(embeddings, labelsIdx, name='class_mean')
        all_class_center = tf.segment_mean(embeddings, labelzero, name='class_mean')
        val_Multiply_class=tf.constant(1/(inter_class_num/2))
        val_Multiply_sample = tf.constant(1/(intra_class_num*sample_class))
        print('all_class_center', all_class_center)
        print('class_mean',class_mean)


        # inra-class calculation
        sampleNum=0
        for intra_classIdx in range(intra_class_num):
            class_mean_single = tf.slice(class_mean, [intra_classIdx, 0], [1, embedding_size])

            for sampleIdx in range(sample_class):
                sample_embeddings = tf.slice(embeddings, [sampleNum, 0], [1, embedding_size])
                class_embeddings_subtract = tf.subtract(sample_embeddings, class_mean_single)
                class_embeddings_subtract_square = tf.square(class_embeddings_subtract)
                pos_dist = tf.reduce_sum(class_embeddings_subtract_square)
                class_embeddings_subtract_square = tf.multiply(pos_dist, val_Multiply_sample)

                if sampleNum == 0:
                    Sw = class_embeddings_subtract_square
                    print('Sw = Sw_Tmp', sampleNum)
                else:
                    Sw = tf.add(Sw, class_embeddings_subtract_square)
                    print('Sw = tf.add(Sw, Sw_Tmp)', sampleNum)

                sampleNum += 1

                print('class_embeddings_subtract', class_embeddings_subtract)

        # Deta=0.5
        # inter-class calculation
        for classIdx in range(intra_class_num,inter_class_num+intra_class_num,2):
            print('classIdx',classIdx)
            class_mean_single_1=tf.slice(class_mean,[classIdx,0],[1,embedding_size])
            class_mean_single_2 = tf.slice(class_mean, [classIdx+1, 0], [1, embedding_size])
            class_mean_single_subtract = tf.subtract(class_mean_single_1, class_mean_single_2)
            class_mean_single_subtract_square = tf.square(class_mean_single_subtract)

            neg_dist = tf.reduce_sum(class_mean_single_subtract_square)
            Sb_loss = tf.add(tf.subtract(0.0, neg_dist), Deta)
            Sb_loss = tf.maximum(Sb_loss, 0.0)

            class_mean_single_subtract_square = tf.multiply(Sb_loss, val_Multiply_class)
            if classIdx == intra_class_num:
                Sb = class_mean_single_subtract_square
            else:
                Sb = tf.add(Sb, class_mean_single_subtract_square)
            print('class_mean_single_subtract', class_mean_single_subtract)




        print('Sw',Sw)
        print('Sb',Sb)
        # loss = tf.div(tf.trace(Sw), tf.trace(Sb))
        # loss = tf.divide(tf.reduce_sum(Sw), tf.reduce_sum(Sb))
        # pos_dist=tf.reduce_sum(Sw)
        # neg_dist=tf.reduce_sum(Sb)
        # loss = tf.subtract(pos_dist, neg_dist)
        # alpha=0.2
        # basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        # loss = tf.maximum(basic_loss, 0.0)

        loss = tf.add(Sw, Sb)
        print('loss', loss)

    return loss


def RQloss(embeddings,class_num,sample_class,embedding_size,Deta=1.5):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    labelsIdx = []
    labelzero=[]
    for idx in range(class_num):
        for i in range(sample_class):
            labelsIdx.append(idx)
            labelzero.append(0)
    print('labelsIdx', labelsIdx)
    labelsIdx = tf.constant(labelsIdx)

    with tf.variable_scope('RQ_loss'):
        class_mean = tf.segment_mean(embeddings, labelsIdx, name='class_mean')
        all_class_center = tf.segment_mean(embeddings, labelzero, name='class_mean')
        val_Multiply_class=tf.constant(1/(class_num*class_num-class_num)) #(class_num*class_num+class_num)/2  ??
        val_Multiply_sample = tf.constant(1/(sample_class*class_num))
        # print('all_class_center', all_class_center)
        # print('class_mean',class_mean)
        sampleNum=0
        # Deta=0.5
        for classIdx in range(class_num):

            class_mean_single=tf.slice(class_mean,[classIdx,0],[1,embedding_size])

            for classIdx2 in range(classIdx+1,class_num):
                class_mean_single_other = tf.slice(class_mean, [classIdx2, 0], [1, embedding_size])
                class_mean_single_subtract = tf.subtract(class_mean_single, class_mean_single_other)
                class_mean_single_subtract_square = tf.square(class_mean_single_subtract)

                neg_dist = tf.reduce_sum(class_mean_single_subtract_square)
                Sb_loss = tf.add(tf.subtract(0.0, neg_dist), Deta)
                Sb_loss = tf.maximum(Sb_loss, 0.0)


                class_mean_single_subtract_square = tf.multiply(Sb_loss, val_Multiply_class)
                if classIdx==0 and classIdx2==1:
                    Sb = class_mean_single_subtract_square
                else:
                    Sb = tf.add(Sb,class_mean_single_subtract_square)
                # print('classIdx2',classIdx,classIdx2)
            for sampleIdx in range(sample_class):


                sample_embeddings = tf.slice(embeddings, [sampleNum, 0], [1, embedding_size])
                class_embeddings_subtract = tf.subtract(sample_embeddings, class_mean_single)
                class_embeddings_subtract_square = tf.square(class_embeddings_subtract)
                pos_dist = tf.reduce_sum(class_embeddings_subtract_square)
                class_embeddings_subtract_square = tf.multiply(pos_dist, val_Multiply_sample)

                if sampleNum==0:
                    Sw = class_embeddings_subtract_square
                    # print('Sw = Sw_Tmp',sampleNum)
                else:
                    Sw = tf.add(Sw, class_embeddings_subtract_square)
                    # print('Sw = tf.add(Sw, Sw_Tmp)',sampleNum)

                sampleNum += 1

            # print('class_mean_single', class_mean_single)
            # print('class_mean_single_subtract', class_mean_single_subtract)


        print('Sw',Sw)
        print('Sb',Sb)
        # loss = tf.div(tf.trace(Sw), tf.trace(Sb))
        # loss = tf.divide(tf.reduce_sum(Sw), tf.reduce_sum(Sb))
        # pos_dist=tf.reduce_sum(Sw)
        # neg_dist=tf.reduce_sum(Sb)
        # loss = tf.subtract(pos_dist, neg_dist)
        # alpha=0.2
        # basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        # loss = tf.maximum(basic_loss, 0.0)

        loss = tf.add(Sw, Sb)
        print('loss', loss)

    return loss

def get_cos_distance(X1, X2): # https://blog.csdn.net/qq_32797059/article/details/89002313
    # calculate cos distance between two sets
    # more similar more big
    # X1=tf.transpose(X1)
    # X2=tf.transpose(X2)

    (k,n) = X1.shape
    (m,n) = X2.shape
    print('X1.shape',X1.shape)
    # 求模
    X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1), axis=1))
    X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=1))
    # 内积
    X1_X2 = tf.matmul(X1, tf.transpose(X2))
    X1_X2_norm = tf.matmul(tf.reshape(X1_norm,[-1,1]),tf.reshape(X2_norm,[1,-1]))
    # 计算余弦距离
    cos = X1_X2/X1_X2_norm
    return cos

'DFD models Cross-view Triplet Metric (CTM) loss'
def DFD_CTM_loss(embeddings,class_num,sample_class,embedding_size,alpa1=1.0,Deta1=1.0,Deta2=1.0):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    print('==========Cross-domain compact Representation==========')
    with tf.variable_scope('CdR_loss'):

        intra_class_domain_val = tf.constant(1 / (class_num))

        inter_modal_NIS_VIS_val = tf.constant(1 / (class_num))

        ones_matrix = tf.constant(np.ones((int(sample_class / 2), int(sample_class / 2)), dtype=np.float32))
        alpa1_matrix = ones_matrix*alpa1


        # inra-modal variations calculation
        for classIdx in range(0,class_num*sample_class, sample_class):

            embeddings_intra_modalityA = tf.slice(embeddings, [classIdx, 0], [int(sample_class/2), embedding_size])
            embeddings_intra_modalityB = tf.slice(embeddings, [classIdx+int(sample_class/2), 0], [int(sample_class/2), embedding_size])

            'cos matrix'
            intra_modal_cosNV_matrix=get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB)
            intra_modal_cosNV_matrix=ones_matrix-intra_modal_cosNV_matrix
            intra_modal_cosNV=tf.reduce_sum(intra_modal_cosNV_matrix)

            print('intra_modal_cosNV_matrix',intra_modal_cosNV_matrix)
            print('intra_modal_cosNV', intra_modal_cosNV)


            if classIdx == 0:
                print('intra_center = tf.multiply(intra_modal_mean_single_subtract_square_reduce, intra_class_center_val)')
                intra_domain = tf.multiply(intra_modal_cosNV, intra_class_domain_val)
            else:
                intra_domain += tf.multiply(intra_modal_cosNV, intra_class_domain_val)


            if classIdx!=(class_num-1)*sample_class:
                for classIdx2 in range(classIdx+sample_class, class_num * sample_class, sample_class):
                    print('classIdx,classIdx2', classIdx, classIdx2)
                    embeddings_intra_modalityA_classNext = tf.slice(embeddings, [classIdx2, 0], [int(sample_class / 2), embedding_size])
                    embeddings_intra_modalityB_classNext = tf.slice(embeddings, [classIdx2 + int(sample_class / 2), 0], [int(sample_class / 2), embedding_size])

                    'inter-NIR-VIS'

                    'cos matrix'
                    inter_modal_cosAB_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB_classNext)
                    inter_modal_cosAB_matrix = inter_modal_cosAB_matrix+ones_matrix - alpa1_matrix
                    inter_modal_cosAB_matrix=tf.maximum(inter_modal_cosAB_matrix, 0.0)
                    inter_modal_cosAB = tf.reduce_sum(inter_modal_cosAB_matrix)

                    print('inter_modal_cosAB_matrix', inter_modal_cosAB_matrix)
                    print('inter_modal_cosAB', inter_modal_cosAB)

                    inter_modal_cosBA_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityA_classNext)
                    inter_modal_cosBA_matrix = inter_modal_cosBA_matrix+ones_matrix - alpa1_matrix
                    inter_modal_cosBA_matrix = tf.maximum(inter_modal_cosBA_matrix, 0.0)
                    inter_modal_cosBA = tf.reduce_sum(inter_modal_cosBA_matrix)

                    print('inter_modal_cosBA_matrix', inter_modal_cosBA_matrix)

                    print('inter_modal_cosBA', inter_modal_cosBA)


                    inter_modal_NIR_VIS_dist=inter_modal_cosAB+inter_modal_cosBA
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_NIR_VIS_dist_sum = inter_modal_NIR_VIS_dist')
                        inter_modal_NIR_VIS_dist_sum = tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_NIR_VIS_dist_sum += tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)

            else:
                print('classIdx,classIdx2', classIdx, classIdx2)


        print('intra_domain',intra_domain)

        print('inter_modal_NIR_VIS_dist_sum', inter_modal_NIR_VIS_dist_sum)

        loss_intra_domain = tf.multiply(Deta1, intra_domain)

        loss_inter_modal_NIR_VIS_dist_sum = tf.multiply(Deta2, inter_modal_NIR_VIS_dist_sum)

        loss = tf.add_n([loss_intra_domain, loss_inter_modal_NIR_VIS_dist_sum ], name='total_loss')

    return loss,loss_intra_domain,loss_inter_modal_NIR_VIS_dist_sum



'DFD models Cross-domain compact Representation (CdR) loss'
def DFD_CdR_loss(embeddings,class_num,sample_class,embedding_size,alpa1=1.0,alpa2=1.0,Deta1=1.0,Deta2=1.0,Deta3=1.0,Deta4=1.0):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    print('==========Cross-domain compact Representation==========')
    with tf.variable_scope('CdR_loss'):

        intra_class_domain_val = tf.constant(1 / (class_num))
        intra_class_var_val = tf.constant(1 / (class_num))
        # inter_modal_NIS_VIS_val= tf.constant(1/(class_num*class_num))
        inter_modal_NIS_VIS_val = tf.constant(1 / (class_num))

        ones_matrix = tf.constant(np.ones((int(sample_class / 2), int(sample_class / 2)), dtype=np.float32))
        zeros_matrix = tf.constant(np.zeros((int(sample_class / 2), int(sample_class / 2)), dtype=np.float32))
        exp_zeros_matrix=tf.exp(zeros_matrix)

        exp_alpa1_matrix = tf.exp(ones_matrix*alpa1)
        exp_alpa2_matrix = tf.exp(ones_matrix * alpa2)
        exp_one_matrix = tf.exp(ones_matrix)
        sess = tf.Session()
        print('sess.run(?)', sess.run(ones_matrix*alpa1))
        print('sess.run(ones_matrix)', sess.run(ones_matrix))
        print('sess.run(exp_zeros_matrix)', sess.run(exp_zeros_matrix))
        print('sess.run(exp_alpa1_matrix)', sess.run(exp_alpa1_matrix))
        print('sess.run(exp_alpa2_matrix)', sess.run(exp_alpa2_matrix))
        print('sess.run(exp_one_matrix)', sess.run(exp_one_matrix))
        # inra-modal variations calculation
        for classIdx in range(0,class_num*sample_class, sample_class):

            embeddings_intra_modalityA = tf.slice(embeddings, [classIdx, 0], [int(sample_class/2), embedding_size])
            embeddings_intra_modalityB = tf.slice(embeddings, [classIdx+int(sample_class/2), 0], [int(sample_class/2), embedding_size])

            'cos matrix'
            intra_modal_cosNV_matrix=get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB)
            intra_modal_cosNV_matrix_exp=tf.exp(ones_matrix-intra_modal_cosNV_matrix)-exp_zeros_matrix
            intra_modal_cosNV=tf.reduce_sum(intra_modal_cosNV_matrix_exp)

            print('intra_modal_cosNV_matrix',intra_modal_cosNV_matrix)
            print('intra_modal_cosNV_matrix_exp', intra_modal_cosNV_matrix_exp)
            print('intra_modal_cosNV', intra_modal_cosNV)


            if classIdx == 0:
                print('intra_center = tf.multiply(intra_modal_mean_single_subtract_square_reduce, intra_class_center_val)')
                intra_domain = tf.multiply(intra_modal_cosNV, intra_class_domain_val)
            else:
                intra_domain += tf.multiply(intra_modal_cosNV, intra_class_domain_val)

            'intra-cor'
            'intra-cor'
            H1 = tf.transpose(embeddings_intra_modalityA)
            H2 = tf.transpose(embeddings_intra_modalityB)

            m = tf.shape(H1)[1]

            H1bar = H1 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H1, tf.ones([m, m]))
            H2bar = H2 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H2, tf.ones([m, m]))

            SigmaHat11 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H1bar, H1bar, transpose_b=True)
            SigmaHat22 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H2bar, H2bar, transpose_b=True)

            # var_11=tf.diag_part(SigmaHat11)
            # var_22 = tf.diag_part(SigmaHat22)

            intra_var_subtract=tf.subtract(SigmaHat11, SigmaHat22)
            intra_var_subtract_square=tf.square(intra_var_subtract)
            intra_var_subtract_square_reduce=tf.reduce_sum(intra_var_subtract_square)
            if classIdx == 0:
                print('intra_var = tf.multiply(intra_var_subtract_square_reduce, intra_class_var_val)')
                intra_cor = tf.multiply(intra_var_subtract_square_reduce, intra_class_var_val)
            else:
                intra_cor += tf.multiply(intra_var_subtract_square_reduce, intra_class_var_val)

            if classIdx!=(class_num-1)*sample_class:
                for classIdx2 in range(classIdx+sample_class, class_num * sample_class, sample_class):
                    print('classIdx,classIdx2', classIdx, classIdx2)
                    embeddings_intra_modalityA_classNext = tf.slice(embeddings, [classIdx2, 0], [int(sample_class / 2), embedding_size])
                    embeddings_intra_modalityB_classNext = tf.slice(embeddings, [classIdx2 + int(sample_class / 2), 0], [int(sample_class / 2), embedding_size])

                    'inter-NIR-VIS'

                    'cos matrix'
                    inter_modal_cosAB_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB_classNext)
                    inter_modal_cosAB_matrix_exp = tf.exp(inter_modal_cosAB_matrix+ones_matrix) - exp_alpa1_matrix
                    inter_modal_cosAB_matrix_exp=tf.maximum(inter_modal_cosAB_matrix_exp, 0.0)
                    inter_modal_cosAB = tf.reduce_sum(inter_modal_cosAB_matrix_exp)

                    print('inter_modal_cosAB_matrix', inter_modal_cosAB_matrix)
                    print('inter_modal_cosAB_matrix_exp', inter_modal_cosAB_matrix_exp)
                    print('inter_modal_cosAB', inter_modal_cosAB)

                    inter_modal_cosBA_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityA_classNext)
                    inter_modal_cosBA_matrix_exp = tf.exp(inter_modal_cosBA_matrix+ones_matrix) - exp_alpa1_matrix
                    inter_modal_cosBA_matrix_exp = tf.maximum(inter_modal_cosBA_matrix_exp, 0.0)
                    inter_modal_cosBA = tf.reduce_sum(inter_modal_cosBA_matrix_exp)

                    print('inter_modal_cosBA_matrix', inter_modal_cosBA_matrix)
                    print('inter_modal_cosBA_matrix_exp', inter_modal_cosBA_matrix_exp)
                    print('inter_modal_cosBA', inter_modal_cosBA)


                    inter_modal_NIR_VIS_dist=inter_modal_cosAB+inter_modal_cosBA
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_NIR_VIS_dist_sum = inter_modal_NIR_VIS_dist')
                        inter_modal_NIR_VIS_dist_sum = tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_NIR_VIS_dist_sum += tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)

                    'inter-NIR-NIR'
                    'cos matrix'
                    inter_modal_cosAA_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityA_classNext)
                    inter_modal_cosAA_matrix_exp = tf.exp(inter_modal_cosAA_matrix+ones_matrix) - exp_alpa2_matrix
                    inter_modal_cosAA_matrix_exp=tf.maximum(inter_modal_cosAA_matrix_exp, 0.0)
                    inter_modal_cosAA = tf.reduce_sum(inter_modal_cosAA_matrix_exp)

                    print('inter_modal_cosAA_matrix', inter_modal_cosAA_matrix)
                    print('inter_modal_cosAA_matrix_exp', inter_modal_cosAA_matrix_exp)
                    print('inter_modal_cosAA', inter_modal_cosAA)

                    inter_modal_cosBB_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityB_classNext)
                    inter_modal_cosBB_matrix_exp = tf.exp(inter_modal_cosBB_matrix+ones_matrix) - exp_alpa2_matrix
                    inter_modal_cosBB_matrix_exp = tf.maximum(inter_modal_cosBB_matrix_exp, 0.0)
                    inter_modal_cosBB = tf.reduce_sum(inter_modal_cosBB_matrix_exp)

                    print('inter_modal_cosBB_matrix', inter_modal_cosBB_matrix)
                    print('inter_modal_cosBB_matrix_exp', inter_modal_cosBB_matrix_exp)
                    print('inter_modal_cosBB', inter_modal_cosBB)


                    inter_modal_SameModality_dist=inter_modal_cosAA+inter_modal_cosBB
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_SameModality_sum = inter_modal_SameModality_dist')
                        inter_modal_SameModality_sum = tf.multiply(inter_modal_SameModality_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_SameModality_sum += tf.multiply(inter_modal_SameModality_dist, inter_modal_NIS_VIS_val)
            else:
                print('classIdx,classIdx2', classIdx, classIdx2)


        print('intra_domain',intra_domain)
        print('intra_cor',intra_cor)
        print('inter_modal_NIR_VIS_dist_sum', inter_modal_NIR_VIS_dist_sum)
        print('inter_modal_SameModality_sum', inter_modal_SameModality_sum)
        loss_intra_domain = tf.multiply(Deta1, intra_domain)
        loss_intra_cor = tf.multiply(Deta2, intra_cor)
        loss_inter_modal_NIR_VIS_dist_sum = tf.multiply(Deta3, inter_modal_NIR_VIS_dist_sum)
        loss_inter_modal_SameModality_sum = tf.multiply(Deta4, inter_modal_SameModality_sum)
        loss = tf.add_n([loss_intra_domain,loss_intra_cor, loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum ], name='total_loss')

    return loss,loss_intra_domain,loss_intra_cor,loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum

'DFD models Cross-domain compact Representation (CdR) loss'
def DFD_CdR_NonExp_loss(embeddings,class_num,sample_class,embedding_size,alpa1=1.0,alpa2=1.0,Deta1=1.0,Deta2=1.0,Deta3=1.0,Deta4=1.0):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    print('==========Cross-domain compact Representation==========')
    with tf.variable_scope('CdR_loss'):

        intra_class_domain_val = tf.constant(1 / (class_num))
        intra_class_var_val = tf.constant(1 / (class_num))
        # inter_modal_NIS_VIS_val= tf.constant(1/(class_num*class_num))
        inter_modal_NIS_VIS_val = tf.constant(1 / (class_num))

        ones_matrix = tf.constant(np.ones((int(sample_class / 2), int(sample_class / 2)), dtype=np.float32))
        zeros_matrix = tf.constant(np.zeros((int(sample_class / 2), int(sample_class / 2)), dtype=np.float32))
        exp_zeros_matrix=tf.exp(zeros_matrix)

        alpa1_matrix = ones_matrix*alpa1
        alpa2_matrix = ones_matrix * alpa2

        # inra-modal variations calculation
        for classIdx in range(0,class_num*sample_class, sample_class):

            embeddings_intra_modalityA = tf.slice(embeddings, [classIdx, 0], [int(sample_class/2), embedding_size])
            embeddings_intra_modalityB = tf.slice(embeddings, [classIdx+int(sample_class/2), 0], [int(sample_class/2), embedding_size])

            'cos matrix'
            intra_modal_cosNV_matrix=get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB)
            intra_modal_cosNV_matrix=ones_matrix-intra_modal_cosNV_matrix
            intra_modal_cosNV=tf.reduce_sum(intra_modal_cosNV_matrix)

            print('intra_modal_cosNV_matrix',intra_modal_cosNV_matrix)
            print('intra_modal_cosNV', intra_modal_cosNV)


            if classIdx == 0:
                print('intra_center = tf.multiply(intra_modal_mean_single_subtract_square_reduce, intra_class_center_val)')
                intra_domain = tf.multiply(intra_modal_cosNV, intra_class_domain_val)
            else:
                intra_domain += tf.multiply(intra_modal_cosNV, intra_class_domain_val)

            'intra-cor'
            'intra-cor'
            H1 = tf.transpose(embeddings_intra_modalityA)
            H2 = tf.transpose(embeddings_intra_modalityB)

            m = tf.shape(H1)[1]

            H1bar = H1 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H1, tf.ones([m, m]))
            H2bar = H2 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H2, tf.ones([m, m]))

            SigmaHat11 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H1bar, H1bar, transpose_b=True)
            SigmaHat22 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H2bar, H2bar, transpose_b=True)

            # var_11=tf.diag_part(SigmaHat11)
            # var_22 = tf.diag_part(SigmaHat22)

            intra_var_subtract=tf.subtract(SigmaHat11, SigmaHat22)
            intra_var_subtract_square=tf.square(intra_var_subtract)
            intra_var_subtract_square_reduce=tf.reduce_sum(intra_var_subtract_square)
            if classIdx == 0:
                print('intra_var = tf.multiply(intra_var_subtract_square_reduce, intra_class_var_val)')
                intra_cor = tf.multiply(intra_var_subtract_square_reduce, intra_class_var_val)
            else:
                intra_cor += tf.multiply(intra_var_subtract_square_reduce, intra_class_var_val)

            if classIdx!=(class_num-1)*sample_class:
                for classIdx2 in range(classIdx+sample_class, class_num * sample_class, sample_class):
                    print('classIdx,classIdx2', classIdx, classIdx2)
                    embeddings_intra_modalityA_classNext = tf.slice(embeddings, [classIdx2, 0], [int(sample_class / 2), embedding_size])
                    embeddings_intra_modalityB_classNext = tf.slice(embeddings, [classIdx2 + int(sample_class / 2), 0], [int(sample_class / 2), embedding_size])

                    'inter-NIR-VIS'

                    'cos matrix'
                    inter_modal_cosAB_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB_classNext)
                    inter_modal_cosAB_matrix = inter_modal_cosAB_matrix+ones_matrix - alpa1_matrix
                    inter_modal_cosAB_matrix=tf.maximum(inter_modal_cosAB_matrix, 0.0)
                    inter_modal_cosAB = tf.reduce_sum(inter_modal_cosAB_matrix)

                    print('inter_modal_cosAB_matrix', inter_modal_cosAB_matrix)
                    print('inter_modal_cosAB', inter_modal_cosAB)

                    inter_modal_cosBA_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityA_classNext)
                    inter_modal_cosBA_matrix = inter_modal_cosBA_matrix+ones_matrix - alpa1_matrix
                    inter_modal_cosBA_matrix = tf.maximum(inter_modal_cosBA_matrix, 0.0)
                    inter_modal_cosBA = tf.reduce_sum(inter_modal_cosBA_matrix)

                    print('inter_modal_cosBA_matrix', inter_modal_cosBA_matrix)

                    print('inter_modal_cosBA', inter_modal_cosBA)


                    inter_modal_NIR_VIS_dist=inter_modal_cosAB+inter_modal_cosBA
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_NIR_VIS_dist_sum = inter_modal_NIR_VIS_dist')
                        inter_modal_NIR_VIS_dist_sum = tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_NIR_VIS_dist_sum += tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)

                    'inter-NIR-NIR'
                    'cos matrix'
                    inter_modal_cosAA_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityA_classNext)
                    inter_modal_cosAA_matrix = inter_modal_cosAA_matrix+ones_matrix - alpa2_matrix
                    inter_modal_cosAA_matrix=tf.maximum(inter_modal_cosAA_matrix, 0.0)
                    inter_modal_cosAA = tf.reduce_sum(inter_modal_cosAA_matrix)

                    print('inter_modal_cosAA_matrix', inter_modal_cosAA_matrix)

                    print('inter_modal_cosAA', inter_modal_cosAA)

                    inter_modal_cosBB_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityB_classNext)
                    inter_modal_cosBB_matrix = inter_modal_cosBB_matrix+ones_matrix - alpa2_matrix
                    inter_modal_cosBB_matrix = tf.maximum(inter_modal_cosBB_matrix, 0.0)
                    inter_modal_cosBB = tf.reduce_sum(inter_modal_cosBB_matrix)

                    print('inter_modal_cosBB_matrix', inter_modal_cosBB_matrix)

                    print('inter_modal_cosBB', inter_modal_cosBB)


                    inter_modal_SameModality_dist=inter_modal_cosAA+inter_modal_cosBB
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_SameModality_sum = inter_modal_SameModality_dist')
                        inter_modal_SameModality_sum = tf.multiply(inter_modal_SameModality_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_SameModality_sum += tf.multiply(inter_modal_SameModality_dist, inter_modal_NIS_VIS_val)
            else:
                print('classIdx,classIdx2', classIdx, classIdx2)


        print('intra_domain',intra_domain)
        print('intra_cor',intra_cor)
        print('inter_modal_NIR_VIS_dist_sum', inter_modal_NIR_VIS_dist_sum)
        print('inter_modal_SameModality_sum', inter_modal_SameModality_sum)
        loss_intra_domain = tf.multiply(Deta1, intra_domain)
        loss_intra_cor = tf.multiply(Deta2, intra_cor)
        loss_inter_modal_NIR_VIS_dist_sum = tf.multiply(Deta3, inter_modal_NIR_VIS_dist_sum)
        loss_inter_modal_SameModality_sum = tf.multiply(Deta4, inter_modal_SameModality_sum)
        loss = tf.add_n([loss_intra_domain,loss_intra_cor, loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum ], name='total_loss')

    return loss,loss_intra_domain,loss_intra_cor,loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum


'DFD models DFD_CdR_alignmentCenter_loss'
def DFD_CdR_alignmentCenter_loss(embeddings,class_num,sample_class,embedding_size,alpa1=1.0,alpa2=1.0,Deta1=1.0,Deta2=1.0,Deta3=1.0,Deta4=1.0):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    print('==========DFD_CdR_alignmentCenter_loss==========')
    with tf.variable_scope('DRR_loss'):

        intra_class_domain_val = tf.constant(1 / (class_num))
        intra_class_var_val = tf.constant(1 / (class_num))
        # inter_modal_NIS_VIS_val= tf.constant(1/(class_num*class_num))
        inter_modal_NIS_VIS_val = tf.constant(1 / (class_num))

        ones_matrix = tf.constant(np.ones((int(sample_class / 2), int(sample_class / 2)), dtype=np.float32))
        zeros_matrix = tf.constant(np.zeros((int(sample_class / 2), int(sample_class / 2)), dtype=np.float32))
        exp_zeros_matrix=tf.exp(zeros_matrix)

        alpa1_matrix = ones_matrix*alpa1
        alpa2_matrix = ones_matrix * alpa2

        # inra-modal variations calculation
        for classIdx in range(0,class_num*sample_class, sample_class):

            embeddings_intra_modalityA = tf.slice(embeddings, [classIdx, 0], [int(sample_class/2), embedding_size])
            embeddings_intra_modalityB = tf.slice(embeddings, [classIdx+int(sample_class/2), 0], [int(sample_class/2), embedding_size])

            'cos matrix'
            intra_modal_cosNV_matrix=get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB)
            intra_modal_cosNV_matrix=ones_matrix-intra_modal_cosNV_matrix
            intra_modal_cosNV=tf.reduce_sum(intra_modal_cosNV_matrix)

            print('intra_modal_cosNV_matrix',intra_modal_cosNV_matrix)
            print('intra_modal_cosNV', intra_modal_cosNV)


            if classIdx == 0:
                print('intra_center = tf.multiply(intra_modal_mean_single_subtract_square_reduce, intra_class_center_val)')
                intra_domain = tf.multiply(intra_modal_cosNV, intra_class_domain_val)
            else:
                intra_domain += tf.multiply(intra_modal_cosNV, intra_class_domain_val)

            'intra-alignmentCenter'
            embeddings_intra_individual = tf.slice(embeddings, [classIdx, 0], [sample_class, embedding_size])
            intra_individual_cos_matrix = get_cos_distance(embeddings_intra_individual, embeddings_intra_individual)

            intra_individual_cos_matrix_rowSum = tf.reduce_sum(intra_individual_cos_matrix,axis=1)
            max_idx=tf.argmax(intra_individual_cos_matrix_rowSum,axis=0,output_type=tf.int32)
            embeddings_intra_center = embeddings_intra_individual[max_idx,:]
            embeddings_intra_center=tf.expand_dims(embeddings_intra_center,0)
            # tf.slice(embeddings_intra_individual, [max_idx, 0], [1, embedding_size])
            embeddings_intra_center_copy = tf.tile(embeddings_intra_center, [sample_class, 1])

            intra_center_subtract=tf.subtract(embeddings_intra_center_copy, embeddings_intra_individual)
            intra_center_subtract_square=tf.square(intra_center_subtract)
            intra_center_subtract_square_reduce=tf.reduce_sum(intra_center_subtract_square)
            if classIdx == 0:
                print('intra_center_alignment = tf.multiply(intra_center_subtract_square_reduce, intra_class_var_val)')
                intra_center_alignment = tf.multiply(intra_center_subtract_square_reduce, intra_class_var_val)
            else:
                intra_center_alignment += tf.multiply(intra_center_subtract_square_reduce, intra_class_var_val)


            if classIdx!=(class_num-1)*sample_class:
                for classIdx2 in range(classIdx+sample_class, class_num * sample_class, sample_class):
                    print('classIdx,classIdx2', classIdx, classIdx2)
                    embeddings_intra_modalityA_classNext = tf.slice(embeddings, [classIdx2, 0], [int(sample_class / 2), embedding_size])
                    embeddings_intra_modalityB_classNext = tf.slice(embeddings, [classIdx2 + int(sample_class / 2), 0], [int(sample_class / 2), embedding_size])

                    'inter-NIR-VIS'

                    'cos matrix'
                    inter_modal_cosAB_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB_classNext)
                    inter_modal_cosAB_matrix = inter_modal_cosAB_matrix+ones_matrix - alpa1_matrix
                    inter_modal_cosAB_matrix=tf.maximum(inter_modal_cosAB_matrix, 0.0)
                    inter_modal_cosAB = tf.reduce_sum(inter_modal_cosAB_matrix)

                    print('inter_modal_cosAB_matrix', inter_modal_cosAB_matrix)
                    print('inter_modal_cosAB', inter_modal_cosAB)

                    inter_modal_cosBA_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityA_classNext)
                    inter_modal_cosBA_matrix = inter_modal_cosBA_matrix+ones_matrix - alpa1_matrix
                    inter_modal_cosBA_matrix = tf.maximum(inter_modal_cosBA_matrix, 0.0)
                    inter_modal_cosBA = tf.reduce_sum(inter_modal_cosBA_matrix)

                    print('inter_modal_cosBA_matrix', inter_modal_cosBA_matrix)

                    print('inter_modal_cosBA', inter_modal_cosBA)


                    inter_modal_NIR_VIS_dist=inter_modal_cosAB+inter_modal_cosBA
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_NIR_VIS_dist_sum = inter_modal_NIR_VIS_dist')
                        inter_modal_NIR_VIS_dist_sum = tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_NIR_VIS_dist_sum += tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)

                    'inter-NIR-NIR'
                    'cos matrix'
                    inter_modal_cosAA_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityA_classNext)
                    inter_modal_cosAA_matrix = inter_modal_cosAA_matrix+ones_matrix - alpa2_matrix
                    inter_modal_cosAA_matrix=tf.maximum(inter_modal_cosAA_matrix, 0.0)
                    inter_modal_cosAA = tf.reduce_sum(inter_modal_cosAA_matrix)

                    print('inter_modal_cosAA_matrix', inter_modal_cosAA_matrix)

                    print('inter_modal_cosAA', inter_modal_cosAA)

                    inter_modal_cosBB_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityB_classNext)
                    inter_modal_cosBB_matrix = inter_modal_cosBB_matrix+ones_matrix - alpa2_matrix
                    inter_modal_cosBB_matrix = tf.maximum(inter_modal_cosBB_matrix, 0.0)
                    inter_modal_cosBB = tf.reduce_sum(inter_modal_cosBB_matrix)

                    print('inter_modal_cosBB_matrix', inter_modal_cosBB_matrix)

                    print('inter_modal_cosBB', inter_modal_cosBB)


                    inter_modal_SameModality_dist=inter_modal_cosAA+inter_modal_cosBB
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_SameModality_sum = inter_modal_SameModality_dist')
                        inter_modal_SameModality_sum = tf.multiply(inter_modal_SameModality_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_SameModality_sum += tf.multiply(inter_modal_SameModality_dist, inter_modal_NIS_VIS_val)
            else:
                print('classIdx,classIdx2', classIdx, classIdx2)


        print('intra_domain',intra_domain)
        print('intra_center_alignment',intra_center_alignment)
        print('inter_modal_NIR_VIS_dist_sum', inter_modal_NIR_VIS_dist_sum)
        print('inter_modal_SameModality_sum', inter_modal_SameModality_sum)
        loss_intra_domain = tf.multiply(Deta1, intra_domain)
        loss_intra_center_alignment = tf.multiply(Deta2, intra_center_alignment)
        loss_inter_modal_NIR_VIS_dist_sum = tf.multiply(Deta3, inter_modal_NIR_VIS_dist_sum)
        loss_inter_modal_SameModality_sum = tf.multiply(Deta4, inter_modal_SameModality_sum)
        loss = tf.add_n([loss_intra_domain,loss_intra_center_alignment, loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum ], name='total_loss')

    return loss,loss_intra_domain,loss_intra_center_alignment,loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum


'DFD models Domain-invariant Relation Representation (DRR) loss'
def DFD_DRR_loss_noRelation(embeddings,class_num,sample_class,embedding_size,alpa1=1.0,alpa2=1.0,Deta1=1.0,Deta2=1.0,Deta3=1.0,Deta4=1.0):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    print('==========Domain-invariant Relation Representation (DRR) loss==========')
    with tf.variable_scope('DRR_loss'):

        intra_class_domain_val = tf.constant(1 / (class_num))
        intra_class_relation_val = tf.constant(1 / (class_num))
        # inter_modal_NIS_VIS_val= tf.constant(1/(class_num*class_num))
        inter_modal_NIS_VIS_val = tf.constant(1 / (class_num))

        ones_matrix = tf.constant(np.ones((int(sample_class / 2), int(sample_class / 2)), dtype=np.float32))
        zeros_matrix = tf.constant(np.zeros((int(sample_class / 2), int(sample_class / 2)), dtype=np.float32))

        alpa1_matrix = ones_matrix*alpa1
        alpa2_matrix = ones_matrix * alpa2

        # inra-modal variations calculation
        for classIdx in range(0,class_num*sample_class, sample_class):

            embeddings_intra_modalityA = tf.slice(embeddings, [classIdx, 0], [int(sample_class/2), embedding_size])
            embeddings_intra_modalityB = tf.slice(embeddings, [classIdx+int(sample_class/2), 0], [int(sample_class/2), embedding_size])



            'cos matrix'
            intra_modal_cosNV_matrix=get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB)
            intra_modal_cosNV_matrix=ones_matrix-intra_modal_cosNV_matrix
            intra_modal_cosNV=tf.reduce_sum(intra_modal_cosNV_matrix)

            print('intra_modal_cosNV_matrix',intra_modal_cosNV_matrix)
            print('intra_modal_cosNV', intra_modal_cosNV)


            if classIdx == 0:
                print('intra_center = tf.multiply(intra_modal_mean_single_subtract_square_reduce, intra_class_center_val)')
                intra_domain = tf.multiply(intra_modal_cosNV, intra_class_domain_val)
            else:
                intra_domain += tf.multiply(intra_modal_cosNV, intra_class_domain_val)

            'intra-relation'
            coefficient_modalityA=np.ones([1,class_num*sample_class], dtype=np.float32)
            coefficient_modalityB = np.ones([1, class_num * sample_class], dtype=np.float32)
            coefficient_modalityA[0,classIdx:classIdx + sample_class]=0
            coefficient_modalityB[0,classIdx:classIdx + sample_class] = 0
            for idx in range(class_num):
                coefficient_modalityA[0,idx*sample_class:(idx+1)*sample_class-int(sample_class/2)]=0
                coefficient_modalityB[0,idx * sample_class+ int(sample_class / 2):(idx + 1) * sample_class] = 0

            coefficient_modalityA_single = tf.constant(coefficient_modalityA,name="coefficient_modalityA_"+str(classIdx))
            coefficient_modalityB_single = tf.constant(coefficient_modalityB,name="coefficient_modalityB_"+str(classIdx))
            coefficient_modalityA_tensor= tf.tile(coefficient_modalityA_single,[int(sample_class / 2),1])
            coefficient_modalityB_tensor = tf.tile(coefficient_modalityB_single, [int(sample_class / 2), 1])

            embeddings_intra_modalityA_embeddings_cos = get_cos_distance(embeddings_intra_modalityA, embeddings)
            embeddings_intra_modalityB_embeddings_cos = get_cos_distance(embeddings_intra_modalityB, embeddings)

            modalityA_cos_matrix= tf.multiply(coefficient_modalityA_tensor, embeddings_intra_modalityA_embeddings_cos)
            modalityB_cos_matrix = tf.multiply(coefficient_modalityB_tensor, embeddings_intra_modalityB_embeddings_cos)
            for idx1 in range(int(sample_class / 2)):
                modalityA_cos_matrix_emb = tf.slice(modalityA_cos_matrix, [idx1, 0], [1, class_num*sample_class])
                modalityB_cos_matrix_emb = tf.slice(modalityB_cos_matrix, [idx1, 0], [1, class_num * sample_class])

                if idx1 != int(sample_class / 2)-1:
                    for idx2 in range(idx1 + 1, int(sample_class / 2)):
                        print('idx1, idx2',idx1,idx2)
                        modalityA_cos_matrix_emb2 = tf.slice(modalityA_cos_matrix, [idx2, 0], [1, class_num * sample_class])
                        modalityB_cos_matrix_emb2 = tf.slice(modalityB_cos_matrix, [idx2, 0], [1, class_num * sample_class])
                        intra_relationA_subtract = tf.subtract(modalityA_cos_matrix_emb, modalityA_cos_matrix_emb2)
                        intra_relationB_subtract = tf.subtract(modalityB_cos_matrix_emb, modalityB_cos_matrix_emb2)
                        intra_relationA_subtract_square = tf.square(intra_relationA_subtract)
                        intra_relationB_subtract_square = tf.square(intra_relationB_subtract)
                        intra_relationA_subtract_square_reduce = tf.reduce_sum(intra_relationA_subtract_square)
                        intra_relationB_subtract_square_reduce = tf.reduce_sum(intra_relationB_subtract_square)
                        if idx1 == 0 and classIdx==0:
                            print('tf.multiply(intra_relationA_subtract_square_reduce+intra_relationB_subtract_square_reduce, intra_class_relation_val)')
                            intra_relation = tf.multiply(intra_relationA_subtract_square_reduce+intra_relationB_subtract_square_reduce, intra_class_relation_val)
                        else:
                            intra_relation += tf.multiply(intra_relationA_subtract_square_reduce+intra_relationB_subtract_square_reduce, intra_class_relation_val)

            if classIdx!=(class_num-1)*sample_class:
                for classIdx2 in range(classIdx+sample_class, class_num * sample_class, sample_class):
                    print('classIdx,classIdx2', classIdx, classIdx2)
                    embeddings_intra_modalityA_classNext = tf.slice(embeddings, [classIdx2, 0], [int(sample_class / 2), embedding_size])
                    embeddings_intra_modalityB_classNext = tf.slice(embeddings, [classIdx2 + int(sample_class / 2), 0], [int(sample_class / 2), embedding_size])

                    'inter-NIR-VIS'

                    'cos matrix'
                    inter_modal_cosAB_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB_classNext)
                    inter_modal_cosAB_matrix = inter_modal_cosAB_matrix+ones_matrix - alpa1_matrix
                    inter_modal_cosAB_matrix=tf.maximum(inter_modal_cosAB_matrix, 0.0)
                    inter_modal_cosAB = tf.reduce_sum(inter_modal_cosAB_matrix)

                    print('inter_modal_cosAB_matrix', inter_modal_cosAB_matrix)
                    print('inter_modal_cosAB', inter_modal_cosAB)

                    inter_modal_cosBA_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityA_classNext)
                    inter_modal_cosBA_matrix = inter_modal_cosBA_matrix+ones_matrix - alpa1_matrix
                    inter_modal_cosBA_matrix = tf.maximum(inter_modal_cosBA_matrix, 0.0)
                    inter_modal_cosBA = tf.reduce_sum(inter_modal_cosBA_matrix)

                    print('inter_modal_cosBA_matrix', inter_modal_cosBA_matrix)

                    print('inter_modal_cosBA', inter_modal_cosBA)


                    inter_modal_NIR_VIS_dist=inter_modal_cosAB+inter_modal_cosBA
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_NIR_VIS_dist_sum = inter_modal_NIR_VIS_dist')
                        inter_modal_NIR_VIS_dist_sum = tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_NIR_VIS_dist_sum += tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)

                    'inter-NIR-NIR'
                    'cos matrix'
                    inter_modal_cosAA_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityA_classNext)
                    inter_modal_cosAA_matrix = inter_modal_cosAA_matrix+ones_matrix - alpa2_matrix
                    inter_modal_cosAA_matrix=tf.maximum(inter_modal_cosAA_matrix, 0.0)
                    inter_modal_cosAA = tf.reduce_sum(inter_modal_cosAA_matrix)

                    print('inter_modal_cosAA_matrix', inter_modal_cosAA_matrix)

                    print('inter_modal_cosAA', inter_modal_cosAA)

                    inter_modal_cosBB_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityB_classNext)
                    inter_modal_cosBB_matrix = inter_modal_cosBB_matrix+ones_matrix - alpa2_matrix
                    inter_modal_cosBB_matrix = tf.maximum(inter_modal_cosBB_matrix, 0.0)
                    inter_modal_cosBB = tf.reduce_sum(inter_modal_cosBB_matrix)

                    print('inter_modal_cosBB_matrix', inter_modal_cosBB_matrix)

                    print('inter_modal_cosBB', inter_modal_cosBB)


                    inter_modal_SameModality_dist=inter_modal_cosAA+inter_modal_cosBB
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_SameModality_sum = inter_modal_SameModality_dist')
                        inter_modal_SameModality_sum = tf.multiply(inter_modal_SameModality_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_SameModality_sum += tf.multiply(inter_modal_SameModality_dist, inter_modal_NIS_VIS_val)
            else:
                print('classIdx,classIdx2', classIdx, classIdx2)


        print('intra_domain',intra_domain)
        print('intra_relation',intra_relation)
        print('inter_modal_NIR_VIS_dist_sum', inter_modal_NIR_VIS_dist_sum)
        print('inter_modal_SameModality_sum', inter_modal_SameModality_sum)
        loss_intra_domain = tf.multiply(Deta1, intra_domain)
        loss_intra_relation = tf.multiply(Deta2, intra_relation)
        loss_inter_modal_NIR_VIS_dist_sum = tf.multiply(Deta3, inter_modal_NIR_VIS_dist_sum)
        loss_inter_modal_SameModality_sum = tf.multiply(Deta4, inter_modal_SameModality_sum)
        loss = tf.add_n([loss_intra_domain, loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum ], name='total_loss')

    return loss,loss_intra_domain,loss_intra_relation,loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum


'DFD models Domain-invariant Relation Representation (DRR) loss'
def DFD_DRR_loss(embeddings,class_num,sample_class,embedding_size,alpa1=1.0,alpa2=1.0,Deta1=1.0,Deta2=1.0,Deta3=1.0,Deta4=1.0):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    print('==========Domain-invariant Relation Representation (DRR) loss==========')
    with tf.variable_scope('DRR_loss'):

        intra_class_domain_val = tf.constant(1 / (class_num))
        intra_class_relation_val = tf.constant(1 / (class_num))
        # inter_modal_NIS_VIS_val= tf.constant(1/(class_num*class_num))
        inter_modal_NIS_VIS_val = tf.constant(1 / (class_num))

        ones_matrix = tf.constant(np.ones((int(sample_class / 2), int(sample_class / 2)), dtype=np.float32))
        zeros_matrix = tf.constant(np.zeros((int(sample_class / 2), int(sample_class / 2)), dtype=np.float32))

        alpa1_matrix = ones_matrix*alpa1
        alpa2_matrix = ones_matrix * alpa2

        # inra-modal variations calculation
        for classIdx in range(0,class_num*sample_class, sample_class):

            embeddings_intra_modalityA = tf.slice(embeddings, [classIdx, 0], [int(sample_class/2), embedding_size])
            embeddings_intra_modalityB = tf.slice(embeddings, [classIdx+int(sample_class/2), 0], [int(sample_class/2), embedding_size])



            'cos matrix'
            intra_modal_cosNV_matrix=get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB)
            intra_modal_cosNV_matrix=ones_matrix-intra_modal_cosNV_matrix
            intra_modal_cosNV=tf.reduce_sum(intra_modal_cosNV_matrix)

            print('intra_modal_cosNV_matrix',intra_modal_cosNV_matrix)
            print('intra_modal_cosNV', intra_modal_cosNV)


            if classIdx == 0:
                print('intra_center = tf.multiply(intra_modal_mean_single_subtract_square_reduce, intra_class_center_val)')
                intra_domain = tf.multiply(intra_modal_cosNV, intra_class_domain_val)
            else:
                intra_domain += tf.multiply(intra_modal_cosNV, intra_class_domain_val)

            'intra-relation'
            coefficient_modalityA=np.ones([1,class_num*sample_class], dtype=np.float32)
            coefficient_modalityB = np.ones([1, class_num * sample_class], dtype=np.float32)
            coefficient_modalityA[0,classIdx:classIdx + sample_class]=0
            coefficient_modalityB[0,classIdx:classIdx + sample_class] = 0
            for idx in range(class_num):
                coefficient_modalityA[0,idx*sample_class:(idx+1)*sample_class-int(sample_class/2)]=0
                coefficient_modalityB[0,idx * sample_class+ int(sample_class / 2):(idx + 1) * sample_class] = 0

            coefficient_modalityA_single = tf.constant(coefficient_modalityA,name="coefficient_modalityA_"+str(classIdx))
            coefficient_modalityB_single = tf.constant(coefficient_modalityB,name="coefficient_modalityB_"+str(classIdx))
            coefficient_modalityA_tensor= tf.tile(coefficient_modalityA_single,[int(sample_class / 2),1])
            coefficient_modalityB_tensor = tf.tile(coefficient_modalityB_single, [int(sample_class / 2), 1])

            embeddings_intra_modalityA_embeddings_cos = get_cos_distance(embeddings_intra_modalityA, embeddings)
            embeddings_intra_modalityB_embeddings_cos = get_cos_distance(embeddings_intra_modalityB, embeddings)

            modalityA_cos_matrix= tf.multiply(coefficient_modalityA_tensor, embeddings_intra_modalityA_embeddings_cos)
            modalityB_cos_matrix = tf.multiply(coefficient_modalityB_tensor, embeddings_intra_modalityB_embeddings_cos)
            for idx1 in range(int(sample_class / 2)):
                modalityA_cos_matrix_emb = tf.slice(modalityA_cos_matrix, [idx1, 0], [1, class_num*sample_class])
                modalityB_cos_matrix_emb = tf.slice(modalityB_cos_matrix, [idx1, 0], [1, class_num * sample_class])

                if idx1 != int(sample_class / 2)-1:
                    for idx2 in range(idx1 + 1, int(sample_class / 2)):
                        print('idx1, idx2',idx1,idx2)
                        modalityA_cos_matrix_emb2 = tf.slice(modalityA_cos_matrix, [idx2, 0], [1, class_num * sample_class])
                        modalityB_cos_matrix_emb2 = tf.slice(modalityB_cos_matrix, [idx2, 0], [1, class_num * sample_class])
                        intra_relationA_subtract = tf.subtract(modalityA_cos_matrix_emb, modalityA_cos_matrix_emb2)
                        intra_relationB_subtract = tf.subtract(modalityB_cos_matrix_emb, modalityB_cos_matrix_emb2)
                        intra_relationA_subtract_square = tf.square(intra_relationA_subtract)
                        intra_relationB_subtract_square = tf.square(intra_relationB_subtract)
                        intra_relationA_subtract_square_reduce = tf.reduce_sum(intra_relationA_subtract_square)
                        intra_relationB_subtract_square_reduce = tf.reduce_sum(intra_relationB_subtract_square)
                        if idx1 == 0 and classIdx==0:
                            print('tf.multiply(intra_relationA_subtract_square_reduce+intra_relationB_subtract_square_reduce, intra_class_relation_val)')
                            intra_relation = tf.multiply(intra_relationA_subtract_square_reduce+intra_relationB_subtract_square_reduce, intra_class_relation_val)
                        else:
                            intra_relation += tf.multiply(intra_relationA_subtract_square_reduce+intra_relationB_subtract_square_reduce, intra_class_relation_val)

            if classIdx!=(class_num-1)*sample_class:
                for classIdx2 in range(classIdx+sample_class, class_num * sample_class, sample_class):
                    print('classIdx,classIdx2', classIdx, classIdx2)
                    embeddings_intra_modalityA_classNext = tf.slice(embeddings, [classIdx2, 0], [int(sample_class / 2), embedding_size])
                    embeddings_intra_modalityB_classNext = tf.slice(embeddings, [classIdx2 + int(sample_class / 2), 0], [int(sample_class / 2), embedding_size])

                    'inter-NIR-VIS'

                    'cos matrix'
                    inter_modal_cosAB_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB_classNext)
                    inter_modal_cosAB_matrix = inter_modal_cosAB_matrix+ones_matrix - alpa1_matrix
                    inter_modal_cosAB_matrix=tf.maximum(inter_modal_cosAB_matrix, 0.0)
                    inter_modal_cosAB = tf.reduce_sum(inter_modal_cosAB_matrix)

                    print('inter_modal_cosAB_matrix', inter_modal_cosAB_matrix)
                    print('inter_modal_cosAB', inter_modal_cosAB)

                    inter_modal_cosBA_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityA_classNext)
                    inter_modal_cosBA_matrix = inter_modal_cosBA_matrix+ones_matrix - alpa1_matrix
                    inter_modal_cosBA_matrix = tf.maximum(inter_modal_cosBA_matrix, 0.0)
                    inter_modal_cosBA = tf.reduce_sum(inter_modal_cosBA_matrix)

                    print('inter_modal_cosBA_matrix', inter_modal_cosBA_matrix)

                    print('inter_modal_cosBA', inter_modal_cosBA)


                    inter_modal_NIR_VIS_dist=inter_modal_cosAB+inter_modal_cosBA
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_NIR_VIS_dist_sum = inter_modal_NIR_VIS_dist')
                        inter_modal_NIR_VIS_dist_sum = tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_NIR_VIS_dist_sum += tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)

                    'inter-NIR-NIR'
                    'cos matrix'
                    inter_modal_cosAA_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityA_classNext)
                    inter_modal_cosAA_matrix = inter_modal_cosAA_matrix+ones_matrix - alpa2_matrix
                    inter_modal_cosAA_matrix=tf.maximum(inter_modal_cosAA_matrix, 0.0)
                    inter_modal_cosAA = tf.reduce_sum(inter_modal_cosAA_matrix)

                    print('inter_modal_cosAA_matrix', inter_modal_cosAA_matrix)

                    print('inter_modal_cosAA', inter_modal_cosAA)

                    inter_modal_cosBB_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityB_classNext)
                    inter_modal_cosBB_matrix = inter_modal_cosBB_matrix+ones_matrix - alpa2_matrix
                    inter_modal_cosBB_matrix = tf.maximum(inter_modal_cosBB_matrix, 0.0)
                    inter_modal_cosBB = tf.reduce_sum(inter_modal_cosBB_matrix)

                    print('inter_modal_cosBB_matrix', inter_modal_cosBB_matrix)

                    print('inter_modal_cosBB', inter_modal_cosBB)


                    inter_modal_SameModality_dist=inter_modal_cosAA+inter_modal_cosBB
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_SameModality_sum = inter_modal_SameModality_dist')
                        inter_modal_SameModality_sum = tf.multiply(inter_modal_SameModality_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_SameModality_sum += tf.multiply(inter_modal_SameModality_dist, inter_modal_NIS_VIS_val)
            else:
                print('classIdx,classIdx2', classIdx, classIdx2)


        print('intra_domain',intra_domain)
        print('intra_relation',intra_relation)
        print('inter_modal_NIR_VIS_dist_sum', inter_modal_NIR_VIS_dist_sum)
        print('inter_modal_SameModality_sum', inter_modal_SameModality_sum)
        loss_intra_domain = tf.multiply(Deta1, intra_domain)
        loss_intra_relation = tf.multiply(Deta2, intra_relation)
        loss_inter_modal_NIR_VIS_dist_sum = tf.multiply(Deta3, inter_modal_NIR_VIS_dist_sum)
        loss_inter_modal_SameModality_sum = tf.multiply(Deta4, inter_modal_SameModality_sum)
        loss = tf.add_n([loss_intra_domain,loss_intra_relation, loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum ], name='total_loss')

    return loss,loss_intra_domain,loss_intra_relation,loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum


'DFD models Domain-invariant Relation Representation (DRR) loss'
def DFD_DRR_loss_RelationAB(embeddings,class_num,sample_class,embedding_size,alpa1=1.0,alpa2=1.0,Deta1=1.0,Deta2=1.0,Deta3=1.0,Deta4=1.0):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:Neutral_Face_IFA_intensity_texture_gray_loss
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    print('==========Domain-invariant Relation Representation (DRR) loss==========')
    with tf.variable_scope('DRR_loss'):

        intra_class_domain_val = tf.constant(1 / (class_num))
        intra_class_relation_val = tf.constant(1 / (class_num))
        # inter_modal_NIS_VIS_val= tf.constant(1/(class_num*class_num))
        inter_modal_NIS_VIS_val = tf.constant(1 / (class_num))

        ones_matrix = tf.constant(np.ones((int(sample_class / 2), int(sample_class / 2)), dtype=np.float32))
        zeros_matrix = tf.constant(np.zeros((int(sample_class / 2), int(sample_class / 2)), dtype=np.float32))

        alpa1_matrix = ones_matrix*alpa1
        alpa2_matrix = ones_matrix * alpa2

        # inra-modal variations calculation
        for classIdx in range(0,class_num*sample_class, sample_class):

            embeddings_intra_modalityA = tf.slice(embeddings, [classIdx, 0], [int(sample_class/2), embedding_size])
            embeddings_intra_modalityB = tf.slice(embeddings, [classIdx+int(sample_class/2), 0], [int(sample_class/2), embedding_size])



            'cos matrix'
            intra_modal_cosNV_matrix=get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB)
            intra_modal_cosNV_matrix=ones_matrix-intra_modal_cosNV_matrix
            intra_modal_cosNV=tf.reduce_sum(intra_modal_cosNV_matrix)

            print('intra_modal_cosNV_matrix',intra_modal_cosNV_matrix)
            print('intra_modal_cosNV', intra_modal_cosNV)


            if classIdx == 0:
                print('intra_center = tf.multiply(intra_modal_mean_single_subtract_square_reduce, intra_class_center_val)')
                intra_domain = tf.multiply(intra_modal_cosNV, intra_class_domain_val)
            else:
                intra_domain += tf.multiply(intra_modal_cosNV, intra_class_domain_val)

            'intra-relation'
            coefficient_modalityA=np.ones([1,class_num*sample_class], dtype=np.float32)
            coefficient_modalityB = np.ones([1, class_num * sample_class], dtype=np.float32)
            coefficient_modalityA[0,classIdx:classIdx + sample_class]=0
            coefficient_modalityB[0,classIdx:classIdx + sample_class] = 0

            coefficient_modalityA_single = tf.constant(coefficient_modalityA,name="coefficient_modalityA_"+str(classIdx))
            coefficient_modalityB_single = tf.constant(coefficient_modalityB,name="coefficient_modalityB_"+str(classIdx))
            coefficient_modalityA_tensor= tf.tile(coefficient_modalityA_single,[int(sample_class / 2),1])
            coefficient_modalityB_tensor = tf.tile(coefficient_modalityB_single, [int(sample_class / 2), 1])

            embeddings_intra_modalityA_embeddings_cos = get_cos_distance(embeddings_intra_modalityA, embeddings)
            embeddings_intra_modalityB_embeddings_cos = get_cos_distance(embeddings_intra_modalityB, embeddings)

            modalityA_cos_matrix= tf.multiply(coefficient_modalityA_tensor, embeddings_intra_modalityA_embeddings_cos)
            modalityB_cos_matrix = tf.multiply(coefficient_modalityB_tensor, embeddings_intra_modalityB_embeddings_cos)
            for idx1 in range(int(sample_class / 2)):
                modalityA_cos_matrix_emb = tf.slice(modalityA_cos_matrix, [idx1, 0], [1, class_num*sample_class])

                for idx2 in range(int(sample_class / 2)):
                    print('idx1, idx2',idx1,idx2)

                    modalityB_cos_matrix_emb = tf.slice(modalityB_cos_matrix, [idx2, 0], [1, class_num * sample_class])
                    intra_relationAB_subtract = tf.subtract(modalityA_cos_matrix_emb, modalityB_cos_matrix_emb)

                    intra_relationAB_subtract_square = tf.square(intra_relationAB_subtract)

                    intra_relationAB_subtract_square_reduce = tf.reduce_sum(intra_relationAB_subtract_square)

                    if idx1 == 0 and classIdx==0:
                        print('tf.multiply(intra_relationA_subtract_square_reduce+intra_relationB_subtract_square_reduce, intra_class_relation_val)')
                        intra_relation = tf.multiply(intra_relationAB_subtract_square_reduce, intra_class_relation_val)
                    else:
                        intra_relation += tf.multiply(intra_relationAB_subtract_square_reduce, intra_class_relation_val)

            if classIdx!=(class_num-1)*sample_class:
                for classIdx2 in range(classIdx+sample_class, class_num * sample_class, sample_class):
                    print('classIdx,classIdx2', classIdx, classIdx2)
                    embeddings_intra_modalityA_classNext = tf.slice(embeddings, [classIdx2, 0], [int(sample_class / 2), embedding_size])
                    embeddings_intra_modalityB_classNext = tf.slice(embeddings, [classIdx2 + int(sample_class / 2), 0], [int(sample_class / 2), embedding_size])

                    'inter-NIR-VIS'

                    'cos matrix'
                    inter_modal_cosAB_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB_classNext)
                    inter_modal_cosAB_matrix = inter_modal_cosAB_matrix+ones_matrix - alpa1_matrix
                    inter_modal_cosAB_matrix=tf.maximum(inter_modal_cosAB_matrix, 0.0)
                    inter_modal_cosAB = tf.reduce_sum(inter_modal_cosAB_matrix)

                    print('inter_modal_cosAB_matrix', inter_modal_cosAB_matrix)
                    print('inter_modal_cosAB', inter_modal_cosAB)

                    inter_modal_cosBA_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityA_classNext)
                    inter_modal_cosBA_matrix = inter_modal_cosBA_matrix+ones_matrix - alpa1_matrix
                    inter_modal_cosBA_matrix = tf.maximum(inter_modal_cosBA_matrix, 0.0)
                    inter_modal_cosBA = tf.reduce_sum(inter_modal_cosBA_matrix)

                    print('inter_modal_cosBA_matrix', inter_modal_cosBA_matrix)

                    print('inter_modal_cosBA', inter_modal_cosBA)


                    inter_modal_NIR_VIS_dist=inter_modal_cosAB+inter_modal_cosBA
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_NIR_VIS_dist_sum = inter_modal_NIR_VIS_dist')
                        inter_modal_NIR_VIS_dist_sum = tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_NIR_VIS_dist_sum += tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)

                    'inter-NIR-NIR'
                    'cos matrix'
                    inter_modal_cosAA_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityA_classNext)
                    inter_modal_cosAA_matrix = inter_modal_cosAA_matrix+ones_matrix - alpa2_matrix
                    inter_modal_cosAA_matrix=tf.maximum(inter_modal_cosAA_matrix, 0.0)
                    inter_modal_cosAA = tf.reduce_sum(inter_modal_cosAA_matrix)

                    print('inter_modal_cosAA_matrix', inter_modal_cosAA_matrix)

                    print('inter_modal_cosAA', inter_modal_cosAA)

                    inter_modal_cosBB_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityB_classNext)
                    inter_modal_cosBB_matrix = inter_modal_cosBB_matrix+ones_matrix - alpa2_matrix
                    inter_modal_cosBB_matrix = tf.maximum(inter_modal_cosBB_matrix, 0.0)
                    inter_modal_cosBB = tf.reduce_sum(inter_modal_cosBB_matrix)

                    print('inter_modal_cosBB_matrix', inter_modal_cosBB_matrix)

                    print('inter_modal_cosBB', inter_modal_cosBB)


                    inter_modal_SameModality_dist=inter_modal_cosAA+inter_modal_cosBB
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_SameModality_sum = inter_modal_SameModality_dist')
                        inter_modal_SameModality_sum = tf.multiply(inter_modal_SameModality_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_SameModality_sum += tf.multiply(inter_modal_SameModality_dist, inter_modal_NIS_VIS_val)
            else:
                print('classIdx,classIdx2', classIdx, classIdx2)


        print('intra_domain',intra_domain)
        print('intra_relation',intra_relation)
        print('inter_modal_NIR_VIS_dist_sum', inter_modal_NIR_VIS_dist_sum)
        print('inter_modal_SameModality_sum', inter_modal_SameModality_sum)
        loss_intra_domain = tf.multiply(Deta1, intra_domain)
        loss_intra_relation = tf.multiply(Deta2, intra_relation)
        loss_inter_modal_NIR_VIS_dist_sum = tf.multiply(Deta3, inter_modal_NIR_VIS_dist_sum)
        loss_inter_modal_SameModality_sum = tf.multiply(Deta4, inter_modal_SameModality_sum)
        loss = tf.add_n([loss_intra_domain,loss_intra_relation, loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum ], name='total_loss')

    return loss,loss_intra_domain,loss_intra_relation,loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum


'DFD models Domain-invariant Relation Representation (DRR) loss'
def DFD_DRR_loss_RelationAB_margin(embeddings,class_num,sample_class,embedding_size,alpa1=1.0,alpa2=1.0,alpa3=1.0,Deta1=1.0,Deta2=1.0,Deta3=1.0,Deta4=1.0):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    print('==========Domain-invariant Relation Representation (DRR) loss==========')
    with tf.variable_scope('DRR_loss'):

        intra_class_domain_val = tf.constant(1 / (class_num))
        intra_class_relation_val = tf.constant(1 / (class_num))
        # inter_modal_NIS_VIS_val= tf.constant(1/(class_num*class_num))
        inter_modal_NIS_VIS_val = tf.constant(1 / (class_num))

        ones_matrix = tf.constant(np.ones((int(sample_class / 2), int(sample_class / 2)), dtype=np.float32))
        zeros_matrix = tf.constant(np.zeros((int(sample_class / 2), int(sample_class / 2)), dtype=np.float32))

        alpa1_matrix = ones_matrix*alpa1
        alpa2_matrix = ones_matrix * alpa2

        # inra-modal variations calculation
        for classIdx in range(0,class_num*sample_class, sample_class):

            embeddings_intra_modalityA = tf.slice(embeddings, [classIdx, 0], [int(sample_class/2), embedding_size])
            embeddings_intra_modalityB = tf.slice(embeddings, [classIdx+int(sample_class/2), 0], [int(sample_class/2), embedding_size])



            'cos matrix'
            intra_modal_cosNV_matrix=get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB)
            intra_modal_cosNV_matrix=ones_matrix-intra_modal_cosNV_matrix
            intra_modal_cosNV=tf.reduce_sum(intra_modal_cosNV_matrix)

            print('intra_modal_cosNV_matrix',intra_modal_cosNV_matrix)
            print('intra_modal_cosNV', intra_modal_cosNV)


            if classIdx == 0:
                print('intra_center = tf.multiply(intra_modal_mean_single_subtract_square_reduce, intra_class_center_val)')
                intra_domain = tf.multiply(intra_modal_cosNV, intra_class_domain_val)
            else:
                intra_domain += tf.multiply(intra_modal_cosNV, intra_class_domain_val)

            'intra-relation'
            coefficient_modalityA=np.ones([1,class_num*sample_class], dtype=np.float32)
            coefficient_modalityB = np.ones([1, class_num * sample_class], dtype=np.float32)
            coefficient_modalityA[0,classIdx:classIdx + sample_class]=0
            coefficient_modalityB[0,classIdx:classIdx + sample_class] = 0

            coefficient_modalityA_single = tf.constant(coefficient_modalityA,name="coefficient_modalityA_"+str(classIdx))
            coefficient_modalityB_single = tf.constant(coefficient_modalityB,name="coefficient_modalityB_"+str(classIdx))
            coefficient_modalityA_tensor= tf.tile(coefficient_modalityA_single,[int(sample_class / 2),1])
            coefficient_modalityB_tensor = tf.tile(coefficient_modalityB_single, [int(sample_class / 2), 1])

            embeddings_intra_modalityA_embeddings_cos = get_cos_distance(embeddings_intra_modalityA, embeddings)
            embeddings_intra_modalityB_embeddings_cos = get_cos_distance(embeddings_intra_modalityB, embeddings)

            modalityA_cos_matrix= tf.multiply(coefficient_modalityA_tensor, embeddings_intra_modalityA_embeddings_cos)
            modalityB_cos_matrix = tf.multiply(coefficient_modalityB_tensor, embeddings_intra_modalityB_embeddings_cos)
            for idx1 in range(int(sample_class / 2)):
                modalityA_cos_matrix_emb = tf.slice(modalityA_cos_matrix, [idx1, 0], [1, class_num*sample_class])

                for idx2 in range(int(sample_class / 2)):
                    print('idx1, idx2',idx1,idx2)

                    modalityB_cos_matrix_emb = tf.slice(modalityB_cos_matrix, [idx2, 0], [1, class_num * sample_class])
                    intra_relationAB_subtract = tf.subtract(modalityA_cos_matrix_emb, modalityB_cos_matrix_emb)

                    intra_relationAB_subtract_square = tf.square(intra_relationAB_subtract)

                    intra_relationAB_subtract_square_reduce = tf.reduce_sum(intra_relationAB_subtract_square)
                    intra_relationAB_subtract_square_reduce_ = intra_relationAB_subtract_square_reduce - alpa3
                    intra_relationAB_subtract_square_reduce_max = tf.maximum(intra_relationAB_subtract_square_reduce_, 0.0)
                    if idx1 == 0 and classIdx==0:
                        print('tf.multiply(intra_relationA_subtract_square_reduce+intra_relationB_subtract_square_reduce, intra_class_relation_val)')
                        intra_relation = tf.multiply(intra_relationAB_subtract_square_reduce_max, intra_class_relation_val)
                    else:
                        intra_relation += tf.multiply(intra_relationAB_subtract_square_reduce_max, intra_class_relation_val)

            if classIdx!=(class_num-1)*sample_class:
                for classIdx2 in range(classIdx+sample_class, class_num * sample_class, sample_class):
                    print('classIdx,classIdx2', classIdx, classIdx2)
                    embeddings_intra_modalityA_classNext = tf.slice(embeddings, [classIdx2, 0], [int(sample_class / 2), embedding_size])
                    embeddings_intra_modalityB_classNext = tf.slice(embeddings, [classIdx2 + int(sample_class / 2), 0], [int(sample_class / 2), embedding_size])

                    'inter-NIR-VIS'

                    'cos matrix'
                    inter_modal_cosAB_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB_classNext)
                    inter_modal_cosAB_matrix = inter_modal_cosAB_matrix+ones_matrix - alpa1_matrix
                    inter_modal_cosAB_matrix=tf.maximum(inter_modal_cosAB_matrix, 0.0)
                    inter_modal_cosAB = tf.reduce_sum(inter_modal_cosAB_matrix)

                    print('inter_modal_cosAB_matrix', inter_modal_cosAB_matrix)
                    print('inter_modal_cosAB', inter_modal_cosAB)

                    inter_modal_cosBA_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityA_classNext)
                    inter_modal_cosBA_matrix = inter_modal_cosBA_matrix+ones_matrix - alpa1_matrix
                    inter_modal_cosBA_matrix = tf.maximum(inter_modal_cosBA_matrix, 0.0)
                    inter_modal_cosBA = tf.reduce_sum(inter_modal_cosBA_matrix)

                    print('inter_modal_cosBA_matrix', inter_modal_cosBA_matrix)

                    print('inter_modal_cosBA', inter_modal_cosBA)


                    inter_modal_NIR_VIS_dist=inter_modal_cosAB+inter_modal_cosBA
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_NIR_VIS_dist_sum = inter_modal_NIR_VIS_dist')
                        inter_modal_NIR_VIS_dist_sum = tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_NIR_VIS_dist_sum += tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)

                    'inter-NIR-NIR'
                    'cos matrix'
                    inter_modal_cosAA_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityA_classNext)
                    inter_modal_cosAA_matrix = inter_modal_cosAA_matrix+ones_matrix - alpa2_matrix
                    inter_modal_cosAA_matrix=tf.maximum(inter_modal_cosAA_matrix, 0.0)
                    inter_modal_cosAA = tf.reduce_sum(inter_modal_cosAA_matrix)

                    print('inter_modal_cosAA_matrix', inter_modal_cosAA_matrix)

                    print('inter_modal_cosAA', inter_modal_cosAA)

                    inter_modal_cosBB_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityB_classNext)
                    inter_modal_cosBB_matrix = inter_modal_cosBB_matrix+ones_matrix - alpa2_matrix
                    inter_modal_cosBB_matrix = tf.maximum(inter_modal_cosBB_matrix, 0.0)
                    inter_modal_cosBB = tf.reduce_sum(inter_modal_cosBB_matrix)

                    print('inter_modal_cosBB_matrix', inter_modal_cosBB_matrix)

                    print('inter_modal_cosBB', inter_modal_cosBB)


                    inter_modal_SameModality_dist=inter_modal_cosAA+inter_modal_cosBB
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_SameModality_sum = inter_modal_SameModality_dist')
                        inter_modal_SameModality_sum = tf.multiply(inter_modal_SameModality_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_SameModality_sum += tf.multiply(inter_modal_SameModality_dist, inter_modal_NIS_VIS_val)
            else:
                print('classIdx,classIdx2', classIdx, classIdx2)


        print('intra_domain',intra_domain)
        print('intra_relation',intra_relation)
        print('inter_modal_NIR_VIS_dist_sum', inter_modal_NIR_VIS_dist_sum)
        print('inter_modal_SameModality_sum', inter_modal_SameModality_sum)
        loss_intra_domain = tf.multiply(Deta1, intra_domain)
        loss_intra_relation = tf.multiply(Deta2, intra_relation)
        loss_inter_modal_NIR_VIS_dist_sum = tf.multiply(Deta3, inter_modal_NIR_VIS_dist_sum)
        loss_inter_modal_SameModality_sum = tf.multiply(Deta4, inter_modal_SameModality_sum)
        loss = tf.add_n([loss_intra_domain,loss_intra_relation, loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum ], name='total_loss')

    return loss,loss_intra_domain,loss_intra_relation,loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum



'DFD models Weighted Domain-invariant Representation (WDR) loss'
def DFD_WDR_loss(embeddings,class_num,sample_class,embedding_size,Deta1=1.0,Deta2=1.0,Deta3=1.0,Deta4=1.0):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    print('==========Weighted Domain-invariant Representation (WDR) loss==========')
    with tf.variable_scope('DFD_WDR_loss'):

        intra_class_domain_val = tf.constant(1 / (class_num))
        intra_class_var_val = tf.constant(1 / (class_num))
        # inter_modal_NIS_VIS_val= tf.constant(1/(class_num*class_num))
        inter_modal_NIS_VIS_val = tf.constant(1 / (class_num))


        intra_var_subtract_square_reduce_list=[]
        inter_modal_cos_list=[]
        # inra-modal variations calculation
        for classIdx in range(0,class_num*sample_class, sample_class):

            embeddings_intra_modalityA = tf.slice(embeddings, [classIdx, 0], [int(sample_class/2), embedding_size])
            embeddings_intra_modalityB = tf.slice(embeddings, [classIdx+int(sample_class/2), 0], [int(sample_class/2), embedding_size])

            'cos matrix'
            intra_modal_cosNV_matrix=get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB)
            intra_modal_cosNV_matrix_exp=tf.exp(intra_modal_cosNV_matrix)
            intra_modal_cosNV_sum=tf.reduce_sum(intra_modal_cosNV_matrix_exp)
            W_I_ij=tf.multiply(intra_modal_cosNV_matrix_exp,1/intra_modal_cosNV_sum)
            W_I_ij=-W_I_ij
            intra_modal_cosNV_weighted=tf.multiply(W_I_ij,intra_modal_cosNV_matrix)
            intra_modal_cosNV_weighted_exp=tf.exp(intra_modal_cosNV_weighted)
            intra_modal_cosNV = tf.reduce_sum(intra_modal_cosNV_weighted_exp)

            print('intra_modal_cosNV_matrix', intra_modal_cosNV_matrix)
            print('W_I_ij',W_I_ij)
            print('intra_modal_cosNV', intra_modal_cosNV)


            if classIdx == 0:
                print('intra_center = tf.multiply(intra_modal_mean_single_subtract_square_reduce, intra_class_center_val)')
                intra_domain = tf.multiply(intra_modal_cosNV, intra_class_domain_val)
            else:
                intra_domain += tf.multiply(intra_modal_cosNV, intra_class_domain_val)

            'intra-cor'
            'intra-cor'
            H1 = tf.transpose(embeddings_intra_modalityA)
            H2 = tf.transpose(embeddings_intra_modalityB)

            m = tf.shape(H1)[1]

            H1bar = H1 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H1, tf.ones([m, m]))
            H2bar = H2 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H2, tf.ones([m, m]))

            SigmaHat11 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H1bar, H1bar, transpose_b=True)
            SigmaHat22 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H2bar, H2bar, transpose_b=True)

            # var_11=tf.diag_part(SigmaHat11)
            # var_22 = tf.diag_part(SigmaHat22)

            intra_var_subtract=tf.subtract(SigmaHat11, SigmaHat22)
            intra_var_subtract_square=tf.square(intra_var_subtract)
            intra_var_subtract_square_reduce=tf.reduce_sum(intra_var_subtract_square)
            print('intra_var_subtract_square_reduce',intra_var_subtract_square_reduce)
            intra_var_subtract_square_reduce_list.append(intra_var_subtract_square_reduce)


            if classIdx!=(class_num-1)*sample_class:
                for classIdx2 in range(classIdx+sample_class, class_num * sample_class, sample_class):
                    print('classIdx,classIdx2', classIdx, classIdx2)
                    embeddings_intra_modalityA_classNext = tf.slice(embeddings, [classIdx2, 0], [int(sample_class / 2), embedding_size])
                    embeddings_intra_modalityB_classNext = tf.slice(embeddings, [classIdx2 + int(sample_class / 2), 0], [int(sample_class / 2), embedding_size])

                    'inter-NIR-VIS'

                    'cos matrix'
                    inter_modal_cosAB_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB_classNext)

                    print('inter_modal_cosAB_matrix', inter_modal_cosAB_matrix)

                    inter_modal_cosBA_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityA_classNext)

                    print('inter_modal_cosBA_matrix', inter_modal_cosBA_matrix)

                    # inter_modal_cos_list.append(inter_modal_cosAB_matrix)
                    # inter_modal_cos_list.append(inter_modal_cosBA_matrix)
                    inter_modal_sum=tf.reduce_sum(tf.exp(inter_modal_cosAB_matrix))+tf.reduce_sum(tf.exp(inter_modal_cosBA_matrix))
                    W_B_ij_AB = tf.multiply(tf.exp(inter_modal_cosAB_matrix), 1 / inter_modal_sum)
                    print('W_B_ij_AB', W_B_ij_AB)
                    W_B_ij_BA = tf.multiply(tf.exp(inter_modal_cosBA_matrix), 1 / inter_modal_sum)
                    print('W_B_ij_BA', W_B_ij_BA)

                    inter_modal_cos_weighted_AB = tf.multiply(W_B_ij_AB, inter_modal_cosAB_matrix)
                    inter_modal_cos_weighted_BA = tf.multiply(W_B_ij_BA, inter_modal_cosBA_matrix)
                    inter_modal_cos_weighted_AB_exp = tf.exp(inter_modal_cos_weighted_AB)
                    inter_modal_cos_weighted_BA_exp = tf.exp(inter_modal_cos_weighted_BA)
                    inter_modal_NIR_VIS_dist = tf.reduce_sum(inter_modal_cos_weighted_AB_exp)+tf.reduce_sum(inter_modal_cos_weighted_BA_exp)

                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_NIR_VIS_dist_sum = inter_modal_NIR_VIS_dist')
                        inter_modal_NIR_VIS_dist_sum = tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_NIR_VIS_dist_sum += tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)

            else:
                print('classIdx,classIdx2', classIdx, classIdx2)

        'Weighted intra_cor'
        intra_cor_sum=0
        for idx,val in enumerate(intra_var_subtract_square_reduce_list):
            print('intra_var_subtract_square_reduce_list',idx,val)
            intra_cor_sum+=tf.exp(intra_var_subtract_square_reduce_list[idx])
        for idx, val in enumerate(intra_var_subtract_square_reduce_list):
            W_C_i = tf.multiply(tf.exp(intra_var_subtract_square_reduce_list[idx]), 1 / intra_cor_sum)
            intra_cor_weighted=tf.multiply(W_C_i,intra_var_subtract_square_reduce_list[idx])
            intra_cor_weighted_exp=tf.exp(intra_cor_weighted)
            if idx == 0:
                print('tf.multiply(intra_cor_weighted_exp, intra_class_var_val)')
                intra_cor = tf.multiply(intra_cor_weighted_exp, intra_class_var_val)
            else:
                intra_cor += tf.multiply(intra_cor_weighted_exp, intra_class_var_val)


        # 'Weighted inter_modal'
        # inter_modal_sum=0
        # for idx,val in enumerate(inter_modal_cos_list):
        #     print('inter_modal_cos_list',idx,val)
        #     inter_modal_sum +=tf.reduce_sum(tf.exp(inter_modal_cos_list[idx]))
        # for idx, val in enumerate(inter_modal_cos_list):
        #
        #     W_B_ij = tf.multiply(tf.exp(inter_modal_cos_list[idx]), 1 / inter_modal_sum)
        #     print('W_B_ij', W_B_ij)
        #
        #     inter_modal_cos_weighted = tf.multiply(W_B_ij, inter_modal_cos_list[idx])
        #     inter_modal_cos_weighted_exp = tf.exp(inter_modal_cos_weighted)
        #     inter_modal_NIR_VIS_dist = tf.reduce_sum(inter_modal_cos_weighted_exp)
        #
        #     if idx == 0:
        #         print('tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)')
        #         inter_modal_NIR_VIS_dist_sum = tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)
        #     else:
        #         inter_modal_NIR_VIS_dist_sum += tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)


        print('intra_domain',intra_domain)
        print('intra_cor',intra_cor)
        print('inter_modal_NIR_VIS_dist_sum', inter_modal_NIR_VIS_dist_sum)

        loss_intra_domain = tf.multiply(Deta1, intra_domain)
        loss_intra_cor = tf.multiply(Deta2, intra_cor)
        loss_inter_modal_NIR_VIS_dist_sum = tf.multiply(Deta3, inter_modal_NIR_VIS_dist_sum)

        # loss = tf.log(1 + loss_intra_domain + loss_intra_cor + loss_inter_modal_NIR_VIS_dist_sum, name='WDR_total_loss')
        loss = tf.add_n([loss_intra_domain, loss_intra_cor, loss_inter_modal_NIR_VIS_dist_sum], name='WDR_total_loss')
    return loss,loss_intra_domain,loss_intra_cor,loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_NIR_VIS_dist_sum

'DFD models Weighted Domain-invariant Representation (WDR) loss'
def DFD_WDR_loss_constant_weight(embeddings,class_num,sample_class,embedding_size,Deta1=1.0,Deta2=1.0,Deta3=1.0,Deta4=1.0):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    print('==========Weighted Domain-invariant Representation (WDR) loss==========')
    with tf.variable_scope('DFD_WDR_loss'):

        intra_class_domain_val = tf.constant(1 / (class_num))
        intra_class_var_val = tf.constant(1 / (class_num))
        # inter_modal_NIS_VIS_val= tf.constant(1/(class_num*class_num))
        inter_modal_NIS_VIS_val = tf.constant(1 / (class_num))


        intra_var_subtract_square_reduce_list=[]
        inter_modal_cos_list=[]
        # inra-modal variations calculation
        # tf.constant([[0],[1];[2],[3]])
        intra_dist_idx=np.array([[val] for val in range(int(sample_class/2))])
        for classIdx in range(0,class_num*sample_class, sample_class):

            embeddings_intra_modalityA = tf.slice(embeddings, [classIdx, 0], [int(sample_class/2), embedding_size])
            embeddings_intra_modalityB = tf.slice(embeddings, [classIdx+int(sample_class/2), 0], [int(sample_class/2), embedding_size])

            'cos matrix'
            intra_modal_cosNV_matrix=get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB)
            intra_modal_cosNV_matrix_exp=tf.exp(intra_modal_cosNV_matrix)
            intra_modal_cosNV_sum=tf.reduce_sum(intra_modal_cosNV_matrix_exp)
            W_I_ij=tf.multiply(intra_modal_cosNV_matrix_exp,1/intra_modal_cosNV_sum)
            W_I_ij=-W_I_ij
            print('W_I_ij',W_I_ij)
            W_I_ij_tensor = tf.get_variable('W_I_ij_tensor_'+str(classIdx+1), [int(sample_class/2), int(sample_class/2)], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
            W_I_ij_tensor = tf.scatter_nd_update(W_I_ij_tensor, intra_dist_idx, W_I_ij)
            intra_modal_cosNV_weighted=tf.multiply(W_I_ij_tensor,intra_modal_cosNV_matrix)
            # intra_modal_cosNV_weighted_exp=tf.exp(intra_modal_cosNV_weighted)
            intra_modal_cosNV = tf.reduce_sum(intra_modal_cosNV_weighted)

            print('intra_modal_cosNV_matrix', intra_modal_cosNV_matrix)
            print('W_I_ij_tensor',W_I_ij_tensor)
            print('intra_modal_cosNV', intra_modal_cosNV)


            if classIdx == 0:
                print('intra_center = tf.multiply(intra_modal_mean_single_subtract_square_reduce, intra_class_center_val)')
                intra_domain = tf.multiply(intra_modal_cosNV, intra_class_domain_val)
            else:
                intra_domain += tf.multiply(intra_modal_cosNV, intra_class_domain_val)

            'intra-cor'
            'intra-cor'
            H1 = tf.transpose(embeddings_intra_modalityA)
            H2 = tf.transpose(embeddings_intra_modalityB)

            m = tf.shape(H1)[1]

            H1bar = H1 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H1, tf.ones([m, m]))
            H2bar = H2 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H2, tf.ones([m, m]))

            SigmaHat11 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H1bar, H1bar, transpose_b=True)
            SigmaHat22 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H2bar, H2bar, transpose_b=True)

            # var_11=tf.diag_part(SigmaHat11)
            # var_22 = tf.diag_part(SigmaHat22)

            intra_var_subtract=tf.subtract(SigmaHat11, SigmaHat22)
            intra_var_subtract_square=tf.square(intra_var_subtract)
            intra_var_subtract_square_reduce=tf.reduce_sum(intra_var_subtract_square)
            print('intra_var_subtract_square_reduce',intra_var_subtract_square_reduce)
            intra_var_subtract_square_reduce_list.append(intra_var_subtract_square_reduce)


            if classIdx!=(class_num-1)*sample_class:
                for classIdx2 in range(classIdx+sample_class, class_num * sample_class, sample_class):
                    print('classIdx,classIdx2', classIdx, classIdx2)
                    embeddings_intra_modalityA_classNext = tf.slice(embeddings, [classIdx2, 0], [int(sample_class / 2), embedding_size])
                    embeddings_intra_modalityB_classNext = tf.slice(embeddings, [classIdx2 + int(sample_class / 2), 0], [int(sample_class / 2), embedding_size])

                    'inter-NIR-VIS    inter-VIS-NIR'

                    'cos matrix'
                    inter_modal_cosAB_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityB_classNext)

                    print('inter_modal_cosAB_matrix', inter_modal_cosAB_matrix)

                    inter_modal_cosBA_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityA_classNext)

                    print('inter_modal_cosBA_matrix', inter_modal_cosBA_matrix)

                    'inter-NIR-NIR    inter-VIS-VIS'
                    'cos matrix'
                    inter_modal_cosAA_matrix = get_cos_distance(embeddings_intra_modalityA, embeddings_intra_modalityA_classNext)

                    inter_modal_cosBB_matrix = get_cos_distance(embeddings_intra_modalityB, embeddings_intra_modalityB_classNext)

                    print('inter_modal_cosAA_matrix', inter_modal_cosAA_matrix)
                    print('inter_modal_cosBB_matrix', inter_modal_cosBB_matrix)

                    'inter-AB'
                    inter_modal_sum=tf.reduce_sum(tf.exp(inter_modal_cosAB_matrix))+tf.reduce_sum(tf.exp(inter_modal_cosBA_matrix))+tf.reduce_sum(tf.exp(inter_modal_cosAA_matrix))+tf.reduce_sum(tf.exp(inter_modal_cosBB_matrix))
                    W_B_ij_AB = tf.multiply(tf.exp(inter_modal_cosAB_matrix), 1 / inter_modal_sum)
                    W_B_ij_AB_tensor = tf.get_variable('W_B_ij_AB_tensor_' + str(classIdx + 1)+'_'+ str(classIdx2 + 1),
                                                    [int(sample_class / 2), int(sample_class / 2)], dtype=tf.float32,
                                                    initializer=tf.constant_initializer(0), trainable=False)
                    W_B_ij_AB_tensor = tf.scatter_nd_update(W_B_ij_AB_tensor, intra_dist_idx, W_B_ij_AB)
                    print('W_B_ij_AB_tensor', W_B_ij_AB_tensor)

                    'inter-BA'
                    W_B_ij_BA = tf.multiply(tf.exp(inter_modal_cosBA_matrix), 1 / inter_modal_sum)
                    W_B_ij_BA_tensor = tf.get_variable('W_B_ij_BA_tensor_' + str(classIdx + 1)+'_'+ str(classIdx2 + 1),
                                                    [int(sample_class / 2), int(sample_class / 2)], dtype=tf.float32,
                                                    initializer=tf.constant_initializer(0), trainable=False)
                    W_B_ij_BA_tensor = tf.scatter_nd_update(W_B_ij_BA_tensor, intra_dist_idx, W_B_ij_BA)
                    print('W_B_ij_BA', W_B_ij_BA)
                    print('W_B_ij_BA_tensor', W_B_ij_BA_tensor)

                    'inter-AA'
                    W_B_ij_AA = tf.multiply(tf.exp(inter_modal_cosAA_matrix), 1 / inter_modal_sum)
                    W_B_ij_AA_tensor = tf.get_variable('W_B_ij_AA_tensor_' + str(classIdx + 1)+'_'+ str(classIdx2 + 1),
                                                    [int(sample_class / 2), int(sample_class / 2)], dtype=tf.float32,
                                                    initializer=tf.constant_initializer(0), trainable=False)
                    W_B_ij_AA_tensor = tf.scatter_nd_update(W_B_ij_AA_tensor, intra_dist_idx, W_B_ij_AA)
                    print('W_B_ij_AA', W_B_ij_AA)
                    print('W_B_ij_AA_tensor', W_B_ij_AA_tensor)

                    'inter-BB'
                    W_B_ij_BB = tf.multiply(tf.exp(inter_modal_cosBB_matrix), 1 / inter_modal_sum)
                    W_B_ij_BB_tensor = tf.get_variable('W_B_ij_BB_tensor_' + str(classIdx + 1)+'_'+ str(classIdx2 + 1),
                                                    [int(sample_class / 2), int(sample_class / 2)], dtype=tf.float32,
                                                    initializer=tf.constant_initializer(0), trainable=False)
                    W_B_ij_BB_tensor = tf.scatter_nd_update(W_B_ij_BB_tensor, intra_dist_idx, W_B_ij_BB)
                    print('W_B_ij_BB', W_B_ij_BB)
                    print('W_B_ij_BB_tensor', W_B_ij_BB_tensor)


                    inter_modal_cos_weighted_AB = tf.multiply(W_B_ij_AB_tensor, inter_modal_cosAB_matrix)
                    inter_modal_cos_weighted_BA = tf.multiply(W_B_ij_BA_tensor, inter_modal_cosBA_matrix)
                    inter_modal_cos_weighted_AA = tf.multiply(W_B_ij_AA_tensor, inter_modal_cosAA_matrix)
                    inter_modal_cos_weighted_BB = tf.multiply(W_B_ij_BB_tensor, inter_modal_cosBB_matrix)
                    # inter_modal_cos_weighted_AB_exp = tf.exp(inter_modal_cos_weighted_AB)
                    # inter_modal_cos_weighted_BA_exp = tf.exp(inter_modal_cos_weighted_BA)
                    inter_modal_NIR_VIS_dist = tf.reduce_sum(inter_modal_cos_weighted_AB)+tf.reduce_sum(inter_modal_cos_weighted_BA)+tf.reduce_sum(inter_modal_cos_weighted_AA)+tf.reduce_sum(inter_modal_cos_weighted_BB)

                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_NIR_VIS_dist_sum = inter_modal_NIR_VIS_dist')
                        inter_modal_NIR_VIS_dist_sum = tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)
                    else:
                        inter_modal_NIR_VIS_dist_sum += tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)

            else:
                print('classIdx,classIdx2', classIdx, classIdx2)

        'Weighted intra_cor'
        intra_cor_sum=0
        for idx,val in enumerate(intra_var_subtract_square_reduce_list):
            print('intra_var_subtract_square_reduce_list',idx,val)
            intra_cor_sum+=tf.exp(intra_var_subtract_square_reduce_list[idx])
        for idx, val in enumerate(intra_var_subtract_square_reduce_list):
            W_C_i = tf.multiply(tf.exp(intra_var_subtract_square_reduce_list[idx]), 1 / intra_cor_sum)
            W_C_i_tensor = tf.get_variable('W_C_i_tensor_' + str(idx + 1) + '_' + str(classIdx2 + 1),
                                               [1], dtype=tf.float32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            W_C_i_tensor = tf.squeeze(tf.scatter_nd_update(W_C_i_tensor, [[0]], [W_C_i]))
            print('W_C_i', W_C_i)
            print('W_C_i_tensor', W_C_i_tensor)
            intra_cor_weighted=tf.multiply(W_C_i_tensor,intra_var_subtract_square_reduce_list[idx])
            # intra_cor_weighted_exp=tf.exp(intra_cor_weighted)
            if idx == 0:
                print('tf.multiply(intra_cor_weighted_exp, intra_class_var_val)')
                intra_cor = tf.multiply(intra_cor_weighted, intra_class_var_val)
            else:
                intra_cor += tf.multiply(intra_cor_weighted, intra_class_var_val)


        # 'Weighted inter_modal'
        # inter_modal_sum=0
        # for idx,val in enumerate(inter_modal_cos_list):
        #     print('inter_modal_cos_list',idx,val)
        #     inter_modal_sum +=tf.reduce_sum(tf.exp(inter_modal_cos_list[idx]))
        # for idx, val in enumerate(inter_modal_cos_list):
        #
        #     W_B_ij = tf.multiply(tf.exp(inter_modal_cos_list[idx]), 1 / inter_modal_sum)
        #     print('W_B_ij', W_B_ij)
        #
        #     inter_modal_cos_weighted = tf.multiply(W_B_ij, inter_modal_cos_list[idx])
        #     inter_modal_cos_weighted_exp = tf.exp(inter_modal_cos_weighted)
        #     inter_modal_NIR_VIS_dist = tf.reduce_sum(inter_modal_cos_weighted_exp)
        #
        #     if idx == 0:
        #         print('tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)')
        #         inter_modal_NIR_VIS_dist_sum = tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)
        #     else:
        #         inter_modal_NIR_VIS_dist_sum += tf.multiply(inter_modal_NIR_VIS_dist, inter_modal_NIS_VIS_val)


        print('intra_domain',intra_domain)
        print('intra_cor',intra_cor)
        print('inter_modal_NIR_VIS_dist_sum', inter_modal_NIR_VIS_dist_sum)

        loss_intra_domain = tf.multiply(Deta1, intra_domain)
        loss_intra_cor = tf.multiply(Deta2, intra_cor)
        loss_inter_modal_NIR_VIS_dist_sum = tf.multiply(Deta3, inter_modal_NIR_VIS_dist_sum)

        # loss = tf.log(1 + loss_intra_domain + loss_intra_cor + loss_inter_modal_NIR_VIS_dist_sum, name='WDR_total_loss')
        loss = tf.add_n([loss_intra_domain, loss_intra_cor, loss_inter_modal_NIR_VIS_dist_sum], name='WDR_total_loss')
    return loss,loss_intra_domain,loss_intra_cor,loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_NIR_VIS_dist_sum



'MI loss'
def modality_invariant_loss(embeddings,class_num,sample_class,embedding_size,alpa1=1.5,alpa2=1.5,Deta1=1.0,Deta2=1.0,Deta3=1.0,Deta4=1.0):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    print('==========modality_invariant_loss==========')
    with tf.variable_scope('MI_loss'):

        intra_class_center_val = tf.constant(1 / (class_num))
        intra_class_var_val = tf.constant(1.0)
        # inter_modal_NIS_VIS_val= tf.constant(1/(class_num*class_num))
        inter_modal_NIS_VIS_val = tf.constant(1 / (class_num))
        # inra-modal variations calculation
        for classIdx in range(0,class_num*sample_class, sample_class):

            embeddings_intra_modalityA = tf.slice(embeddings, [classIdx, 0], [int(sample_class/2), embedding_size])
            embeddings_intra_modalityB = tf.slice(embeddings, [classIdx+int(sample_class/2), 0], [int(sample_class/2), embedding_size])
            'mean_dist between H1 and H2'
            embeddings_intra_meanA = tf.reduce_mean(embeddings_intra_modalityA, axis=0)
            embeddings_intra_meanB = tf.reduce_mean(embeddings_intra_modalityB, axis=0)

            print('embeddings_intra_modalityA',embeddings_intra_modalityA)
            print('embeddings_intra_modalityB', embeddings_intra_modalityB)
            print('embeddings_intra_meanA', embeddings_intra_meanA)
            print('embeddings_intra_meanB', embeddings_intra_meanB)


            'intra-center'
            intra_modal_mean_single_subtract = tf.subtract(embeddings_intra_meanA, embeddings_intra_meanB)
            intra_modal_mean_single_subtract_square = tf.square(intra_modal_mean_single_subtract)
            intra_modal_mean_single_subtract_square_reduce = tf.reduce_sum(intra_modal_mean_single_subtract_square)

            if classIdx == 0:
                print('intra_center = tf.multiply(intra_modal_mean_single_subtract_square_reduce, intra_class_center_val)')
                intra_center = tf.multiply(intra_modal_mean_single_subtract_square_reduce, intra_class_center_val)
            else:
                intra_center += tf.multiply(intra_modal_mean_single_subtract_square_reduce, intra_class_center_val)

            'intra-var'
            'intra-var'
            H1 = tf.transpose(embeddings_intra_modalityA)
            H2 = tf.transpose(embeddings_intra_modalityB)

            m = tf.shape(H1)[1]

            H1bar = H1 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H1, tf.ones([m, m]))
            H2bar = H2 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H2, tf.ones([m, m]))

            SigmaHat11 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H1bar, H1bar, transpose_b=True)
            SigmaHat22 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H2bar, H2bar, transpose_b=True)

            var_11=tf.diag_part(SigmaHat11)
            var_22 = tf.diag_part(SigmaHat22)

            intra_var_subtract=tf.subtract(var_11, var_22)
            intra_var_subtract_square=tf.square(intra_var_subtract)
            intra_var_subtract_square_reduce=tf.reduce_sum(intra_var_subtract_square)
            if classIdx == 0:
                print('intra_var = tf.multiply(intra_var_subtract_square_reduce, intra_class_var_val)')
                intra_var = tf.multiply(intra_var_subtract_square_reduce, intra_class_var_val)
            else:
                intra_var += tf.multiply(intra_var_subtract_square_reduce, intra_class_var_val)

            if classIdx!=(class_num-1)*sample_class:
                for classIdx2 in range(classIdx+sample_class, class_num * sample_class, sample_class):
                    print('classIdx,classIdx2', classIdx, classIdx2)
                    embeddings_intra_modalityA_classNext = tf.slice(embeddings, [classIdx2, 0], [int(sample_class / 2), embedding_size])
                    embeddings_intra_modalityB_classNext = tf.slice(embeddings, [classIdx2 + int(sample_class / 2), 0], [int(sample_class / 2), embedding_size])

                    embeddings_intra_meanA_classNext, embeddings_intra_varA_classNext = tf.nn.moments(embeddings_intra_modalityA_classNext, axes=0)
                    embeddings_intra_meanB_classNext, embeddings_intra_varB_classNext = tf.nn.moments(embeddings_intra_modalityB_classNext, axes=0)

                    'inter-NIR-VIS'
                    inter_modal_NIR_VIS_mean_single_subtract_AB = tf.subtract(embeddings_intra_meanA, embeddings_intra_meanB_classNext)
                    inter_modal_NIR_VIS_mean_single_subtract_square_AB = tf.square(inter_modal_NIR_VIS_mean_single_subtract_AB)
                    inter_modal_NIR_VIS_mean_single_subtract_square_reduce_AB = tf.reduce_sum(inter_modal_NIR_VIS_mean_single_subtract_square_AB)
                    inter_modal_NIR_VIS_dist_AB = tf.add(tf.subtract(0.0, inter_modal_NIR_VIS_mean_single_subtract_square_reduce_AB), alpa1)
                    inter_modal_NIR_VIS_dist_AB = tf.maximum(inter_modal_NIR_VIS_dist_AB, 0.0)
                    inter_modal_NIR_VIS_dist_AB = tf.multiply(inter_modal_NIR_VIS_dist_AB, inter_modal_NIS_VIS_val)

                    inter_modal_NIR_VIS_mean_single_subtract_BA = tf.subtract(embeddings_intra_meanB, embeddings_intra_meanA_classNext)
                    inter_modal_NIR_VIS_mean_single_subtract_square_BA = tf.square(inter_modal_NIR_VIS_mean_single_subtract_BA)
                    inter_modal_NIR_VIS_mean_single_subtract_square_reduce_BA = tf.reduce_sum(inter_modal_NIR_VIS_mean_single_subtract_square_BA)
                    inter_modal_NIR_VIS_dist_BA = tf.add(tf.subtract(0.0, inter_modal_NIR_VIS_mean_single_subtract_square_reduce_BA), alpa1)
                    inter_modal_NIR_VIS_dist_BA = tf.maximum(inter_modal_NIR_VIS_dist_BA, 0.0)
                    inter_modal_NIR_VIS_dist_BA = tf.multiply(inter_modal_NIR_VIS_dist_BA, inter_modal_NIS_VIS_val)

                    inter_modal_NIR_VIS_dist=inter_modal_NIR_VIS_dist_AB+inter_modal_NIR_VIS_dist_BA
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_NIR_VIS_dist_sum = inter_modal_NIR_VIS_dist')
                        inter_modal_NIR_VIS_dist_sum = inter_modal_NIR_VIS_dist
                    else:
                        inter_modal_NIR_VIS_dist_sum += inter_modal_NIR_VIS_dist

                    'inter-NIR-NIR, inter-VIS-VIS'
                    inter_modal_NIR_NIR_mean_single_subtract_AA = tf.subtract(embeddings_intra_meanA, embeddings_intra_meanA_classNext)
                    inter_modal_SameModality_mean_single_subtract_square_AA = tf.square(inter_modal_NIR_NIR_mean_single_subtract_AA)
                    inter_modal_SameModality_mean_single_subtract_square_reduce_AA = tf.reduce_sum(inter_modal_SameModality_mean_single_subtract_square_AA)
                    inter_modal_SameModality_dist_AA = tf.add(tf.subtract(0.0, inter_modal_SameModality_mean_single_subtract_square_reduce_AA), alpa2)
                    inter_modal_SameModality_dist_AA = tf.maximum(inter_modal_SameModality_dist_AA, 0.0)
                    inter_modal_SameModality_dist_AA = tf.multiply(inter_modal_SameModality_dist_AA, inter_modal_NIS_VIS_val)


                    inter_modal_VIS_VIS_mean_single_subtract_BB = tf.subtract(embeddings_intra_meanB,embeddings_intra_meanB_classNext)
                    inter_modal_SameModality_mean_single_subtract_square_BB = tf.square(inter_modal_VIS_VIS_mean_single_subtract_BB)
                    inter_modal_SameModality_mean_single_subtract_square_reduce_BB = tf.reduce_sum(inter_modal_SameModality_mean_single_subtract_square_BB)
                    inter_modal_SameModality_dist_BB = tf.add(tf.subtract(0.0, inter_modal_SameModality_mean_single_subtract_square_reduce_BB), alpa2)
                    inter_modal_SameModality_dist_BB = tf.maximum(inter_modal_SameModality_dist_BB, 0.0)
                    inter_modal_SameModality_dist_BB = tf.multiply(inter_modal_SameModality_dist_BB, inter_modal_NIS_VIS_val)

                    inter_modal_SameModality_dist=inter_modal_SameModality_dist_AA+inter_modal_SameModality_dist_BB
                    if classIdx2 == classIdx+sample_class and classIdx == 0:
                        print('inter_modal_SameModality_sum = inter_modal_SameModality_dist')
                        inter_modal_SameModality_sum = inter_modal_SameModality_dist
                    else:
                        inter_modal_SameModality_sum += inter_modal_SameModality_dist
            else:
                print('classIdx,classIdx2', classIdx, classIdx2)


        print('intra_center',intra_center)
        print('intra_var',intra_var)
        print('inter_modal_NIR_VIS_dist_sum', inter_modal_NIR_VIS_dist_sum)
        print('inter_modal_SameModality_sum', inter_modal_SameModality_sum)
        loss_intra_center = tf.multiply(Deta1, intra_center)
        loss_intra_var = tf.multiply(Deta2, intra_var)
        loss_inter_modal_NIR_VIS_dist_sum = tf.multiply(Deta3, inter_modal_NIR_VIS_dist_sum)
        loss_inter_modal_SameModality_sum = tf.multiply(Deta4, inter_modal_SameModality_sum)
        loss = tf.add_n([loss_intra_center,loss_intra_var, loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum ], name='total_loss')

    return loss,loss_intra_center,loss_intra_var,loss_inter_modal_NIR_VIS_dist_sum,loss_inter_modal_SameModality_sum


'CdFD loss'
def Cross_domain_Factor_Detachment_loss(embeddings_ID,embeddings_VIS,embeddings_NIR,class_num,sample_class,embedding_size,alpa=0.1,Deta=1.0):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    print('==========Cross_domain_Factor_Detachment_loss==========')
    with tf.variable_scope('CdFD_loss'):

        class_val = tf.constant(1 / (class_num))

        # CdFD calculation
        numImg=class_num*sample_class

        embeddingsM_VIS_TV = tf.slice(embeddings_VIS, [0, 0], [int(numImg/2), embedding_size]) #TV=YVI+YV
        embeddingsM_NIR_TN = tf.slice(embeddings_NIR, [int(numImg/2), 0], [int(numImg/2), embedding_size])#TN=YNI+YN
        embeddingsID_VIS_UV = tf.slice(embeddings_ID, [0, 0], [int(numImg/2), embedding_size])#UV=YVI
        embeddingsID_NIR_UN = tf.slice(embeddings_ID, [int(numImg / 2), 0], [int(numImg/2), embedding_size])#UN=YNI
        print('embeddingsM_VIS_TV', embeddingsM_VIS_TV)
        print('embeddingsM_NIR_TN', embeddingsM_NIR_TN)
        print('embeddingsID_VIS_UV', embeddingsID_VIS_UV)
        print('embeddingsID_NIR_UN', embeddingsID_NIR_UN)
        UV_UN = tf.matmul(embeddingsID_VIS_UV, tf.transpose(embeddingsID_NIR_UN), transpose_b=False) #n samples and d dimensions, return n*n matrix
        TV_TN = tf.matmul(embeddingsM_VIS_TV, tf.transpose(embeddingsM_NIR_TN), transpose_b=False)
        UV_TN = tf.matmul(embeddingsID_VIS_UV, tf.transpose(embeddingsM_NIR_TN), transpose_b=False)
        UN_TV = tf.matmul(embeddingsID_NIR_UN, tf.transpose(embeddingsM_VIS_TV), transpose_b=False)

        print('UV_UN', UV_UN)
        print('TV_TN', TV_TN)
        print('UV_TN', UV_TN)
        print('UN_TV', UN_TV)

        YV_YN=UV_UN+TV_TN-UV_TN-UN_TV
        print('YV_YN', YV_YN)
        one_matrix=tf.constant(np.ones((int(numImg/2),int(numImg/2)),dtype=np.float32))
        print('one_matrix', one_matrix)
        alpa_one_matrix=alpa * one_matrix
        print('alpa_one_matrix', alpa_one_matrix)
        YV_YN_subtractOnes=YV_YN-alpa_one_matrix
        YV_YN_subtractOnes_norm = tf.norm(YV_YN_subtractOnes, ord='euclidean')

        YV_YN_subtractOnes_norm = tf.multiply(YV_YN_subtractOnes_norm, class_val)
        loss_CdFD = tf.multiply(YV_YN_subtractOnes_norm,Deta)

        print('YV_YN_subtractOnes', YV_YN_subtractOnes)
        print('YV_YN_subtractOnes_norm', YV_YN_subtractOnes_norm)

    return loss_CdFD,YV_YN,alpa_one_matrix

'CdFD loss'
def Cross_domain_Factor_Dual_Detachment_loss(embeddings_ID,embeddings_VIS,embeddings_NIR,class_num,sample_class,embedding_size,Deta1=1.0,Deta2=1.0):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    print('==========Cross_domain_Factor_Detachment_loss==========')
    with tf.variable_scope('CdFD_loss'):

        class_val = tf.constant(1 / (class_num))

        # CdFD calculation
        numImg=class_num*sample_class

        embeddingsM_VIS_TV = tf.slice(embeddings_VIS, [0, 0], [int(numImg/2), embedding_size]) #TV=YVI+YV
        embeddingsM_NIR_TN = tf.slice(embeddings_NIR, [int(numImg/2), 0], [int(numImg/2), embedding_size])#TN=YNI+YN
        embeddingsID_VIS_UV = tf.slice(embeddings_ID, [0, 0], [int(numImg/2), embedding_size])#UV=YVI
        embeddingsID_NIR_UN = tf.slice(embeddings_ID, [int(numImg / 2), 0], [int(numImg/2), embedding_size])#UN=YNI
        print('embeddingsM_VIS_TV', embeddingsM_VIS_TV)
        print('embeddingsM_NIR_TN', embeddingsM_NIR_TN)
        print('embeddingsID_VIS_UV', embeddingsID_VIS_UV)
        print('embeddingsID_NIR_UN', embeddingsID_NIR_UN)
        UV_UN = tf.matmul(embeddingsID_VIS_UV, tf.transpose(embeddingsID_NIR_UN), transpose_b=False) #n samples and d dimensions, return n*n matrix
        TV_TN = tf.matmul(embeddingsM_VIS_TV, tf.transpose(embeddingsM_NIR_TN), transpose_b=False)
        UV_TN = tf.matmul(embeddingsID_VIS_UV, tf.transpose(embeddingsM_NIR_TN), transpose_b=False)
        TV_UN = tf.matmul(embeddingsM_VIS_TV, tf.transpose(embeddingsID_NIR_UN), transpose_b=False)

        # UN_TV = tf.matmul(embeddingsID_NIR_UN, tf.transpose(embeddingsM_VIS_TV), transpose_b=False)

        print('UV_UN', UV_UN)
        print('TV_TN', TV_TN)
        print('UV_TN', UV_TN)
        print('TV_UN', TV_UN)

        YV_YN=UV_UN+TV_TN-UV_TN-TV_UN
        print('YV_YN', YV_YN)
        YV_YN_norm = tf.norm(YV_YN, ord='euclidean')


        YVI_YN__YV_YNI=UV_TN+TV_UN-2*UV_UN

        print('YVI_YN__YV_YNI', YVI_YN__YV_YNI)
        YVI_YN__YV_YNI_norm = tf.norm(YVI_YN__YV_YNI, ord='euclidean')

        YV_YN_subtractOnes_norm = tf.multiply(YV_YN_norm, class_val)
        loss_CdFD_YV_YN = tf.multiply(YV_YN_subtractOnes_norm,Deta1)

        YVI_YN__YV_YNI_norm = tf.multiply(YVI_YN__YV_YNI_norm, class_val)
        loss_CdFD_YVI_YN__YV_YNI = tf.multiply(YVI_YN__YV_YNI_norm,Deta2)

        print('YV_YN_norm', YV_YN_norm)
        loss_CdFD=loss_CdFD_YV_YN+loss_CdFD_YVI_YN__YV_YNI
    return loss_CdFD,loss_CdFD_YV_YN,loss_CdFD_YVI_YN__YV_YNI

'CdFD loss'
def Cross_domain_Factor_Dual_Detachment_simple_loss(embeddings_ID,embeddings_VIS,embeddings_NIR,class_num,sample_class,embedding_size,Deta=1.0):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    print('==========Cross_domain_Factor_Detachment_loss==========')
    with tf.variable_scope('CdFD_loss'):

        class_val = tf.constant(1 / (class_num))

        # CdFD calculation
        numImg=class_num*sample_class

        embeddingsM_VIS_TV = tf.slice(embeddings_VIS, [0, 0], [int(numImg/2), embedding_size]) #TV=YVI+YV
        embeddingsM_NIR_TN = tf.slice(embeddings_NIR, [int(numImg/2), 0], [int(numImg/2), embedding_size])#TN=YNI+YN
        embeddingsID_VIS_UV = tf.slice(embeddings_ID, [0, 0], [int(numImg/2), embedding_size])#UV=YVI
        embeddingsID_NIR_UN = tf.slice(embeddings_ID, [int(numImg / 2), 0], [int(numImg/2), embedding_size])#UN=YNI
        print('embeddingsM_VIS_TV', embeddingsM_VIS_TV)
        print('embeddingsM_NIR_TN', embeddingsM_NIR_TN)
        print('embeddingsID_VIS_UV', embeddingsID_VIS_UV)
        print('embeddingsID_NIR_UN', embeddingsID_NIR_UN)
        UV_UN = tf.matmul(embeddingsID_VIS_UV, tf.transpose(embeddingsID_NIR_UN), transpose_b=False) #n samples and d dimensions, return n*n matrix
        TV_TN = tf.matmul(embeddingsM_VIS_TV, tf.transpose(embeddingsM_NIR_TN), transpose_b=False)
        UV_TN = tf.matmul(embeddingsID_VIS_UV, tf.transpose(embeddingsM_NIR_TN), transpose_b=False)
        UN_TV = tf.matmul(embeddingsID_NIR_UN, tf.transpose(embeddingsM_VIS_TV), transpose_b=False)

        print('UV_UN', UV_UN)
        print('TV_TN', TV_TN)
        print('UV_TN', UV_TN)
        print('UN_TV', UN_TV)


        YV_YN=UV_UN+TV_TN-UV_TN-UN_TV

        print('YV_YN', YV_YN)
        YV_YN_norm = tf.norm(YV_YN, ord='euclidean')

        YVI_YN__YNI_YV=UV_TN+UN_TV-2*UV_UN

        print('YVI_YN__YNI_YV', YVI_YN__YNI_YV)
        YVI_YN__YNI_YV_norm = tf.norm(YVI_YN__YNI_YV, ord='euclidean')

        YV_YN_YVI_YN_YNI_YV=TV_TN-UV_UN

        print('YV_YN_YVI_YN_YNI_YV', YV_YN_YVI_YN_YNI_YV)
        YV_YN_YVI_YN_YNI_YV_norm = tf.norm(YV_YN_YVI_YN_YNI_YV, ord='euclidean')

        YV_YN_subtractOnes_norm = tf.multiply(YV_YN_norm, class_val)
        loss_CdFD_YV_YN = tf.multiply(YV_YN_subtractOnes_norm, Deta)

        YVI_YN__YNI_YV_norm = tf.multiply(YVI_YN__YNI_YV_norm, class_val)
        loss_CdFD_YVI_YN__YNI_YV = tf.multiply(YVI_YN__YNI_YV_norm, Deta)

        YV_YN_subtractOnes_norm = tf.multiply(YV_YN_YVI_YN_YNI_YV_norm, class_val)
        loss_CdFD = tf.multiply(YV_YN_subtractOnes_norm,Deta)


    return loss_CdFD,loss_CdFD_YV_YN,loss_CdFD_YVI_YN__YNI_YV


#TODO 20200322 DCCA
def inner_cca_objective(H1, H2, use_all_singular_values=True, outdim_size=128):
    """
    It is the loss function of CCA as introduced in the original paper. There can be other formulations.
    It is implemented on Tensorflow based on github@VahidooX's cca loss on Theano.
    y_true is just ignored
    """

    r1 = 1e-10
    r2 = 1e-10

    # r1 = 1e-4
    # r2 = 1e-4
    eps = 1e-12
    o1 = o2 = int(H1.shape[1])

    # unpack (separate) the output of networks for view 1 and view 2
    H1 = tf.transpose(H1)
    H2 = tf.transpose(H2)

    m = tf.shape(H1)[1]

    H1bar = H1 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H1, tf.ones([m, m]))
    H2bar = H2 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H2, tf.ones([m, m]))

    SigmaHat12 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H1bar, H2bar, transpose_b=True)  # [dim, dim]
    SigmaHat11 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H1bar, H1bar, transpose_b=True) + r1 * tf.eye(o1)
    SigmaHat22 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H2bar, H2bar, transpose_b=True) + r2 * tf.eye(o2)

    # Calculating the root inverse of covariance matrices by using eigen decomposition
    [D1, V1] = tf.self_adjoint_eig(SigmaHat11)
    [D2, V2] = tf.self_adjoint_eig(SigmaHat22)  # Added to increase stability

    posInd1 = tf.where(tf.greater(D1, eps))
    D1 = tf.gather_nd(D1, posInd1)  # get eigen values that are larger than eps
    V1 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(V1), tf.squeeze(posInd1)))

    posInd2 = tf.where(tf.greater(D2, eps))
    D2 = tf.gather_nd(D2, posInd2)
    V2 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(V2), tf.squeeze(posInd2)))

    SigmaHat11RootInv = tf.matmul(tf.matmul(V1, tf.diag(D1 ** -0.5)), V1, transpose_b=True)  # [dim, dim]
    SigmaHat22RootInv = tf.matmul(tf.matmul(V2, tf.diag(D2 ** -0.5)), V2, transpose_b=True)

    Tval = tf.matmul(tf.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    if use_all_singular_values:
        corr = tf.sqrt(tf.trace(tf.matmul(Tval, Tval, transpose_a=True)))
        print('use_all_singular_values',use_all_singular_values)
    else:
        [U, V] = tf.self_adjoint_eig(tf.matmul(Tval, Tval, transpose_a=True))
        U = tf.gather_nd(U, tf.where(tf.greater(U, eps)))
        kk = tf.reshape(tf.cast(tf.shape(U), tf.int32), [])
        print('kk',kk,'outdim_size',outdim_size)
        K = tf.minimum(kk, outdim_size)
        w, _ = tf.nn.top_k(U, k=K)
        corr = tf.reduce_sum(tf.sqrt(w))
        print('use_all_singular_values', use_all_singular_values,'outdim_size',outdim_size)
    return corr

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def get_image_paths_and_labels_FFA(dataset):
    image_paths_flat = []
    labels_flat = []
    labels_flat_ori=[]
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
        labels_flat_ori+=[dataset[i].name]
    return image_paths_flat, labels_flat, labels_flat_ori

def shuffle_examples(image_paths, labels):
    shuffle_list = list(zip(image_paths, labels))
    random.shuffle(shuffle_list)
    image_paths_shuff, labels_shuff = zip(*shuffle_list)
    return image_paths_shuff, labels_shuff

def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')
  
# 1: Random rotate 2: Random crop  4: Random flip  8:  Fixed image standardization  16: Flip
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16
def create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder):
    images_and_labels_list = []
    for _ in range(nrof_preprocess_threads):
        filenames, label, control = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, 3)
            image = tf.cond(get_control_flag(control[0], RANDOM_ROTATE),
                            lambda:tf.py_func(random_rotate_image, [image], tf.uint8), 
                            lambda:tf.identity(image))
            image = tf.cond(get_control_flag(control[0], RANDOM_CROP), 
                            lambda:tf.random_crop(image, image_size + (3,)), 
                            lambda:tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1]))
            image = tf.cond(get_control_flag(control[0], RANDOM_FLIP),
                            lambda:tf.image.random_flip_left_right(image),
                            lambda:tf.identity(image))
            image = tf.cond(get_control_flag(control[0], FIXED_STANDARDIZATION),
                            lambda:(tf.cast(image, tf.float32) - 127.5)/128.0,
                            lambda:tf.image.per_image_standardization(image))
            image = tf.cond(get_control_flag(control[0], FLIP),
                            lambda:tf.image.flip_left_right(image),
                            lambda:tf.identity(image))
            #pylint: disable=no-member
            image.set_shape(image_size + (3,))
            images.append(image)
        images_and_labels_list.append([images, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels_list, batch_size=batch_size_placeholder, 
        shapes=[image_size + (3,), ()], enqueue_many=True,
        capacity=4 * nrof_preprocess_threads * 100,
        allow_smaller_final_batch=True)
    
    return image_batch, label_batch

def get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)
  
def _add_loss_summaries(total_loss):
    """Add summaries for losses.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
  
    return loss_averages_op

def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
    
        grads = opt.compute_gradients(total_loss, update_gradient_vars)
        
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
   
    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
  
    return train_op


def train_part_para(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars,
          log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(update_gradient_vars)

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image
  
def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
  
def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images

def get_label_batch(label_data, batch_size, batch_index):
    nrof_examples = np.size(label_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = label_data[j:j+batch_size]
    else:
        x1 = label_data[j:nrof_examples]
        x2 = label_data[0:nrof_examples-j]
        batch = np.vstack([x1,x2])
    batch_int = batch.astype(np.int64)
    return batch_int

def get_batch(image_data, batch_size, batch_index):
    nrof_examples = np.size(image_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = image_data[j:j+batch_size,:,:,:]
    else:
        x1 = image_data[j:nrof_examples,:,:,:]
        x2 = image_data[0:nrof_examples-j,:,:,:]
        batch = np.vstack([x1,x2])
    batch_float = batch.astype(np.float32)
    return batch_float

def get_triplet_batch(triplets, batch_index, batch_size):
    ax, px, nx = triplets
    a = get_batch(ax, int(batch_size/3), batch_index)
    p = get_batch(px, int(batch_size/3), batch_index)
    n = get_batch(nx, int(batch_size/3), batch_index)
    batch = np.vstack([a, p, n])
    return batch

def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                if par[1]=='-':
                    lr = -1
                else:
                    lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
  
def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths
  
def split_dataset(dataset, split_ratio, min_nrof_images_per_class, mode):
    if mode=='SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*(1-split_ratio)))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode=='SPLIT_IMAGES':
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            nrof_images_in_class = len(paths)
            split = int(math.floor(nrof_images_in_class*(1-split_ratio)))
            if split==nrof_images_in_class:
                split = nrof_images_in_class-1
            if split>=min_nrof_images_per_class and nrof_images_in_class-split>=1:
                train_set.append(ImageClass(cls.name, paths[:split]))
                test_set.append(ImageClass(cls.name, paths[split:]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set

def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def load_model_collection(model,fix_variables):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)


        # saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))

        saver = tf.train.Saver(fix_variables)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file
  
def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


  
def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
    
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
  
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def store_revision_info(src_path, output_dir, arg_string):
    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' +  e.strerror
  
    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' +  e.strerror
    
    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('tensorflow version: %s\n--------------------\n' % tf.__version__)  # @UndefinedVariable
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)

def list_variables(filename):
    reader = training.NewCheckpointReader(filename)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    return names

def put_images_on_grid(images, shape=(16,8)):
    nrof_images = images.shape[0]
    img_size = images.shape[1]
    bw = 3
    img = np.zeros((shape[1]*(img_size+bw)+bw, shape[0]*(img_size+bw)+bw, 3), np.float32)
    for i in range(shape[1]):
        x_start = i*(img_size+bw)+bw
        for j in range(shape[0]):
            img_index = i*shape[0]+j
            if img_index>=nrof_images:
                break
            y_start = j*(img_size+bw)+bw
            img[x_start:x_start+img_size, y_start:y_start+img_size, :] = images[img_index, :, :, :]
        if img_index>=nrof_images:
            break
    return img

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))
