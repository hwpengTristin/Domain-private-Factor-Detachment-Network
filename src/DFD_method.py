"""Training a face recognizer with TensorFlow using softmax cross entropy loss
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import lfw
import h5py
import math
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import NewPaper_validate_on_CASIA_NIR_VIS_2_0_Rank_1_speedUp as CASIA_Rank1
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
def main(args):

    network = importlib.import_module(args.model_def)
    image_size = (args.image_size, args.image_size)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    # TODO save result log
    resultLog=os.path.join(model_dir, 'resultLog.txt')
    sys.stdout=Logger(resultLog)
    for key in vars(args):
        print(key,args.__getattribute__(key))
    #TODO save best acc
    best_model_dir = os.path.join(os.path.expanduser(args.best_models_base_dir), subdir)
    if not os.path.isdir(best_model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(best_model_dir)

    stat_file_name = os.path.join(log_dir, 'stat.h5')

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    dataset = facenet.get_dataset(args.data_dir)
    if args.filter_filename:
        dataset = filter_dataset(dataset, os.path.expanduser(args.filter_filename),
            args.filter_percentile, args.filter_min_nrof_images_per_class)

    #TODO 20200318 start
    train_set_NIR = facenet.get_dataset(args.data_dir_NIR)
    train_set_VIS = facenet.get_dataset(args.data_dir_VIS)
    # TODO 20200318 end

    if args.validation_set_split_ratio>0.0:
        train_set, val_set = facenet.split_dataset(dataset, args.validation_set_split_ratio, args.min_nrof_val_images_per_class, 'SPLIT_IMAGES')
    else:
        train_set, val_set = dataset, []

    nrof_classes = len(train_set)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)

    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Get a list of image paths and their labels
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        assert len(image_list)>0, 'The training set should not be empty'

        # TODO 20200318 start
        image_list_NIR, label_list_NIR = facenet.get_image_paths_and_labels(train_set_NIR)
        assert len(image_list_NIR)>0, 'The training set should not be empty'
        # Create a queue that produces indices into the image_list and label_list
        labels_NIR = ops.convert_to_tensor(label_list_NIR, dtype=tf.int32)
        range_size_NIR = array_ops.shape(labels_NIR)[0]
        index_queue_NIR = tf.train.range_input_producer(range_size_NIR, num_epochs=None,
                                                    shuffle=True, seed=None, capacity=32)
        index_dequeue_op_NIR = index_queue_NIR.dequeue_many(args.batch_size * args.iteration_NIR, 'index_dequeue_NIR')


        image_list_VIS, label_list_VIS = facenet.get_image_paths_and_labels(train_set_VIS)
        assert len(image_list_VIS)>0, 'The training set should not be empty'
        # Create a queue that produces indices into the image_list and label_list
        labels_VIS = ops.convert_to_tensor(label_list_VIS, dtype=tf.int32)
        range_size_VIS = array_ops.shape(labels_VIS)[0]
        index_queue_VIS = tf.train.range_input_producer(range_size_VIS, num_epochs=None,
                                                    shuffle=True, seed=None, capacity=32)
        index_dequeue_op_VIS = index_queue_VIS.dequeue_many(args.batch_size * args.iteration_VIS, 'index_dequeue_VIS')
        # TODO 20200318 end



        val_image_list, val_label_list = facenet.get_image_paths_and_labels(val_set)

        # Create a queue that produces indices into the image_list and label_list
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32)

        index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.iteration_ID, 'index_dequeue')

        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
        control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')

        nrof_preprocess_threads = 4
        input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                    dtypes=[tf.string, tf.int32, tf.int32],
                                    shapes=[(1,), (1,), (1,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='enqueue_op')
        image_batch, label_batch = facenet.create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)

        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')

        print('Number of classes in training set: %d' % nrof_classes)
        print('Number of examples in training set: %d' % len(image_list))

        print('Number of classes in validation set: %d' % len(val_set))
        print('Number of examples in validation set: %d' % len(val_image_list))

        print('Building training graph')

        # Build the inference graphnetwork
        prelogits, prelogits_NIR, prelogits_VIS = network.inference(image_batch, args.keep_probability,
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
            weight_decay=args.weight_decay)
        embeddings_NIR = tf.nn.l2_normalize(prelogits_NIR, 1, 1e-10, name='embeddings_NIR')
        embeddings_VIS = tf.nn.l2_normalize(prelogits_VIS, 1, 1e-10, name='embeddings_VIS')
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings_ID')

        'CdR loss'
        alpa1, alpa2, Deta1, Deta2, Deta3, Deta4 = 1.2, 1.2, 0.02, 0.005, 0.02, 0.02
        print('alpa1,alpa2,Deta1,Deta2,Deta3,Deta4',alpa1,alpa2,Deta1,Deta2,Deta3,Deta4)
        # TODO CdR loss start
        CdR_loss,loss_intra_domain,loss_intra_cor,loss_inter_modal_NIR_VIS,loss_inter_SameModality = facenet.DFD_CdR_alignmentCenter_loss(embeddings, args.people_per_batch_CdR, args.images_per_person_CdR, args.embedding_size,alpa1,alpa2,Deta1,Deta2,Deta3,Deta4)


        with tf.variable_scope('resnet50_CBAM/ID_FC_layer_Para'):
            logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,
                    weights_initializer=slim.initializers.xavier_initializer(),
                    weights_regularizer=slim.l2_regularizer(args.weight_decay),
                    scope='Logits', reuse=False)


            # Norm for the prelogits
            eps = 1e-4
            prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits)+eps, ord=args.prelogits_norm_p, axis=1))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.prelogits_norm_loss_factor)

            # Add center loss
            prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)


        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')


        # TODO 20200318 start
        with tf.variable_scope('resnet50_CBAM/Modality_structure'):
            with tf.variable_scope('NIR_layer_Para'):
                with tf.variable_scope('NIR_layer_Para_FC'):
                    # TODO 20200318 NIR
                    logits_NIR = slim.fully_connected(prelogits_NIR, len(train_set_NIR), activation_fn=None,
                            weights_initializer=slim.initializers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(args.weight_decay),
                            scope='Logits_NIR', reuse=False)

                    # Norm for the prelogits
                    eps = 1e-4
                    prelogits_norm_NIR = tf.reduce_mean(tf.norm(tf.abs(prelogits_NIR) + eps, ord=args.prelogits_norm_p, axis=1))
                    # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm_NIR * args.prelogits_norm_loss_factor)

                    # Add center loss
                    prelogits_center_loss_NIR, _ = facenet.center_loss(prelogits_NIR, label_batch, args.center_loss_alfa, nrof_classes)
                    # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss_NIR * args.center_loss_factor)

            # Calculate the average cross entropy loss across the batch
            cross_entropy_NIR = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label_batch, logits=logits_NIR, name='cross_entropy_per_example_NIR')
            cross_entropy_mean_NIR = tf.reduce_mean(cross_entropy_NIR, name='cross_entropy_NIR')
            tf.add_to_collection('losses_NIR', cross_entropy_mean_NIR)

            correct_prediction_NIR = tf.cast(tf.equal(tf.argmax(logits_NIR, 1), tf.cast(label_batch, tf.int64)), tf.float32)
            accuracy_NIR = tf.reduce_mean(correct_prediction_NIR)

            with tf.variable_scope('VIS_layer_Para'):
                with tf.variable_scope('VIS_layer_Para_FC'):
                # TODO 20200318 VIS
                    logits_VIS = slim.fully_connected(prelogits_VIS, len(train_set_VIS), activation_fn=None,
                            weights_initializer=slim.initializers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(args.weight_decay),
                            scope='Logits_VIS', reuse=False)

                    # Norm for the prelogits
                    eps = 1e-4
                    prelogits_norm_VIS = tf.reduce_mean(tf.norm(tf.abs(prelogits_VIS) + eps, ord=args.prelogits_norm_p, axis=1))
                    # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm_VIS * args.prelogits_norm_loss_factor)

                    # Add center loss
                    prelogits_center_loss_VIS, _ = facenet.center_loss(prelogits_VIS, label_batch, args.center_loss_alfa, nrof_classes)
                    # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss_VIS * args.center_loss_factor)

            # Calculate the average cross entropy loss across the batch
            cross_entropy_VIS = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label_batch, logits=logits_VIS, name='cross_entropy_per_example_VIS')
            cross_entropy_mean_VIS = tf.reduce_mean(cross_entropy_VIS, name='cross_entropy_VIS')
            tf.add_to_collection('losses_VIS', cross_entropy_mean_VIS)

            correct_prediction_VIS = tf.cast(tf.equal(tf.argmax(logits_VIS, 1), tf.cast(label_batch, tf.int64)), tf.float32)
            accuracy_VIS = tf.reduce_mean(correct_prediction_VIS)

            # Calculate the total losses
            regularization_losses_NIR_VIS = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss_NIR = tf.add_n([cross_entropy_mean_NIR] , name='total_loss_NIR')
            total_loss_VIS = tf.add_n([cross_entropy_mean_VIS] , name='total_loss_VIS')


            with tf.variable_scope('NIR_VIS_domain_classfier_Para'):
                with tf.variable_scope('NIR_VIS_domain_Para_FC'):
                    # TODO 20201103 domain
                    logits_domain_NIR = slim.fully_connected(prelogits_NIR, 2, activation_fn=None,
                            weights_initializer=slim.initializers.xavier_initializer(),
                            weights_regularizer=None,
                            scope='Logits_domain', reuse=False)

                    logits_domain_VIS = slim.fully_connected(prelogits_VIS, 2, activation_fn=None,
                            weights_initializer=slim.initializers.xavier_initializer(),
                            weights_regularizer=None,
                            scope='Logits_domain', reuse=True)

                    # with tf.variable_scope('VIS_domain_Para_FC_center'):
                    #     # Add center loss
                    #     prelogits_center_loss_domain_VIS, _ = facenet.center_loss(prelogits_VIS, label_batch, args.center_loss_alfa, 2)
                    #     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss_domain_VIS * args.center_loss_factor)
                    # with tf.variable_scope('NIR_domain_Para_FC_center'):
                    #     # Add center loss
                    #     prelogits_center_loss_domain_NIR, _ = facenet.center_loss(prelogits_NIR, label_batch, args.center_loss_alfa, 2)
                    #     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss_domain_NIR * args.center_loss_factor)

            # Calculate the average cross entropy loss across the batch
            cross_entropy_domain_NIR = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label_batch, logits=logits_domain_NIR, name='cross_entropy_per_example_domain_NIR')
            cross_entropy_mean_domain_NIR = tf.reduce_mean(cross_entropy_domain_NIR, name='cross_entropy_domain_NIR')
            # tf.add_to_collection('losses_domain_NIR', cross_entropy_mean_domain_NIR)
            correct_prediction_domain_NIR = tf.cast(tf.equal(tf.argmax(logits_domain_NIR, 1), tf.cast(label_batch, tf.int64)), tf.float32)
            accuracy_domain_NIR = tf.reduce_mean(correct_prediction_domain_NIR)


            # Calculate the average cross entropy loss across the batch
            cross_entropy_domain_VIS = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label_batch, logits=logits_domain_VIS, name='cross_entropy_per_example_domain_VIS')
            cross_entropy_mean_domain_VIS = tf.reduce_mean(cross_entropy_domain_VIS, name='cross_entropy_domain_VIS')
            # tf.add_to_collection('losses_domain_VIS', cross_entropy_mean_domain_VIS)
            correct_prediction_domain_VIS = tf.cast(tf.equal(tf.argmax(logits_domain_VIS, 1), tf.cast(label_batch, tf.int64)), tf.float32)
            accuracy_domain_VIS = tf.reduce_mean(correct_prediction_domain_VIS)


            # Calculate the total losses
            # regularization_losses_domain_NIR_VIS = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss_domain_NIR = tf.add_n([cross_entropy_mean_domain_NIR] , name='total_loss_domain_NIR')
            total_loss_domain_VIS = tf.add_n([cross_entropy_mean_domain_VIS] , name='total_loss_domain_VIS')


        # TODO 20200318 end

        # TODO 20201029 CdFD loss
        Deta1 , Deta2 = 0.1,0.02
        'CdFD loss domain para'
        loss_CdFD_domain,loss_CdFD_YV_YN,loss_CdFD_YVI_YN__YNI_YV=facenet.Cross_domain_Factor_Dual_Detachment_loss(embeddings, embeddings_VIS, embeddings_NIR, args.people_per_batch_CdFD, args.images_per_person_CdFD, args.embedding_size, Deta1=Deta1, Deta2=Deta2)

        'Optimizer var'
        listVar=[]
        all_varibales = tf.trainable_variables()

        print('----------------------------all_varibales-------------------------------------')
        print('-----------------------------------------------------------------')
        for index, var in enumerate(all_varibales):
            print(index,var)

        print('---------------------------Domain_faltten_to_softmax_Para--------------------------------------')
        print('-----------------------------------------------------------------')
        Domain_faltten_to_softmax_Para=[]
        for index, var in enumerate(all_varibales):
            result=var.name.find('Modality_structure')
            if result > 0:
                Domain_faltten_to_softmax_Para.append(var)
                print(index,var)

        #ID_Varaible_all#
        print('---------------------------ID_variables_centerloss--------------------------------------')
        print('-----------------------------------------------------------------')
        ID_variables_centerloss = list(set(all_varibales) - set(Domain_faltten_to_softmax_Para))
        for index, var in enumerate(ID_variables_centerloss):
            print(var)
        #ID_Varaible_FC#
        print('---------------------------ID_embedding_to_softmax_layer_Para--------------------------------------')
        print('-----------------------------------------------------------------')
        ID_embedding_to_softmax_layer_Para=[]
        for index, var in enumerate(all_varibales):
            result=var.name.find('ID_FC_layer_Para')
            if result > 0:
                ID_embedding_to_softmax_layer_Para.append(var)
                print(index,var)

        fix_variables=list(set(all_varibales)-set(Domain_faltten_to_softmax_Para)-set(ID_embedding_to_softmax_layer_Para))
        print('-----------------------------fix_variables------------------------------------')
        print('-----------------------------------------------------------------')
        for index, var in enumerate(fix_variables):
            print(var)

        print('-----------------------------ID_flatten_to_embeddings_para------------------------------------')
        print('-----------------------------------------------------------------')
        ID_flatten_to_embeddings_para=[]
        for index, var in enumerate(all_varibales):
            if index >= 340 and index <= 343:
                ID_flatten_to_embeddings_para.append(var)
                print(index,var)


        # ID_Centerloss_allpara
        print('-----------------------------ID_Centerloss_allpara------------------------------------')
        print('-----------------------------------------------------------------')
        ID_Centerloss_allpara=list(set(all_varibales)-set(Domain_faltten_to_softmax_Para))
        for index, var in enumerate(ID_Centerloss_allpara):
            print(index,var)


        #NIR_embedding_to_softmax_Para#
        print('---------------------------NIR_embedding_to_softmax_Para--------------------------------------')
        print('-----------------------------------------------------------------')
        NIR_embedding_to_softmax_Para=[]
        for index, var in enumerate(all_varibales):
            result=var.name.find('NIR_layer_Para_FC')
            if result > 0:
                NIR_embedding_to_softmax_Para.append(var)
                print(index,var)

        #VIS_embedding_to_softmax_Para#
        print('---------------------------VIS_embedding_to_softmax_Para--------------------------------------')
        print('-----------------------------------------------------------------')
        VIS_embedding_to_softmax_Para=[]
        for index, var in enumerate(all_varibales):
            result=var.name.find('VIS_layer_Para_FC')
            if result > 0:
                VIS_embedding_to_softmax_Para.append(var)
                print(index,var)

        # CdfD_loss_train_backbone_para
        print('-----------------------------CdfDloss_train_backbone_para------------------------------------')
        print('-----------------------------------------------------------------')
        CdfD_loss_train_backbone_para=list(set(fix_variables)-set(ID_flatten_to_embeddings_para))
        for index, var in enumerate(CdfD_loss_train_backbone_para):
            print(index,var)

        # CdfD_loss_train_domain_para
        print('-----------------------------CdfD_loss_train_domain_para------------------------------------')
        print('-----------------------------------------------------------------')
        CdfD_loss_train_domain_para=Domain_faltten_to_softmax_Para
        for index, var in enumerate(CdfD_loss_train_domain_para):
            print(index,var)

        #NIR_Varaible_FC#
        print('---------------------------NIR_flatten_to_softmax_Para--------------------------------------')
        print('-----------------------------------------------------------------')
        NIR_flatten_to_softmax_Para=[]
        for index, var in enumerate(all_varibales):
            result=var.name.find('NIR_layer_Para')
            if result > 0:
                NIR_flatten_to_softmax_Para.append(var)
                print(index,var)

        print('---------------------------VIS_flatten_to_softmax_Para--------------------------------------')
        print('-----------------------------------------------------------------')
        VIS_flatten_to_softmax_Para=[]
        for index, var in enumerate(all_varibales):
            result=var.name.find('VIS_layer_Para')
            if result > 0:
                VIS_flatten_to_softmax_Para.append(var)
                print(index,var)

        print('---------------------------VIS_Backbone_to_flatten_to_softmax_Para--------------------------------------')
        print('-----------------------------------------------------------------')
        VIS_Backbone_to_flatten_to_softmax_Para=list(set(CdfD_loss_train_backbone_para) | set(VIS_flatten_to_softmax_Para))
        for index, var in enumerate(VIS_Backbone_to_flatten_to_softmax_Para):
            print(index,var)

        print('---------------------------NIR_Backbone_to_flatten_to_softmax_Para--------------------------------------')
        print('-----------------------------------------------------------------')
        NIR_Backbone_to_flatten_to_softmax_Para=list(set(CdfD_loss_train_backbone_para) | set(NIR_flatten_to_softmax_Para))
        for index, var in enumerate(NIR_Backbone_to_flatten_to_softmax_Para):
            print(index,var)

        # NIR_VIS_embedding_to_domain_classfier_Para#
        print(
            '---------------------------NIR_VIS_embedding_to_domain_classfier_Para--------------------------------------')
        print('-----------------------------------------------------------------')
        NIR_VIS_embedding_to_domain_classfier_Para = []
        for index, var in enumerate(all_varibales):
            result = var.name.find('NIR_VIS_domain_classfier_Para')
            if result > 0:
                NIR_VIS_embedding_to_domain_classfier_Para.append(var)
                print(index, var)

        # NIR_flatten_embeddings_Para#
        print('---------------------------NIR_flatten_embeddings_Para--------------------------------------')
        print('-----------------------------------------------------------------')
        NIR_flatten_embeddings_Para = list(set(NIR_flatten_to_softmax_Para)-set(NIR_embedding_to_softmax_Para))
        for index, var in enumerate(NIR_flatten_embeddings_Para):
            print(index,var)

        # VIS_flatten_embeddings_Para#
        print('---------------------------VIS_flatten_embeddings_Para--------------------------------------')
        print('-----------------------------------------------------------------')
        VIS_flatten_embeddings_Para = list(set(VIS_flatten_to_softmax_Para)-set(VIS_embedding_to_softmax_Para))
        for index, var in enumerate(VIS_flatten_embeddings_Para):
            print(index,var)

        # NIR_flatten_to_domain_classfier_Para#
        print('---------------------------NIR_flatten_to_domain_classfier_Para--------------------------------------')
        print('-----------------------------------------------------------------')
        NIR_flatten_to_domain_classfier_Para = list(set(NIR_flatten_embeddings_Para) | set(NIR_VIS_embedding_to_domain_classfier_Para))
        for index, var in enumerate(NIR_flatten_to_domain_classfier_Para):
            print(index,var)

        # NIR_flatten_to_domain_classfier_Para#
        print('---------------------------VIS_flatten_to_domain_classfier_Para--------------------------------------')
        print('-----------------------------------------------------------------')
        VIS_flatten_to_domain_classfier_Para = list(set(VIS_flatten_embeddings_Para) | set(NIR_VIS_embedding_to_domain_classfier_Para))
        for index, var in enumerate(VIS_flatten_to_domain_classfier_Para):
            print(index,var)

        # ID_Varaible #
        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, args.optimizer,
            learning_rate, args.moving_average_decay, ID_Centerloss_allpara, args.log_histograms)

        train_op_FC = facenet.train(total_loss, global_step, args.optimizer,
            learning_rate, args.moving_average_decay, ID_embedding_to_softmax_layer_Para, args.log_histograms)


        # NIR_Varaible#
        # Build a Graph that trains the model with one batch of examples and updates the model parameters

        train_op_FC_NIR = facenet.train(total_loss_NIR, global_step, args.optimizer,
            learning_rate, args.moving_average_decay, NIR_flatten_to_softmax_Para, args.log_histograms)

        train_op_allPara_NIR = facenet.train(total_loss_NIR, global_step, args.optimizer,
            learning_rate, args.moving_average_decay, NIR_Backbone_to_flatten_to_softmax_Para, args.log_histograms)

        # VIS_Varaible#
        # Build a Graph that trains the model with one batch of examples and updates the model parameters

        train_op_FC_VIS = facenet.train(total_loss_VIS, global_step, args.optimizer,
                                        learning_rate, args.moving_average_decay, VIS_flatten_to_softmax_Para,
                                        args.log_histograms)

        train_op_allPara_VIS = facenet.train(total_loss_VIS, global_step, args.optimizer,
                                        learning_rate, args.moving_average_decay, VIS_Backbone_to_flatten_to_softmax_Para,
                                        args.log_histograms)


        train_op_FC_domain_CdFD = facenet.train(loss_CdFD_domain, global_step, args.optimizer,
                                        learning_rate, args.moving_average_decay, Domain_faltten_to_softmax_Para,
                                        args.log_histograms)

        train_op_Backbone_CdFD = facenet.train(loss_CdFD_domain, global_step, args.optimizer,
                                        learning_rate, args.moving_average_decay, CdfD_loss_train_backbone_para,
                                        args.log_histograms)

        train_op_domain_classifier_VIS = facenet.train(total_loss_domain_VIS, global_step, args.optimizer,
                                        learning_rate, args.moving_average_decay, VIS_flatten_to_domain_classfier_Para,
                                        args.log_histograms)

        train_op_domain_classifier_NIR = facenet.train(total_loss_domain_NIR, global_step, args.optimizer,
                                        learning_rate, args.moving_average_decay, NIR_flatten_to_domain_classfier_Para,
                                        args.log_histograms)


        train_op_domain_classifier_FC_VIS = facenet.train(total_loss_domain_VIS, global_step, args.optimizer,
                                        learning_rate, args.moving_average_decay, NIR_VIS_embedding_to_domain_classfier_Para,
                                        args.log_histograms)

        train_op_domain_classifier_FC_NIR = facenet.train(total_loss_domain_NIR, global_step, args.optimizer,
                                        learning_rate, args.moving_average_decay, NIR_VIS_embedding_to_domain_classfier_Para,
                                        args.log_histograms)

        train_op_CdRloss= facenet.train(CdR_loss, global_step, args.optimizer, learning_rate, args.moving_average_decay, fix_variables,
                                        args.log_histograms)
        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        best_saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=20)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        myepoch = -1
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        print('Running training')
        CASIA_JB_acc = []
        CASIA_Cosine_test = []
        LFW_acc = []
        LFW_val = []
        firstSave = True
        bestVal3 = 0
        bestVal_LFW = 0
        with sess.as_default():

            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                # saver.restore(sess, pretrained_model)
                facenet.load_model_collection(args.pretrained_model, fix_variables)

            #TODO 2FC_weight init  begin
            NIR_FC_weight = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/NIR_layer_Para/NIR_Bottleneck/weights:0")
            VIS_FC_weight = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/VIS_layer_Para/VIS_Bottleneck/weights:0")
            ID_FC_weight = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Logits/Bottleneck/weights:0")
            NIR_FC_weight_=tf.assign(NIR_FC_weight,ID_FC_weight)
            VIS_FC_weight_ = tf.assign(VIS_FC_weight, ID_FC_weight)
            sess.run([NIR_FC_weight_,VIS_FC_weight_])


            NIR_FC_weight = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/NIR_layer_Para/NIR_Bottleneck/BatchNorm/beta:0")
            VIS_FC_weight = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/VIS_layer_Para/VIS_Bottleneck/BatchNorm/beta:0")
            ID_FC_weight = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Logits/Bottleneck/BatchNorm/beta:0")
            NIR_FC_weight_=tf.assign(NIR_FC_weight,ID_FC_weight)
            VIS_FC_weight_ = tf.assign(VIS_FC_weight, ID_FC_weight)
            sess.run([NIR_FC_weight_,VIS_FC_weight_])

            NIR_FC_weight = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/NIR_layer_Para/NIR_Bottleneck/BatchNorm/moving_mean:0")
            VIS_FC_weight = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/VIS_layer_Para/VIS_Bottleneck/BatchNorm/moving_mean:0")
            ID_FC_weight = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Logits/Bottleneck/BatchNorm/moving_mean:0")
            NIR_FC_weight_=tf.assign(NIR_FC_weight,ID_FC_weight)
            VIS_FC_weight_ = tf.assign(VIS_FC_weight, ID_FC_weight)
            sess.run([NIR_FC_weight_,VIS_FC_weight_])

            NIR_FC_weight = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/NIR_layer_Para/NIR_Bottleneck/BatchNorm/moving_variance:0")
            VIS_FC_weight = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/VIS_layer_Para/VIS_Bottleneck/BatchNorm/moving_variance:0")
            ID_FC_weight = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Logits/Bottleneck/BatchNorm/moving_variance:0")
            NIR_FC_weight_=tf.assign(NIR_FC_weight,ID_FC_weight)
            VIS_FC_weight_ = tf.assign(VIS_FC_weight, ID_FC_weight)
            sess.run([NIR_FC_weight_,VIS_FC_weight_])
            #TODO 2FC_weight init end

            # Training and validation loop
            print('Running training')
            nrof_steps = args.max_nrof_epochs*args.epoch_size
            nrof_val_samples = int(math.ceil(args.max_nrof_epochs / args.validate_every_n_epochs))   # Validate every validate_every_n_epochs as well as in the last epoch
            stat = {
                'loss': np.zeros((nrof_steps,), np.float32),
                'center_loss': np.zeros((nrof_steps,), np.float32),
                'reg_loss': np.zeros((nrof_steps,), np.float32),
                'xent_loss': np.zeros((nrof_steps,), np.float32),
                'prelogits_norm': np.zeros((nrof_steps,), np.float32),
                'accuracy': np.zeros((nrof_steps,), np.float32),
                'val_loss': np.zeros((nrof_val_samples,), np.float32),
                'val_xent_loss': np.zeros((nrof_val_samples,), np.float32),
                'val_accuracy': np.zeros((nrof_val_samples,), np.float32),
                'lfw_accuracy': np.zeros((args.max_nrof_epochs,), np.float32),
                'lfw_valrate': np.zeros((args.max_nrof_epochs,), np.float32),
                'learning_rate': np.zeros((args.max_nrof_epochs,), np.float32),
                'time_train': np.zeros((args.max_nrof_epochs,), np.float32),
                'time_validate': np.zeros((args.max_nrof_epochs,), np.float32),
                'time_evaluate': np.zeros((args.max_nrof_epochs,), np.float32),
                'prelogits_hist': np.zeros((args.max_nrof_epochs, 1000), np.float32),
              }
            for epoch in range(1,args.max_nrof_epochs+1):
                step = sess.run(global_step, feed_dict=None)
                myepoch += 1
                # # Evaluate on LFW
                print('ID embeddings')
                t = time.time()
                if args.lfw_dir:
                    acc, val =evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
                        embeddings, label_batch, lfw_paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds, log_dir, step, summary_writer, stat, epoch,
                        args.lfw_distance_metric, args.lfw_subtract_mean, args.lfw_use_flipped_images, args.use_fixed_image_standardization)


                    LFW_acc.append(acc)
                    LFW_val.append(val)
                print('LFW_acc', LFW_acc)
                print('LFW_val', LFW_val)
                stat['time_evaluate'][epoch - 1] = time.time() - t
                # TODO evaluate 20180704
                if myepoch>5:
                    acc1, acc2 = CASIA_Rank1.Nir_Vis_evaluate_Rank_1(sess, images_placeholder, embeddings,
                                                                     phase_train_placeholder,Image_size=args.image_size)
                    summary = tf.Summary()
                    summary.value.add(tag='CASIA/JB_Rank1', simple_value=acc1)
                    summary.value.add(tag='CASIA/EcuDis_Rank1', simple_value=acc2)
                    # ver=CASIA_verfication.Nir_Vis_evaluate_verfication(sess,images_placeholder,embeddings,phase_train_placeholder)
                    # summary.value.add(tag='CASIA/verfi@Far=0.1%', simple_value=ver)
                    summary_writer.add_summary(summary, myepoch)


                    CASIA_JB_acc.append(acc1)
                    CASIA_Cosine_test.append(acc2)
                print('CASIA_JB_acc', CASIA_JB_acc)
                print('CASIA_Cosine_test', CASIA_Cosine_test)

                if myepoch>5:
                    for i in range(int(args.epoch_size/(args.iteration_ID+args.iteration_VIS+args.iteration_NIR+args.iteration_CdFD_domain+args.iteration_CdFD_backbone))):
                        print('iteration_ID, iteration_VIS, iteration_NIR, iteration_CdFD_domain, iteration_CdFD_backbone :',i+1,'/',int(args.epoch_size/(args.iteration_ID+args.iteration_VIS+args.iteration_NIR+args.iteration_CdFD_domain+args.iteration_CdFD_backbone)))



                        print('=====================================train_op_allVariable=====================================')
                        cont = train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                            learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, global_step,
                            total_loss, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file,
                            stat, cross_entropy_mean, accuracy, learning_rate,
                            prelogits, prelogits_center_loss, args.random_rotate, args.random_crop, args.random_flip, prelogits_norm, args.prelogits_hist_max, args.use_fixed_image_standardization)


                        print('=====================================train_op_allPara_NIR=====================================')
                        cont = train_allPara_NIR(args, sess, epoch, image_list_NIR, label_list_NIR, index_dequeue_op_NIR, enqueue_op,
                                 image_paths_placeholder, labels_placeholder,
                                 learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
                                 control_placeholder, global_step,
                                 total_loss_NIR, train_op_allPara_NIR, summary_op, summary_writer, regularization_losses_NIR_VIS,
                                 args.learning_rate_schedule_file,
                                 stat, cross_entropy_mean_NIR, accuracy_NIR, learning_rate,
                                 prelogits_NIR, prelogits_center_loss_NIR, args.random_rotate, args.random_crop, args.random_flip,
                                 prelogits_norm_NIR, args.prelogits_hist_max, args.use_fixed_image_standardization)

                        print('=====================================train_op_allPara_VIS=====================================')
                        cont = train_allPara_VIS(args, sess, epoch, image_list_VIS, label_list_VIS, index_dequeue_op_VIS, enqueue_op,
                                 image_paths_placeholder, labels_placeholder,
                                 learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
                                 control_placeholder, global_step,
                                 total_loss_VIS, train_op_allPara_VIS, summary_op, summary_writer, regularization_losses_NIR_VIS,
                                 args.learning_rate_schedule_file,
                                 stat, cross_entropy_mean_VIS, accuracy_VIS, learning_rate,
                                 prelogits_VIS, prelogits_center_loss_VIS, args.random_rotate, args.random_crop, args.random_flip,
                                 prelogits_norm_VIS, args.prelogits_hist_max, args.use_fixed_image_standardization)

                        'VIS_embeddings_center start'
                        NIR_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/NIR_layer_Para/NIR_layer_Para_FC/centers:0")
                        NIR_embeddings_center_array = sess.run(NIR_embeddings_center)

                        VIS_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/VIS_layer_Para/VIS_layer_Para_FC/centers:0")
                        VIS_embeddings_center_array = sess.run(VIS_embeddings_center)

                        ID_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/ID_FC_layer_Para/centers:0")
                        ID_embeddings_center_array = sess.run(ID_embeddings_center)
                        'VIS_embeddings_center end'

                        print('=====================================train_CdRloss=====================================')
                        train_CdRloss(args, sess, train_set, epoch, image_paths_placeholder, labels_placeholder,
                                      label_batch, batch_size_placeholder, learning_rate_placeholder,
                                      phase_train_placeholder,
                                      control_placeholder, enqueue_op, input_queue, global_step,
                                      embeddings, CdR_loss, loss_intra_domain, loss_intra_cor, loss_inter_modal_NIR_VIS,
                                      loss_inter_SameModality, learning_rate, train_op_CdRloss, summary_op,
                                      summary_writer, args.learning_rate_schedule_file)


                        print('=====================================train_op_domain_classifier_VIS=====================================')
                        cont = train_domain_classifier_VIS(args, sess, epoch, train_set, enqueue_op,
                                 image_paths_placeholder, labels_placeholder,
                                 learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
                                 control_placeholder, global_step,
                                 total_loss_domain_VIS, train_op_domain_classifier_VIS, summary_op, summary_writer,
                                 args.learning_rate_schedule_file,
                                 stat, cross_entropy_mean_domain_VIS, accuracy_domain_VIS, learning_rate,
                                 prelogits_VIS, args.random_rotate, args.random_crop, args.random_flip,
                                 prelogits_norm_VIS, args.prelogits_hist_max, args.use_fixed_image_standardization,phase_train=False)

                        print('=====================================train_op_domain_classifier_NIR=====================================')
                        cont = train_domain_classifier_NIR(args, sess, epoch, train_set, enqueue_op,
                                 image_paths_placeholder, labels_placeholder,
                                 learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
                                 control_placeholder, global_step,
                                 total_loss_domain_NIR, train_op_domain_classifier_NIR, summary_op, summary_writer,
                                 args.learning_rate_schedule_file,
                                 stat, cross_entropy_mean_domain_NIR, accuracy_domain_NIR, learning_rate,
                                 prelogits_NIR, args.random_rotate, args.random_crop, args.random_flip,
                                 prelogits_norm_NIR, args.prelogits_hist_max, args.use_fixed_image_standardization,phase_train=False)

                        print('=====================================train_op_CdFD_domain=====================================')
                        train_domain_CdFD(args, sess, train_set, epoch, image_paths_placeholder, labels_placeholder,
                                          label_batch, batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder,
                                          control_placeholder, enqueue_op, input_queue, global_step,
                                          embeddings,loss_CdFD_YV_YN,loss_CdFD_YVI_YN__YNI_YV, loss_CdFD_domain, learning_rate, train_op_FC_domain_CdFD, summary_op, summary_writer, args.learning_rate_schedule_file)

                        print('=====================================train_op_CdFD_backbone=====================================')
                        train_backbone_CdFD(args, sess, train_set, epoch, image_paths_placeholder, labels_placeholder,
                                          label_batch, batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder,
                                          control_placeholder, enqueue_op, input_queue, global_step,
                                          embeddings,loss_CdFD_YV_YN,loss_CdFD_YVI_YN__YNI_YV, loss_CdFD_domain, learning_rate, train_op_Backbone_CdFD, summary_op, summary_writer, args.learning_rate_schedule_file)

                        'VIS_embeddings_center start'
                        NIR_embeddings_center_op = tf.assign(NIR_embeddings_center, NIR_embeddings_center_array)
                        sess.run(NIR_embeddings_center_op)

                        VIS_embeddings_center_op = tf.assign(VIS_embeddings_center, VIS_embeddings_center_array)
                        sess.run(VIS_embeddings_center_op)

                        ID_embeddings_center_op = tf.assign(ID_embeddings_center, ID_embeddings_center_array)
                        sess.run(ID_embeddings_center_op)
                        'VIS_embeddings_center end'
                else:
                    for i in range(int(args.epoch_size/(args.iteration_ID+args.iteration_VIS+args.iteration_NIR))):
                        print('iteration_ID, iteration_VIS','   iteration_NIR:',i+1,'/',int(args.epoch_size/(args.iteration_ID+args.iteration_VIS+args.iteration_NIR)))


                        print('=====================================train_op_FC=====================================')
                        cont = train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op,
                                 image_paths_placeholder, labels_placeholder,
                                 learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
                                 control_placeholder, global_step,
                                 total_loss, train_op_FC, summary_op, summary_writer, regularization_losses,
                                 args.learning_rate_schedule_file,
                                 stat, cross_entropy_mean, accuracy, learning_rate,
                                 prelogits, prelogits_center_loss, args.random_rotate, args.random_crop, args.random_flip,
                                 prelogits_norm, args.prelogits_hist_max, args.use_fixed_image_standardization,phase_train=False)

                        print('=====================================train_op_FC_NIR=====================================')
                        cont = train_NIR(args, sess, epoch, image_list_NIR, label_list_NIR, index_dequeue_op_NIR, enqueue_op,
                                 image_paths_placeholder, labels_placeholder,
                                 learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
                                 control_placeholder, global_step,
                                 total_loss_NIR, train_op_FC_NIR, summary_op, summary_writer, regularization_losses_NIR_VIS,
                                 args.learning_rate_schedule_file,
                                 stat, cross_entropy_mean_NIR, accuracy_NIR, learning_rate,
                                 prelogits_NIR, prelogits_center_loss_NIR, args.random_rotate, args.random_crop, args.random_flip,
                                 prelogits_norm_NIR, args.prelogits_hist_max, args.use_fixed_image_standardization,phase_train=False)

                        print('=====================================train_op_FC_VIS=====================================')
                        cont = train_VIS(args, sess, epoch, image_list_VIS, label_list_VIS, index_dequeue_op_VIS, enqueue_op,
                                 image_paths_placeholder, labels_placeholder,
                                 learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
                                 control_placeholder, global_step,
                                 total_loss_VIS, train_op_FC_VIS, summary_op, summary_writer, regularization_losses_NIR_VIS,
                                 args.learning_rate_schedule_file,
                                 stat, cross_entropy_mean_VIS, accuracy_VIS, learning_rate,
                                 prelogits_VIS, prelogits_center_loss_VIS, args.random_rotate, args.random_crop, args.random_flip,
                                 prelogits_norm_VIS, args.prelogits_hist_max, args.use_fixed_image_standardization,phase_train=False)

                        'VIS_embeddings_center start'
                        NIR_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/NIR_layer_Para/NIR_layer_Para_FC/centers:0")
                        NIR_embeddings_center_array = sess.run(NIR_embeddings_center)

                        VIS_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/VIS_layer_Para/VIS_layer_Para_FC/centers:0")
                        VIS_embeddings_center_array = sess.run(VIS_embeddings_center)

                        ID_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/ID_FC_layer_Para/centers:0")
                        ID_embeddings_center_array = sess.run(ID_embeddings_center)
                        'VIS_embeddings_center end'
                        print('=====================================train_op_FC_domain_classifier_FC_VIS=====================================')
                        cont = train_domain_classifier_VIS(args, sess, epoch, train_set, enqueue_op,
                                 image_paths_placeholder, labels_placeholder,
                                 learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
                                 control_placeholder, global_step,
                                 total_loss_domain_VIS, train_op_domain_classifier_FC_VIS, summary_op, summary_writer,
                                 args.learning_rate_schedule_file,
                                 stat, cross_entropy_mean_domain_VIS, accuracy_domain_VIS, learning_rate,
                                 prelogits_VIS, args.random_rotate, args.random_crop, args.random_flip,
                                 prelogits_norm_VIS, args.prelogits_hist_max, args.use_fixed_image_standardization,phase_train=False)

                        print('=====================================train_op_FC_domain_classifier_FC_NIR=====================================')
                        cont = train_domain_classifier_NIR(args, sess, epoch, train_set, enqueue_op,
                                 image_paths_placeholder, labels_placeholder,
                                 learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
                                 control_placeholder, global_step,
                                 total_loss_domain_NIR, train_op_domain_classifier_FC_NIR, summary_op, summary_writer,
                                 args.learning_rate_schedule_file,
                                 stat, cross_entropy_mean_domain_NIR, accuracy_domain_NIR, learning_rate,
                                 prelogits_NIR, args.random_rotate, args.random_crop, args.random_flip,
                                 prelogits_norm_NIR, args.prelogits_hist_max, args.use_fixed_image_standardization,phase_train=False)

                        'VIS_embeddings_center start'
                        NIR_embeddings_center_op = tf.assign(NIR_embeddings_center, NIR_embeddings_center_array)
                        sess.run(NIR_embeddings_center_op)

                        VIS_embeddings_center_op = tf.assign(VIS_embeddings_center, VIS_embeddings_center_array)
                        sess.run(VIS_embeddings_center_op)

                        ID_embeddings_center_op = tf.assign(ID_embeddings_center, ID_embeddings_center_array)
                        sess.run(ID_embeddings_center_op)
                        'VIS_embeddings_center end'
                if not cont:
                    break

                t = time.time()
                if len(val_image_list)>0 and ((epoch-1) % args.validate_every_n_epochs == args.validate_every_n_epochs-1 or epoch==args.max_nrof_epochs):
                    validate(args, sess, epoch, val_image_list, val_label_list, enqueue_op, image_paths_placeholder, labels_placeholder, control_placeholder,
                        phase_train_placeholder, batch_size_placeholder,
                             stat, total_loss, regularization_losses, cross_entropy_mean, accuracy, args.validate_every_n_epochs, args.use_fixed_image_standardization)
                stat['time_validate'][epoch-1] = time.time() - t

                # Save variables and the metagraph if it doesn't exist already
                # save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, epoch)

                # print('Saving statistics')
                # with h5py.File(stat_file_name, 'w') as f:
                #     for key, value in stat.items():
                #         f.create_dataset(key, data=value)

    return model_dir

def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    #plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold

def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename,'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center>=distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths)<min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del(filtered_dataset[i])

    return filtered_dataset

def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, step,
      loss, train_op, summary_op, summary_writer, reg_losses, learning_rate_schedule_file,
      stat, cross_entropy_mean, accuracy,
      learning_rate, prelogits, prelogits_center_loss, random_rotate, random_crop, random_flip, prelogits_norm, prelogits_hist_max, use_fixed_image_standardization,phase_train=True):
    batch_number = 0

    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    if lr<=0:
        return False

    'VIS_embeddings_center start'
    NIR_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/NIR_layer_Para/NIR_layer_Para_FC/centers:0")
    NIR_embeddings_center_array = sess.run(NIR_embeddings_center)

    VIS_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/VIS_layer_Para/VIS_layer_Para_FC/centers:0")
    VIS_embeddings_center_array = sess.run(VIS_embeddings_center)
    'VIS_embeddings_center end'

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    control_value = facenet.RANDOM_ROTATE * random_rotate + facenet.RANDOM_CROP * random_crop + facenet.RANDOM_FLIP * random_flip + facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
    control_array = np.ones_like(labels_array) * control_value
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})

    # Training loop
    train_time = 0
    while batch_number < args.iteration_ID:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:phase_train, batch_size_placeholder:args.batch_size}
        tensor_list = [loss, train_op, step, reg_losses, prelogits, cross_entropy_mean, learning_rate, prelogits_norm, accuracy, prelogits_center_loss]
        if batch_number % 100 == 0:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_, summary_str = sess.run(tensor_list + [summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step_)
        else:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_ = sess.run(tensor_list, feed_dict=feed_dict)

        duration = time.time() - start_time
        stat['loss'][step_-1] = loss_
        stat['center_loss'][step_-1] = center_loss_
        stat['reg_loss'][step_-1] = np.sum(reg_losses_)
        stat['xent_loss'][step_-1] = cross_entropy_mean_
        stat['prelogits_norm'][step_-1] = prelogits_norm_
        stat['learning_rate'][epoch-1] = lr_
        stat['accuracy'][step_-1] = accuracy_
        stat['prelogits_hist'][epoch-1,:] += np.histogram(np.minimum(np.abs(prelogits_), prelogits_hist_max), bins=1000, range=(0.0, prelogits_hist_max))[0]

        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f\tCl %2.3f' %
              (epoch, batch_number+1, args.epoch_size, duration, loss_, cross_entropy_mean_, np.sum(reg_losses_), accuracy_, lr_, center_loss_))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)

    'VIS_embeddings_center start'
    # VIS_embeddings_center_array2 = sess.run(VIS_embeddings_center)
    NIR_embeddings_center_op = tf.assign(NIR_embeddings_center, NIR_embeddings_center_array)
    sess.run(NIR_embeddings_center_op)

    VIS_embeddings_center_op = tf.assign(VIS_embeddings_center, VIS_embeddings_center_array)
    sess.run(VIS_embeddings_center_op)
    'VIS_embeddings_center end'

    return True


def train_NIR(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, step,
          loss, train_op, summary_op, summary_writer, reg_losses, learning_rate_schedule_file,
          stat, cross_entropy_mean, accuracy,
          learning_rate, prelogits, prelogits_center_loss, random_rotate, random_crop, random_flip, prelogits_norm,
          prelogits_hist_max, use_fixed_image_standardization,phase_train=True):
    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate_Domain_MLR
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    if lr <= 0:
        return False
    'VIS_embeddings_center start'
    VIS_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/VIS_layer_Para/VIS_layer_Para_FC/centers:0")
    VIS_embeddings_center_array = sess.run(VIS_embeddings_center)

    ID_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/ID_FC_layer_Para/centers:0")
    ID_embeddings_center_array = sess.run(ID_embeddings_center)
    'VIS_embeddings_center end'

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    control_value = facenet.RANDOM_ROTATE * random_rotate + facenet.RANDOM_CROP * random_crop + facenet.RANDOM_FLIP * random_flip + facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
    control_array = np.ones_like(labels_array) * control_value
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,
                          control_placeholder: control_array})

    # Training loop
    train_time = 0
    while batch_number < args.iteration_NIR:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: phase_train,
                     batch_size_placeholder: args.batch_size}
        tensor_list = [loss, train_op, step, reg_losses, prelogits, cross_entropy_mean, learning_rate, prelogits_norm,
                       accuracy, prelogits_center_loss]
        if batch_number % 100 == 0:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_, summary_str = sess.run(
                tensor_list + [summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step_)
        else:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_ = sess.run(
                tensor_list, feed_dict=feed_dict)

        duration = time.time() - start_time
        stat['loss'][step_ - 1] = loss_
        stat['center_loss'][step_ - 1] = center_loss_
        stat['reg_loss'][step_ - 1] = np.sum(reg_losses_)
        stat['xent_loss'][step_ - 1] = cross_entropy_mean_
        stat['prelogits_norm'][step_ - 1] = prelogits_norm_
        stat['learning_rate'][epoch - 1] = lr_
        stat['accuracy'][step_ - 1] = accuracy_
        stat['prelogits_hist'][epoch - 1, :] += \
        np.histogram(np.minimum(np.abs(prelogits_), prelogits_hist_max), bins=1000, range=(0.0, prelogits_hist_max))[0]

        duration = time.time() - start_time
        print(
            'Epoch_NIR: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f\tCl %2.3f' %
            (epoch, batch_number + 1, args.epoch_size, duration, loss_, cross_entropy_mean_, np.sum(reg_losses_),
             accuracy_, lr_, center_loss_))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)

    'VIS_embeddings_center start'
    VIS_embeddings_center_op = tf.assign(VIS_embeddings_center, VIS_embeddings_center_array)
    sess.run(VIS_embeddings_center_op)
    ID_embeddings_center_op = tf.assign(ID_embeddings_center, ID_embeddings_center_array)
    sess.run(ID_embeddings_center_op)
    'VIS_embeddings_center end'
    return True

def train_allPara_NIR(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, step,
          loss, train_op, summary_op, summary_writer, reg_losses, learning_rate_schedule_file,
          stat, cross_entropy_mean, accuracy,
          learning_rate, prelogits, prelogits_center_loss, random_rotate, random_crop, random_flip, prelogits_norm,
          prelogits_hist_max, use_fixed_image_standardization,phase_train=True):
    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate_Backbone_MLR
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    if lr <= 0:
        return False
    'VIS_embeddings_center start'
    VIS_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/VIS_layer_Para/VIS_layer_Para_FC/centers:0")
    VIS_embeddings_center_array = sess.run(VIS_embeddings_center)

    ID_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/ID_FC_layer_Para/centers:0")
    ID_embeddings_center_array = sess.run(ID_embeddings_center)
    'VIS_embeddings_center end'
    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    control_value = facenet.RANDOM_ROTATE * random_rotate + facenet.RANDOM_CROP * random_crop + facenet.RANDOM_FLIP * random_flip + facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
    control_array = np.ones_like(labels_array) * control_value
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,
                          control_placeholder: control_array})

    # Training loop
    train_time = 0
    while batch_number < args.iteration_NIR:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: phase_train,
                     batch_size_placeholder: args.batch_size}
        tensor_list = [loss, train_op, step, reg_losses, prelogits, cross_entropy_mean, learning_rate, prelogits_norm,
                       accuracy, prelogits_center_loss]
        if batch_number % 100 == 0:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_, summary_str = sess.run(
                tensor_list + [summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step_)
        else:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_ = sess.run(
                tensor_list, feed_dict=feed_dict)

        duration = time.time() - start_time
        stat['loss'][step_ - 1] = loss_
        stat['center_loss'][step_ - 1] = center_loss_
        stat['reg_loss'][step_ - 1] = np.sum(reg_losses_)
        stat['xent_loss'][step_ - 1] = cross_entropy_mean_
        stat['prelogits_norm'][step_ - 1] = prelogits_norm_
        stat['learning_rate'][epoch - 1] = lr_
        stat['accuracy'][step_ - 1] = accuracy_
        stat['prelogits_hist'][epoch - 1, :] += \
        np.histogram(np.minimum(np.abs(prelogits_), prelogits_hist_max), bins=1000, range=(0.0, prelogits_hist_max))[0]

        duration = time.time() - start_time
        print(
            'Epoch_NIR: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f\tCl %2.3f' %
            (epoch, batch_number + 1, args.epoch_size, duration, loss_, cross_entropy_mean_, np.sum(reg_losses_),
             accuracy_, lr_, center_loss_))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)
    'VIS_embeddings_center start'
    VIS_embeddings_center_op = tf.assign(VIS_embeddings_center, VIS_embeddings_center_array)
    sess.run(VIS_embeddings_center_op)
    ID_embeddings_center_op = tf.assign(ID_embeddings_center, ID_embeddings_center_array)
    sess.run(ID_embeddings_center_op)
    'VIS_embeddings_center end'
    return True

def train_VIS(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, step,
          loss, train_op, summary_op, summary_writer, reg_losses, learning_rate_schedule_file,
          stat, cross_entropy_mean, accuracy,
          learning_rate, prelogits, prelogits_center_loss, random_rotate, random_crop, random_flip, prelogits_norm,
          prelogits_hist_max, use_fixed_image_standardization,phase_train=True):
    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate_Domain_MLR
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    if lr <= 0:
        return False
    'VIS_embeddings_center start'
    NIR_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/NIR_layer_Para/NIR_layer_Para_FC/centers:0")
    NIR_embeddings_center_array = sess.run(NIR_embeddings_center)

    ID_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/ID_FC_layer_Para/centers:0")
    ID_embeddings_center_array = sess.run(ID_embeddings_center)
    # print('VIS_embeddings_center_array',VIS_embeddings_center_array)
    'VIS_embeddings_center end'
    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    control_value = facenet.RANDOM_ROTATE * random_rotate + facenet.RANDOM_CROP * random_crop + facenet.RANDOM_FLIP * random_flip + facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
    control_array = np.ones_like(labels_array) * control_value
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,
                          control_placeholder: control_array})

    # Training loop
    train_time = 0
    while batch_number < args.iteration_VIS:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: phase_train,
                     batch_size_placeholder: args.batch_size}
        tensor_list = [loss, train_op, step, reg_losses, prelogits, cross_entropy_mean, learning_rate, prelogits_norm,
                       accuracy, prelogits_center_loss]
        if batch_number % 100 == 0:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_, summary_str = sess.run(
                tensor_list + [summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step_)
        else:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_ = sess.run(
                tensor_list, feed_dict=feed_dict)

        duration = time.time() - start_time
        stat['loss'][step_ - 1] = loss_
        stat['center_loss'][step_ - 1] = center_loss_
        stat['reg_loss'][step_ - 1] = np.sum(reg_losses_)
        stat['xent_loss'][step_ - 1] = cross_entropy_mean_
        stat['prelogits_norm'][step_ - 1] = prelogits_norm_
        stat['learning_rate'][epoch - 1] = lr_
        stat['accuracy'][step_ - 1] = accuracy_
        stat['prelogits_hist'][epoch - 1, :] += \
        np.histogram(np.minimum(np.abs(prelogits_), prelogits_hist_max), bins=1000, range=(0.0, prelogits_hist_max))[0]

        duration = time.time() - start_time
        print(
            'Epoch_VIS: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f\tCl %2.3f' %
            (epoch, batch_number + 1, args.epoch_size, duration, loss_, cross_entropy_mean_, np.sum(reg_losses_),
             accuracy_, lr_, center_loss_))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)
    'VIS_embeddings_center start'
    # VIS_embeddings_center_array2 = sess.run(VIS_embeddings_center)
    NIR_embeddings_center_op = tf.assign(NIR_embeddings_center, NIR_embeddings_center_array)
    sess.run(NIR_embeddings_center_op)

    ID_embeddings_center_op = tf.assign(ID_embeddings_center, ID_embeddings_center_array)
    sess.run(ID_embeddings_center_op)
    'VIS_embeddings_center end'
    return True
def train_allPara_VIS(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, step,
          loss, train_op, summary_op, summary_writer, reg_losses, learning_rate_schedule_file,
          stat, cross_entropy_mean, accuracy,
          learning_rate, prelogits, prelogits_center_loss, random_rotate, random_crop, random_flip, prelogits_norm,
          prelogits_hist_max, use_fixed_image_standardization,phase_train=True):
    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate_Backbone_MLR
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    if lr <= 0:
        return False

    'VIS_embeddings_center start'
    NIR_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/Modality_structure/NIR_layer_Para/NIR_layer_Para_FC/centers:0")
    NIR_embeddings_center_array = sess.run(NIR_embeddings_center)

    ID_embeddings_center = tf.get_default_graph().get_tensor_by_name("resnet50_CBAM/ID_FC_layer_Para/centers:0")
    ID_embeddings_center_array = sess.run(ID_embeddings_center)
    # print('VIS_embeddings_center_array',VIS_embeddings_center_array)
    'VIS_embeddings_center end'

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    control_value = facenet.RANDOM_ROTATE * random_rotate + facenet.RANDOM_CROP * random_crop + facenet.RANDOM_FLIP * random_flip + facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
    control_array = np.ones_like(labels_array) * control_value
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,
                          control_placeholder: control_array})

    # Training loop
    train_time = 0
    while batch_number < args.iteration_VIS:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: phase_train,
                     batch_size_placeholder: args.batch_size}
        tensor_list = [loss, train_op, step, reg_losses, prelogits, cross_entropy_mean, learning_rate, prelogits_norm,
                       accuracy, prelogits_center_loss]
        if batch_number % 100 == 0:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_, summary_str = sess.run(
                tensor_list + [summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step_)
        else:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_ = sess.run(
                tensor_list, feed_dict=feed_dict)

        duration = time.time() - start_time
        stat['loss'][step_ - 1] = loss_
        stat['center_loss'][step_ - 1] = center_loss_
        stat['reg_loss'][step_ - 1] = np.sum(reg_losses_)
        stat['xent_loss'][step_ - 1] = cross_entropy_mean_
        stat['prelogits_norm'][step_ - 1] = prelogits_norm_
        stat['learning_rate'][epoch - 1] = lr_
        stat['accuracy'][step_ - 1] = accuracy_
        stat['prelogits_hist'][epoch - 1, :] += \
        np.histogram(np.minimum(np.abs(prelogits_), prelogits_hist_max), bins=1000, range=(0.0, prelogits_hist_max))[0]

        duration = time.time() - start_time
        print(
            'Epoch_VIS: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f\tCl %2.3f' %
            (epoch, batch_number + 1, args.epoch_size, duration, loss_, cross_entropy_mean_, np.sum(reg_losses_),
             accuracy_, lr_, center_loss_))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)
    'VIS_embeddings_center start'
    # VIS_embeddings_center_array2 = sess.run(VIS_embeddings_center)
    NIR_embeddings_center_op = tf.assign(NIR_embeddings_center, NIR_embeddings_center_array)
    sess.run(NIR_embeddings_center_op)

    ID_embeddings_center_op = tf.assign(ID_embeddings_center, ID_embeddings_center_array)
    sess.run(ID_embeddings_center_op)
    'VIS_embeddings_center end'
    return True


def train_domain_classifier_VIS(args, sess, epoch, dataset, enqueue_op, image_paths_placeholder,
          labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, step,
          loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          stat, cross_entropy_mean, accuracy,
          learning_rate, prelogits, random_rotate, random_crop, random_flip, prelogits_norm,
          prelogits_hist_max, use_fixed_image_standardization,phase_train=True):


    batch_number = 0
    if args.learning_rate > 0.0:
        lr = args.learning_rate_domain_classifier
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    if lr <= 0:
        return False

    # Training loop
    train_time = 0


    while batch_number < args.iteration_domainClassifier_FC:
        # Sample people randomly from the dataset
        image_paths, num_per_class = sample_people_VIS_or_NIR_domain_classifier(dataset, args.people_per_batch_domainClassifier, args.images_per_person_domainClassifier,domain='VIS')
        # print(image_paths)

        # print('%.3f' % (time.time() - start_time))

        # print('image_paths',image_paths)
        labels = []
        for idx in range(args.people_per_batch_domainClassifier):
            for i in range(args.images_per_person_domainClassifier):
                labels.append([1])
        labels_array = np.array(labels)
        image_paths_array = np.expand_dims(np.array(image_paths), 1)

        # print('image_paths_array',image_paths_array)
        # print('labels_array', labels_array)

        control_value = facenet.RANDOM_ROTATE * random_rotate + facenet.RANDOM_CROP * random_crop + facenet.RANDOM_FLIP * random_flip + facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
        control_array = np.ones_like(labels_array) * control_value
        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,
                              control_placeholder: control_array})


        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: phase_train,
                     batch_size_placeholder: args.batch_size}
        tensor_list = [loss, train_op, step, prelogits, cross_entropy_mean, learning_rate, prelogits_norm,
                       accuracy]
        if batch_number % 100 == 0:
            loss_, _, step_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, summary_str = sess.run(
                tensor_list + [summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step_)
        else:
            loss_, _, step_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_ = sess.run(
                tensor_list, feed_dict=feed_dict)

        duration = time.time() - start_time
        stat['loss'][step_ - 1] = loss_

        stat['xent_loss'][step_ - 1] = cross_entropy_mean_
        stat['prelogits_norm'][step_ - 1] = prelogits_norm_
        stat['learning_rate'][epoch - 1] = lr_
        stat['accuracy'][step_ - 1] = accuracy_
        stat['prelogits_hist'][epoch - 1, :] += \
        np.histogram(np.minimum(np.abs(prelogits_), prelogits_hist_max), bins=1000, range=(0.0, prelogits_hist_max))[0]

        duration = time.time() - start_time
        print(
            'Epoch_domain_classifier_VIS: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tAccuracy %2.3f\tLr %2.5f' %
            (epoch, batch_number + 1, args.epoch_size, duration, loss_, cross_entropy_mean_,
             accuracy_, lr_))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)

    return True

def train_domain_classifier_NIR(args, sess, epoch, dataset, enqueue_op, image_paths_placeholder,
          labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, step,
          loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          stat, cross_entropy_mean, accuracy,
          learning_rate, prelogits, random_rotate, random_crop, random_flip, prelogits_norm,
          prelogits_hist_max, use_fixed_image_standardization,phase_train=True):

    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate_domain_classifier
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    if lr <= 0:
        return False
    # Training loop
    train_time = 0
    while batch_number < args.iteration_domainClassifier_FC:
        # Sample people randomly from the dataset
        image_paths, num_per_class = sample_people_VIS_or_NIR_domain_classifier(dataset, args.people_per_batch_domainClassifier, args.images_per_person_domainClassifier,domain='NIR')
        # print(image_paths)

        # print('%.3f' % (time.time() - start_time))

        # print('image_paths',image_paths)
        labels = []
        for idx in range(args.people_per_batch_domainClassifier):
            for i in range(args.images_per_person_domainClassifier):
                labels.append([0])
        labels_array = np.array(labels)
        image_paths_array = np.expand_dims(np.array(image_paths), 1)

        # print('image_paths_array',image_paths_array)
        # print('labels_array', labels_array)

        control_value = facenet.RANDOM_ROTATE * random_rotate + facenet.RANDOM_CROP * random_crop + facenet.RANDOM_FLIP * random_flip + facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
        control_array = np.ones_like(labels_array) * control_value
        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,
                              control_placeholder: control_array})


        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: phase_train,
                     batch_size_placeholder: args.batch_size}
        tensor_list = [loss, train_op, step, prelogits, cross_entropy_mean, learning_rate, prelogits_norm,
                       accuracy]
        if batch_number % 100 == 0:
            loss_, _, step_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, summary_str = sess.run(
                tensor_list + [summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step_)
        else:
            loss_, _, step_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_ = sess.run(
                tensor_list, feed_dict=feed_dict)

        duration = time.time() - start_time
        stat['loss'][step_ - 1] = loss_

        stat['xent_loss'][step_ - 1] = cross_entropy_mean_
        stat['prelogits_norm'][step_ - 1] = prelogits_norm_
        stat['learning_rate'][epoch - 1] = lr_
        stat['accuracy'][step_ - 1] = accuracy_
        stat['prelogits_hist'][epoch - 1, :] += \
        np.histogram(np.minimum(np.abs(prelogits_), prelogits_hist_max), bins=1000, range=(0.0, prelogits_hist_max))[0]

        duration = time.time() - start_time
        print(
            'Epoch_domain_classifier_NIR: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tAccuracy %2.3f\tLr %2.5f' %
            (epoch, batch_number + 1, args.epoch_size, duration, loss_, cross_entropy_mean_,
             accuracy_, lr_))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)

    return True

def train_domain_CdFD(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, control_placeholder, enqueue_op, input_queue,
          global_step,
          embeddings,loss_CdFD_YV_YN,loss_CdFD_YVI_YN__YNI_YV, loss_CdFD_domain, learning_rate,train_op, summary_op, summary_writer, learning_rate_schedule_file):
    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate_domain_CdFD
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    while batch_number < args.iteration_CdFD_domain:

        start_time = time.time()
        # Sample people randomly from the dataset
        image_paths, num_per_class = sample_people_VIS_NIR_CdFD(dataset, args.people_per_batch_CdFD, args.images_per_person_CdFD)
        # print(image_paths)

        # print('%.3f' % (time.time() - start_time))

        # print('image_paths',image_paths)
        labels = []
        for idx in range(args.people_per_batch_CdFD):
            for i in range(int(args.images_per_person_CdFD/2)):
                labels.append([idx + 1])
        labels=labels+labels
        labels = np.array(labels)
        image_paths = np.expand_dims(np.array(image_paths), 1)

        control_value = facenet.RANDOM_ROTATE * args.random_rotate + facenet.RANDOM_CROP * args.random_crop + facenet.RANDOM_FLIP * args.random_flip + facenet.FIXED_STANDARDIZATION * args.use_fixed_image_standardization
        control_array = np.ones_like(labels) * control_value
        sess.run(enqueue_op, {image_paths_placeholder: image_paths, labels_placeholder: labels, control_placeholder: control_array})
        # print('Running forward pass on sampled images: ', end='')
        # start_time = time.time()
        # batch_size = args.people_per_batch * args.images_per_person
        batch_size = len(image_paths)
        feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
        err, lr_,_, step, emb,loss_CdFD_YV_YN_,loss_CdFD_YVI_YN__YNI_YV_, lab = sess.run([loss_CdFD_domain, learning_rate,train_op, global_step, embeddings,loss_CdFD_YV_YN,loss_CdFD_YVI_YN__YNI_YV, labels_batch], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch_domain_CdFD: [%d][%d/%d]\tTime %.3f\tloss_CdFD_domain %2.5f\tloss_YV_YN %2.5f\tloss_YVI_YN__YNI_YV %2.5f\t learning_rate %2.5f' %
              (epoch, batch_number + 1, args.epoch_size, duration, err,loss_CdFD_YV_YN_,loss_CdFD_YVI_YN__YNI_YV_,lr_))
        # print('YV_YN_domain_', YV_YN_domain_)
        # print('alpa_one_matrix_domain_', alpa_one_matrix_domain_)
        batch_number += 1

        # Add validation loss and accuracy to summary
        summary = tf.Summary()
        # pylint: disable=maybe-no-member
        summary.value.add(tag='loss_CdFD_domain/loss', simple_value=err)
        summary_writer.add_summary(summary, step)

def train_backbone_CdFD(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, control_placeholder, enqueue_op, input_queue,
          global_step,
          embeddings,loss_CdFD_YV_YN,loss_CdFD_YVI_YN__YNI_YV, loss_CdFD_domain, learning_rate,train_op, summary_op, summary_writer, learning_rate_schedule_file):
    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate_backbone_CdFD
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)


    while batch_number < args.iteration_CdFD_backbone:

        start_time = time.time()
        # Sample people randomly from the dataset
        image_paths, num_per_class = sample_people_VIS_NIR_CdFD(dataset, args.people_per_batch_CdFD, args.images_per_person_CdFD)
        # print(image_paths)

        # print('%.3f' % (time.time() - start_time))

        # print('image_paths',image_paths)
        labels = []
        for idx in range(args.people_per_batch_CdFD):
            for i in range(int(args.images_per_person_CdFD/2)):
                labels.append([idx + 1])
        labels=labels+labels
        labels = np.array(labels)
        image_paths = np.expand_dims(np.array(image_paths), 1)

        control_value = facenet.RANDOM_ROTATE * args.random_rotate + facenet.RANDOM_CROP * args.random_crop + facenet.RANDOM_FLIP * args.random_flip + facenet.FIXED_STANDARDIZATION * args.use_fixed_image_standardization
        control_array = np.ones_like(labels) * control_value
        sess.run(enqueue_op, {image_paths_placeholder: image_paths, labels_placeholder: labels, control_placeholder: control_array})
        # print('Running forward pass on sampled images: ', end='')
        # start_time = time.time()
        # batch_size = args.people_per_batch * args.images_per_person
        batch_size = len(image_paths)
        feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
        err, lr_,_, step, emb,loss_CdFD_YV_YN_,loss_CdFD_YVI_YN__YNI_YV_, lab = sess.run([loss_CdFD_domain, learning_rate,train_op, global_step, embeddings,loss_CdFD_YV_YN,loss_CdFD_YVI_YN__YNI_YV, labels_batch], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch_backbone_CdFD: [%d][%d/%d]\tTime %.3f\tloss_CdFD_domain %2.5f\tloss_YV_YN %2.5f\tloss_YVI_YN__YNI_YV %2.5f\t learning_rate %2.5f' %
              (epoch, batch_number + 1, args.epoch_size, duration, err,loss_CdFD_YV_YN_,loss_CdFD_YVI_YN__YNI_YV_,lr_))
        # print('YV_YN_domain_', YV_YN_domain_)
        # print('alpa_one_matrix_domain_', alpa_one_matrix_domain_)
        batch_number += 1

        # Add validation loss and accuracy to summary
        summary = tf.Summary()
        # pylint: disable=maybe-no-member
        summary.value.add(tag='loss_CdFD_backbone/loss', simple_value=err)
        summary_writer.add_summary(summary, step)

def train_CdRloss(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, control_placeholder, enqueue_op, input_queue,
          global_step,
          embeddings,CdR_loss,loss_intra_domain,loss_intra_cor,loss_inter_modal_NIR_VIS,loss_inter_SameModality, learning_rate,train_op, summary_op, summary_writer, learning_rate_schedule_file):

    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate_CdR
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    while batch_number < args.iteration_CdR:

        start_time = time.time()
        # Sample people randomly from the dataset
        image_paths, num_per_class = sample_people_VIS_NIR(dataset, args.people_per_batch_CdR, args.images_per_person_CdR)
        # print(image_paths)

        # print('%.3f' % (time.time() - start_time))

        # print('image_paths',image_paths)
        labels = []
        for idx in range(args.people_per_batch_CdFD):
            for i in range(int(args.images_per_person_CdFD/2)):
                labels.append([idx + 1])
        labels=labels+labels
        labels = np.array(labels)
        image_paths = np.expand_dims(np.array(image_paths), 1)

        control_value = facenet.RANDOM_ROTATE * args.random_rotate + facenet.RANDOM_CROP * args.random_crop + facenet.RANDOM_FLIP * args.random_flip + facenet.FIXED_STANDARDIZATION * args.use_fixed_image_standardization
        control_array = np.ones_like(labels) * control_value
        sess.run(enqueue_op, {image_paths_placeholder: image_paths, labels_placeholder: labels, control_placeholder: control_array})
        # print('Running forward pass on sampled images: ', end='')
        # start_time = time.time()
        # batch_size = args.people_per_batch * args.images_per_person
        batch_size = len(image_paths)
        feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
        CdR_loss_, loss_intra_domain_, loss_intra_cor_, loss_inter_modal_NIR_VIS_, loss_inter_SameModality_, lr_,_, step, emb, lab = sess.run([CdR_loss,loss_intra_domain,loss_intra_cor,loss_inter_modal_NIR_VIS,loss_inter_SameModality, learning_rate,train_op, global_step, embeddings, labels_batch], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch_CdR: [%d][%d/%d]\tTime %.3f\tCdR_loss %2.3f\t ItraDomain %2.3f\t ItraAlign %2.3f\t IterNIRVIS %2.3f\t ItersameM %2.3f\t LR %2.5f' %
              (epoch, batch_number + 1, args.epoch_size, duration, CdR_loss_, loss_intra_domain_, loss_intra_cor_, loss_inter_modal_NIR_VIS_, loss_inter_SameModality_,lr_))
        # print('YV_YN_domain_', YV_YN_domain_)
        # print('alpa_one_matrix_domain_', alpa_one_matrix_domain_)
        batch_number += 1

        # Add validation loss and accuracy to summary
        summary = tf.Summary()
        # pylint: disable=maybe-no-member
        summary.value.add(tag='CdR_loss_/loss', simple_value=CdR_loss_)
        summary_writer.add_summary(summary, step)
    # print('image_paths',image_paths)


def sample_people_VIS_NIR(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        if nrof_images_in_class < images_per_person:
            print('nrof_images_in_class_not_satisfied: ', nrof_images_in_class)
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)

        # VIS samples
        idx_LIST = []

        while True:
            for idx in image_indices:
                path_modality = dataset[class_index].image_paths[idx].split('/')[-1][:3]
                if path_modality == 'VIS':
                    if len(idx_LIST) < int(images_per_person / 2):
                        idx_LIST.append(idx)
                    else:
                        break
            if len(idx_LIST) == int(images_per_person / 2):
                break
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx_LIST]
        image_paths += image_paths_for_class

        # NIR samples
        idx_LIST = []
        while True:
            for idx in image_indices:
                path_modality = dataset[class_index].image_paths[idx].split('/')[-1][:3]
                if path_modality == 'NIR':
                    if len(idx_LIST) < int(images_per_person / 2):
                        idx_LIST.append(idx)
                    else:
                        break
            if len(idx_LIST) == int(images_per_person / 2):
                break

        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx_LIST]
        image_paths += image_paths_for_class

        sampled_class_indices += [class_index] * images_per_person
        num_per_class.append(images_per_person)
        i += 1
    return image_paths, num_per_class


def sample_people_VIS_NIR_CdFD(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    image_paths_NIR=[]
    image_paths_VIS = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths_NIR+image_paths_VIS) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        if nrof_images_in_class < images_per_person:
            print('nrof_images_in_class_not_satisfied: ', nrof_images_in_class)
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)

        # VIS samples
        idx_LIST = []

        while True:
            for idx in image_indices:
                path_modality = dataset[class_index].image_paths[idx].split('/')[-1][:3]
                if path_modality == 'VIS':
                    if len(idx_LIST) < int(images_per_person / 2):
                        idx_LIST.append(idx)
                    else:
                        break
            if len(idx_LIST) == int(images_per_person / 2):
                break
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx_LIST]
        image_paths_VIS += image_paths_for_class

        # NIR samples
        idx_LIST = []
        while True:
            for idx in image_indices:
                path_modality = dataset[class_index].image_paths[idx].split('/')[-1][:3]
                if path_modality == 'NIR':
                    if len(idx_LIST) < int(images_per_person / 2):
                        idx_LIST.append(idx)
                    else:
                        break
            if len(idx_LIST) == int(images_per_person / 2):
                break

        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx_LIST]
        image_paths_NIR += image_paths_for_class

        sampled_class_indices += [class_index] * images_per_person
        num_per_class.append(images_per_person)
        i += 1
    image_paths=image_paths_VIS+image_paths_NIR
    return image_paths, num_per_class

def sample_people_VIS_or_NIR_domain_classifier(dataset, people_per_batch, images_per_person,domain='VIS'):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        if nrof_images_in_class < images_per_person:
            print('nrof_images_in_class_not_satisfied: ', nrof_images_in_class)
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)

        # VIS samples
        idx_LIST = []

        while True:
            for idx in image_indices:
                path_modality = dataset[class_index].image_paths[idx].split('/')[-1][:3]
                if path_modality == domain:
                    if len(idx_LIST) < int(images_per_person):
                        idx_LIST.append(idx)
                    else:
                        break
            if len(idx_LIST) == int(images_per_person):
                break
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx_LIST]
        image_paths += image_paths_for_class

        sampled_class_indices += [class_index] * images_per_person
        num_per_class.append(images_per_person)
        i += 1

    return image_paths, num_per_class


def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        if nrof_images_in_class < images_per_person:
            print('nrof_images_in_class: ', nrof_images_in_class, class_index)
        else:
            image_indices = np.arange(nrof_images_in_class)
            np.random.shuffle(image_indices)
            nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
            idx = image_indices[0:nrof_images_from_class]
            image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
            sampled_class_indices += [class_index] * nrof_images_from_class
            image_paths += image_paths_for_class
            num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class


def validate(args, sess, epoch, image_list, label_list, enqueue_op, image_paths_placeholder, labels_placeholder, control_placeholder,
             phase_train_placeholder, batch_size_placeholder,
             stat, loss, regularization_losses, cross_entropy_mean, accuracy, validate_every_n_epochs, use_fixed_image_standardization):

    print('Running forward pass on validation set')

    nrof_batches = len(label_list) // args.lfw_batch_size
    nrof_images = nrof_batches * args.lfw_batch_size

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_list[:nrof_images]),1)
    image_paths_array = np.expand_dims(np.array(image_list[:nrof_images]),1)
    control_array = np.ones_like(labels_array, np.int32)*facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})

    loss_array = np.zeros((nrof_batches,), np.float32)
    xent_array = np.zeros((nrof_batches,), np.float32)
    accuracy_array = np.zeros((nrof_batches,), np.float32)

    # Training loop
    start_time = time.time()
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:args.lfw_batch_size}
        loss_, cross_entropy_mean_, accuracy_ = sess.run([loss, cross_entropy_mean, accuracy], feed_dict=feed_dict)
        loss_array[i], xent_array[i], accuracy_array[i] = (loss_, cross_entropy_mean_, accuracy_)
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')

    duration = time.time() - start_time

    val_index = (epoch-1)//validate_every_n_epochs
    stat['val_loss'][val_index] = np.mean(loss_array)
    stat['val_xent_loss'][val_index] = np.mean(xent_array)
    stat['val_accuracy'][val_index] = np.mean(accuracy_array)

    print('Validation Epoch: %d\tTime %.3f\tLoss %2.3f\tXent %2.3f\tAccuracy %2.3f' %
          (epoch, duration, np.mean(loss_array), np.mean(xent_array), np.mean(accuracy_array)))


def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
        embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer, stat, epoch, distance_metric, subtract_mean, use_flipped_images, use_fixed_image_standardization):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')

    # Enqueue one epoch of image paths and labels
    nrof_embeddings = len(actual_issame)*2  # nrof_pairs * nrof_images_per_pair
    nrof_flips = 2 if use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips
    labels_array = np.expand_dims(np.arange(0,nrof_images),1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),nrof_flips),1)
    control_array = np.zeros_like(labels_array, np.int32)
    if use_fixed_image_standardization:
        control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION
    if use_flipped_images:
        # Flip every second image
        control_array += (labels_array % 2)*facenet.FLIP
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})

    embedding_size = int(embeddings.get_shape()[1])
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))
    if use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the images
        embeddings[:,:embedding_size] = emb_array[0::2,:]
        embeddings[:,embedding_size:] = emb_array[1::2,:]
    else:
        embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    _, _, accuracy, val, val_std, far = lfw.evaluate(embeddings, actual_issame, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)

    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))
    stat['lfw_accuracy'][epoch-1] = np.mean(accuracy)
    stat['lfw_valrate'][epoch-1] = val
    return np.mean(accuracy),val
def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)

def save_variables_and_metagraph_best(sess, saver, summary_writer, model_dir, model_name, step,firstSave):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename) and firstSave==True:
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)

class Logger(object):
    def __init__(self,filename='logrecord.log'):
        self.terminal=sys.stdout
        self.log=open(filename,"a")
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
        help='Directory where to write event logs.', default='./logs/DFD_20201029_resnet50_CBAM_MSCeleb1M_clean_182_44_256dims_CASIA_NIR_VIS_160_5_3softmaxAllPara_DomainClassifier_CdFD_centerfix_DRR')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='./models/DFD_20201029_resnet50_CBAM_MSCeleb1M_clean_182_44_256dims_CASIA_NIR_VIS_160_5_3softmaxAllPara_DomainClassifier_CdFD_centerfix_DRR')
    parser.add_argument('--best_models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.',
                        default='./models/DFD_resnet34_CBAM_MSCeleb1M_clean_Pretrain_Centerloss_256dims_3SOFTMAX_CdFD2_bestModel')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.85)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.', default='./models/Resnet50_CBAM_160_256dims_182_44_minePretrained_20201013/LFW_BEST/')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='./../../Datasets/CASIA NIR-VIS 2.0/10_folds/NIR_VIS_protocols_1_160_5/')
    parser.add_argument('--data_dir_NIR', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='./../../Datasets/CASIA NIR-VIS 2.0/10_folds/NIR_VIS_protocols_1_NIR_VIS_fold_160_5/NIR_data/')
    parser.add_argument('--data_dir_VIS', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='./../../Datasets/CASIA NIR-VIS 2.0/10_folds/NIR_VIS_protocols_1_NIR_VIS_fold_160_5/VIS_data/')
    parser.add_argument('--data_dir_domain', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='./../../Datasets/CASIA NIR-VIS 2.0/10_folds/NIR_VIS_protocols_1_2class/')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.DFD_20201030_model_ResNet50_CBAM_slim_paper')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=64)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=450)
    parser.add_argument('--iteration_ID', type=int,
        help='Number of batches per epoch.', default=20)
    parser.add_argument('--iteration_VIS', type=int,
        help='Number of batches per epoch.', default=20)
    parser.add_argument('--iteration_NIR', type=int,
        help='Number of batches per epoch.', default=20)
    parser.add_argument('--iteration_CdFD_domain', type=int,
        help='Number of batches per epoch.', default=20)
    parser.add_argument('--iteration_CdFD_backbone', type=int,
        help='Number of batches per epoch.', default=20)
    parser.add_argument('--iteration_domainClassifier_FC', type=int,
        help='Number of batches per epoch.', default=20)
    parser.add_argument('--iteration_domainClassifier_backbone', type=int,
        help='Number of batches per epoch.', default=20)
    parser.add_argument('--iteration_CdR', type=int,
        help='Number of batches per epoch.', default=20)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=256)

    #CdFD_loss
    parser.add_argument('--people_per_batch_CdFD', type=int,
        help='Number of people per batch.', default=16)
    parser.add_argument('--images_per_person_CdFD', type=int,
        help='Number of images per person.', default=4)
    parser.add_argument('--learning_rate_domain_CdFD', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.005)
    parser.add_argument('--learning_rate_backbone_CdFD', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.005)

    # domain classifier
    parser.add_argument('--people_per_batch_domainClassifier', type=int,
        help='Number of people per batch.', default=16)
    parser.add_argument('--images_per_person_domainClassifier', type=int,
        help='Number of images per person.', default=4)
    parser.add_argument('--learning_rate_domain_classifier', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.005)


    # #CdR loss
    parser.add_argument('--people_per_batch_CdR', type=int,
        help='Number of people per batch.', default=16)
    parser.add_argument('--images_per_person_CdR', type=int,
        help='Number of images per person.', default=4)
    parser.add_argument('--learning_rate_CdR', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.001)



    parser.add_argument('--random_crop',
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', default=False)
    parser.add_argument('--random_flip',
        help='Performs random horizontal flipping of training images.', default=False)
    parser.add_argument('--random_rotate',
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--use_fixed_image_standardization',
        help='Performs fixed standardization of images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=0.8)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=5e-5)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.5)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.9)
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
        help='Loss based on the norm of the activations in the prelogits layer.', default=0.0)
    parser.add_argument('--prelogits_norm_p', type=float,
        help='Norm to use for prelogits norm loss.', default=1.0)
    parser.add_argument('--prelogits_hist_max', type=float,
        help='The max value for the prelogits histogram.', default=10.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='RMSPROP')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.01)
    parser.add_argument('--learning_rate_Domain_MLR', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.01)
    parser.add_argument('--learning_rate_Backbone_MLR', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.001)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=4)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=0.95)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms',
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule_classifier_msceleb_inception_resnet_v1_20191212.txt')
    parser.add_argument('--filter_filename', type=str,
        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)
    parser.add_argument('--validate_every_n_epochs', type=int,
        help='Number of epoch between validation', default=5)
    parser.add_argument('--validation_set_split_ratio', type=float,
        help='The ratio of the total dataset to use for validation', default=0.0)
    parser.add_argument('--min_nrof_val_images_per_class', type=float,
        help='Classes with fewer images will be removed from the validation set', default=0)
 
    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='./data/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='./../../Datasets/lfw_mtcnnpy_160/')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--lfw_distance_metric', type=int,
        help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    parser.add_argument('--lfw_use_flipped_images', 
        help='Concatenates embeddings for the image and its horizontally flipped counterpart.', action='store_true')
    parser.add_argument('--lfw_subtract_mean', 
        help='Subtract feature mean before calculating distance.', action='store_true')
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
