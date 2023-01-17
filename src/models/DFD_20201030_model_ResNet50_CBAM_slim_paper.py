# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn

# CBAM Attentin

def channel_attention_module(inputs, reduction_ratio, reuse=None, scope='channel_attention'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            input_channel = inputs.get_shape().as_list()[-1]
            num_squeeze = input_channel // reduction_ratio

            avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keep_dims=True)
            assert avg_pool.get_shape()[1:] == (1, 1, input_channel)
            avg_pool = slim.fully_connected(avg_pool, num_squeeze, activation_fn=None, reuse=None, scope='fc1')
            avg_pool = slim.fully_connected(avg_pool, input_channel, activation_fn=None, reuse=None, scope='fc2')
            assert avg_pool.get_shape()[1:] == (1, 1, input_channel)

            max_pool = tf.reduce_max(inputs, axis=[1, 2], keep_dims=True)
            assert max_pool.get_shape()[1:] == (1, 1, input_channel)
            max_pool = slim.fully_connected(max_pool, num_squeeze, activation_fn=None, reuse=True, scope='fc1')
            max_pool = slim.fully_connected(max_pool, input_channel, activation_fn=None, reuse=True, scope='fc2')
            assert max_pool.get_shape()[1:] == (1, 1, input_channel)


            scale = tf.nn.sigmoid(avg_pool + max_pool)

            channel_attention = scale * inputs
            channel_vec=scale
            return channel_attention, channel_vec


def spatial_attention_module(inputs, kernel_size=7, reuse=None, scope='spatial_attention'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            avg_pool = tf.reduce_mean(inputs, axis=3, keep_dims=True)
            assert avg_pool.get_shape()[-1] == 1
            max_pool = tf.reduce_max(inputs, axis=3, keep_dims=True)
            assert max_pool.get_shape()[-1] == 1

            concat = tf.concat([avg_pool, max_pool], axis=3)
            assert concat.get_shape()[-1] == 2

            concat = slim.conv2d(concat, 1, kernel_size, padding='SAME', activation_fn=None, scope='conv')
            scale = tf.nn.sigmoid(concat)
            spatial_attention = scale * inputs
            spatial_mat= scale
            return spatial_attention, spatial_mat

def cbam_block_parallel(inputs, reduction_ratio=16, reuse=None, scope='CBAM_Block_Parallel'):
    with tf.variable_scope(scope, reuse=reuse):
        spatial_attention, spatial_mat = spatial_attention_module(inputs, kernel_size=7, reuse=None, scope='spatial_attention')
        channel_attention, channel_vec = channel_attention_module(spatial_attention, reduction_ratio, reuse=None, scope='channel_attention')
        out = spatial_attention + channel_attention
        return out


# residual block
def res_block_begin(input, block_1_stride=2, name='block_1', conv1_output_channel=64, conv2_output_channel=64, conv3_output_channel=256):
	with tf.variable_scope(name):
		print('input', input)
		net = slim.conv2d(input, conv1_output_channel, 1, stride=block_1_stride, padding='SAME', activation_fn=nn.relu, scope='Conv1')
		print('Conv1', net)
		net = slim.conv2d(net, conv2_output_channel, 3, stride=1, padding='SAME', activation_fn=nn.relu, scope='Conv2')
		print('Conv2', net)
		net = slim.conv2d(net, conv3_output_channel, 1, stride=1, padding='SAME', activation_fn=nn.relu, scope='Conv3')
		print('Conv3', net)
	return net
def res_block(input, name='block_2', conv1_output_channel=64, conv2_output_channel=64, conv3_output_channel=256):
	with tf.variable_scope(name):
		print('input', input)
		net = slim.conv2d(input, conv1_output_channel, 1, stride=1, padding='SAME', activation_fn=nn.relu, scope='Conv1')
		print('Conv1', net)
		net = slim.conv2d(net, conv2_output_channel, 3, stride=1, padding='SAME', activation_fn=nn.relu, scope='Conv2')
		print('Conv2', net)
		net = slim.conv2d(net, conv3_output_channel, 1, stride=1, padding='SAME', activation_fn=nn.relu, scope='Conv3')
		print('Conv3', net)
		net=cbam_block_parallel(net)
		print('cbam_attention', net)
		net=net+input
	return net


def res_section(input,block_1_stride=2, conv1_output_channel=64, conv2_output_channel=64, conv3_output_channel=256,block_num=3):
	print('block_num', 1)
	net=res_block_begin(input,block_1_stride=block_1_stride, name='block_1', conv1_output_channel=conv1_output_channel, conv2_output_channel=conv2_output_channel, conv3_output_channel=conv3_output_channel)
	for idx in range(1,block_num):
		print('block_num',idx+1)
		net = res_block(net, name='block_'+str(idx+1), conv1_output_channel=conv1_output_channel, conv2_output_channel=conv2_output_channel, conv3_output_channel=conv3_output_channel)
	return net
def inference(images, keep_probability, phase_train=True,
			  bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
	batch_norm_params = {
		# Decay for the moving averages.
		'decay': 0.995,
		# epsilon to prevent 0s in variance.
		'epsilon': 0.001,
		# force in-place updates of mean and variance estimates
		'updates_collections': None,
		# Moving averages ends up in the trainable variables collection
		'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
	}

	with slim.arg_scope([slim.conv2d, slim.fully_connected],
						weights_initializer=slim.initializers.xavier_initializer(),
						weights_regularizer=slim.l2_regularizer(weight_decay),
						normalizer_fn=slim.batch_norm,
						normalizer_params=batch_norm_params):
		return resnet50_CBAM(images, is_training=phase_train,
								   dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size,
								   reuse=reuse)


def resnet50_CBAM(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='resnet50_CBAM'):
	"""Creates the LCNN9 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
	end_points = {}

	with tf.variable_scope(scope, 'resnet', [inputs], reuse=reuse):
		with slim.arg_scope([slim.batch_norm, slim.dropout],
							is_training=is_training):
			with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
								stride=1, padding='SAME'):
				with tf.variable_scope('section1'):
					print('inputs',inputs)
					net = slim.conv2d(inputs, 64, 7, stride=2, padding='SAME', activation_fn=nn.relu, scope='Conv1')
					print('Conv1', net)
					net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='MaxPool_1')
					print('MaxPool_1', net)

				with tf.variable_scope('section2'):
					print('section2')
					OutC=[64,64,256]
					block_num=3
					net=res_section(net,block_1_stride=1, conv1_output_channel=OutC[0], conv2_output_channel=OutC[1], conv3_output_channel=OutC[2], block_num=block_num)

				with tf.variable_scope('section3'):
					print('section3')
					OutC=[128,128,512]
					block_num=4
					net = res_section(net, conv1_output_channel=OutC[0], conv2_output_channel=OutC[1], conv3_output_channel=OutC[2], block_num=block_num)

				with tf.variable_scope('section4'):
					print('section4')
					OutC = [256, 256, 1024]
					block_num = 6
					net = res_section(net, conv1_output_channel=OutC[0], conv2_output_channel=OutC[1], conv3_output_channel=OutC[2], block_num=block_num)

				with tf.variable_scope('section5'):
					print('section5')
					OutC = [512, 512, 2048]
					block_num = 3
					net = res_section(net, conv1_output_channel=OutC[0], conv2_output_channel=OutC[1], conv3_output_channel=OutC[2], block_num=block_num)

				with tf.variable_scope('Logits'):
					print('Logits')
					net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',scope='Global_AvgPool')
					net = slim.flatten(net)
					flatten_layer = net
					print('flatten', net)
					net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='Dropout')
					net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck', reuse=False)
					print('fully_connected', net)

				NIR_Bottleneck_layer_size=bottleneck_layer_size
				with tf.variable_scope('Modality_structure'):
					with tf.variable_scope('NIR_layer_Para'):
						NIR_layer = slim.fully_connected(flatten_layer, NIR_Bottleneck_layer_size, activation_fn=None, scope='NIR_Bottleneck', reuse=False)
						print('NIR_layer', NIR_layer)
					with tf.variable_scope('VIS_layer_Para'):
						VIS_layer = slim.fully_connected(flatten_layer, NIR_Bottleneck_layer_size, activation_fn=None, scope='VIS_Bottleneck', reuse=False)
						print('VIS_layer', VIS_layer)

	return net,NIR_layer,VIS_layer

