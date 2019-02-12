#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-1-29 
# @Author  : David

"""
实现一个基于SHUFFLENETV1的特征编码类
"""
from collections import OrderedDict
import math

import tensorflow as tf

from encoder_decoder_model import cnn_basenet
from config import global_config

CFG = global_config.cfg


class SHUFFLENETV1Encoder(cnn_basenet.CNNBaseModel):
    """
    实现了一个基于SHUFFLENETV1的特征编码类
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(SHUFFLENETV1Encoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def _depthwise_stage(self, input_tensor, k_size, out_dims, name,
                    stride=1, pad='SAME'):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(inputdata=input_tensor, out_channel=out_dims,
                               kernel_size=k_size, stride=stride,
                               use_bias=False, padding=pad, name='conv')

            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def _conv_stage(self, input_tensor, k_size, out_dims, name,
                    stride=1, pad='SAME'):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(inputdata=input_tensor, out_channel=out_dims,
                               kernel_size=k_size, stride=stride,
                               use_bias=False, padding=pad, name='conv')

            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def depthwise_separable_conv_stage(self, input_tensor, k_size, out_dims, name,
                    stride=1, pad='SAME'):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :return:
        """
        with tf.variable_scope(name):
            dw_conv = self.depthwise_conv2d(input_tensor=input_tensor, name='dw_conv',
                               k_size=k_size, stride=stride, padding=pad)
            
            dw_bn = self.layerbn(inputdata=dw_conv, is_training=self._is_training, name='dw_bn')

            relu = self.relu(inputdata=dw_bn, name='relu')

            pw_conv = self.conv2d(inputdata=relu, out_channel=out_dims,
                               kernel_size=1, stride=1,
                               use_bias=False, padding=pad, name='pw_conv') 
            
            pw_bn = self.layerbn(inputdata=pw_conv, is_training=self._is_training, name='pw_bn')     

        return pw_bn

    def res_block(self, input_tensor, expansion_ratio, out_dims, name, stride=1, pad='SAME' 
                  , k_size = 3, shortcut=True):
        with tf.variable_scope(name):
            #pw
            bottleneck_dim=round(expansion_ratio*input_tensor.get_shape().as_list()[-1])

            pw_conv_1 = self.conv2d(inputdata=input_tensor, out_channel=bottleneck_dim,
                                kernel_size=1, stride=1,
                                use_bias=False, padding=pad, name='pw_conv_1')

            pw_bn_1 = self.layerbn(inputdata=pw_conv_1, is_training=self._is_training, name='pw_bn_1')    

            pw_relu = self.relu(inputdata=pw_bn_1, name='pw_relu')

            #dw
            dw_conv = self.depthwise_conv2d(input_tensor=pw_relu, name='dw_conv',
                               k_size=k_size, stride=stride, padding=pad)
            
            dw_bn = self.layerbn(inputdata=dw_conv, is_training=self._is_training, name='dw_bn')

            dw_relu = self.relu(inputdata=dw_bn, name='dw_relu')

            #pw & linear
            pw_conv_2 = self.conv2d(inputdata=dw_relu, out_channel=out_dims,
                                kernel_size=1, stride=1,
                                use_bias=False, padding=pad, name='pw_conv_2')
            
            pw_bn_2 = self.layerbn(inputdata=pw_conv_2, is_training=self._is_training, name='pw_bn_2')  

            #element wise add, only for stride==1
            if shortcut and stride == 1:
                in_dim=int(input_tensor.get_shape().as_list()[-1])
                if in_dim != out_dims:
                    ex_dim = self.conv2d(inputdata=input_tensor, out_channel=out_dims,
                                kernel_size=1, stride=1,
                                use_bias=False, padding=pad, name='ex_dim')
                    pw_bn_2 = pw_bn_2 + ex_dim
                else:
                    pw_bn_2 = input_tensor + pw_bn_2

        return pw_bn_2            

    def channel_shuffle(self, name, x, num_groups):
        with tf.variable_scope(name) as scope:
            n, h, w, c = x.shape.as_list()
            x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
            x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
            output = tf.reshape(x_transposed, [-1, h, w, c])
        return output

    def grouped_conv2d(self, x, num_filters, name, stride=1, kernel_size=1, padding='SAME', num_groups=3):
        with tf.variable_scope(name) as scope:
            sz = x.get_shape()[3].value // num_groups
            conv = [self.conv2d(inputdata=x[:, :, :, i * sz:i * sz + sz], out_channel=num_filters // num_groups,
                               kernel_size=kernel_size, stride=stride,
                               use_bias=False, padding=padding, name=name + "_" + str(i)) for i in range(num_groups)]          
            conv_g = tf.concat(conv, axis=-1)
        return conv_g

    def shuffle_conv_stage(self, input_tensor, out_dims, name, num_groups=3, stride=1, pad='SAME',
                           k_size=3, use_add=True, use_avg_pool=False):#stride,depthwise
        with tf.variable_scope(name):
            if use_avg_pool == True:
                avgpool = self.avgpooling(input_tensor, kernel_size=3, stride=2, name='avgpool')
            else:
                avgpool = input_tensor

            pw_conv_1 =  self.grouped_conv2d(x=input_tensor, num_filters=out_dims//4,
                                   name='pw_conv_1', stride=1, num_groups=num_groups)

            pw_bn_1 = self.layerbn(inputdata=pw_conv_1, is_training=self._is_training, name='pw_bn_1')

            relu_1 = self.relu(inputdata=pw_bn_1, name='relu_1')

            shuffled = self.channel_shuffle(name = 'channel_shuffle', x = relu_1, num_groups = 3)

            dw_conv = self.depthwise_conv2d(input_tensor=shuffled, name='dw_conv',
                               k_size=k_size, stride=stride, padding=pad)

            dw_bn = self.layerbn(inputdata=dw_conv, is_training=self._is_training, name='dw_bn')          

            group_conv = self.grouped_conv2d(x=dw_bn, num_filters=out_dims, name='group_conv',
                                             stride=1, num_groups=num_groups)

            gc_bn = self.layerbn(inputdata=group_conv, is_training=self._is_training, name='gc_bn') 

            if use_add == True:
                res = avgpool + gc_bn
            else:
                res = tf.concat([avgpool, gc_bn], axis=-1)
            
            relu_2 = self.relu(inputdata=res, name='relu_2')

        return relu_2

    def _conv_dilated_stage(self, input_tensor, k_size, out_dims, name,
                    dilation=1, pad='SAME'):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.dilation_conv(input_tensor=input_tensor, out_dims=out_dims,
                               k_size=k_size, rate = dilation,
                               use_bias=False, padding=pad, name='conv')

            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu


    def _fc_stage(self, input_tensor, out_dims, name, use_bias=False):
        """

        :param input_tensor:
        :param out_dims:
        :param name:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name):
            fc = self.fullyconnect(inputdata=input_tensor, out_dim=out_dims, use_bias=use_bias,
                                   name='fc')

            bn = self.layerbn(inputdata=fc, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def encode(self, input_tensor, name):
        """
        根据SHUFFLENETV1框架对输入的tensor进行编码
        :param input_tensor:
        :param name:
        :param flags:
        :return: 输出SHUFFLENETV1编码特征
        """
        ret = OrderedDict()

        with tf.variable_scope(name):
            # conv stage 1
            conv_1 = self._conv_stage(input_tensor=input_tensor, k_size=3,
                                      out_dims=24, name='conv1', stride=2)#(8,144,400,24)

            pool_1_1 = self.maxpooling(conv_1, kernel_size=3, stride=2, name='pool_1_1',padding='SAME')#(8,72,200,24)

            pool_1_2 = self.avgpooling(pool_1_1, kernel_size=3, stride=2, name='pool_1_2',padding='SAME')

            # depth_wise stage 2
            pw_conv_2 = self._conv_stage(input_tensor=pool_1_1, k_size=1,
                                      out_dims=54, name='pw_conv_2', stride=1)
            
            dw_conv_2 = self.depthwise_conv2d(input_tensor=pw_conv_2, name='dw_conv_2',
                               k_size=3, stride=2)
            
            dw_bn_2 = self.layerbn(inputdata=dw_conv_2, is_training=self._is_training, name='dw_bn_2')

            group_conv_2 = self.grouped_conv2d(x=dw_bn_2, num_filters=216, name='group_conv_2', stride=1, kernel_size=1, padding='SAME', num_groups=3)

            group_conv_bn_2 = self.layerbn(inputdata=group_conv_2, is_training=self._is_training, name='group_conv_bn_2')

            concat_2 = tf.concat([pool_1_2, group_conv_bn_2], axis=-1)

            relu_2 = self.relu(inputdata=concat_2, name='relu_2')#(8,36,100,240)

            # shuffle_conv_stage stage 3
            conv_3_1_shuffle = self.shuffle_conv_stage(input_tensor=relu_2, out_dims=240, name='conv_3_1_shuffle',
                              num_groups=3, stride=1, pad='SAME', k_size=3)

            conv_3_2_shuffle = self.shuffle_conv_stage(input_tensor=conv_3_1_shuffle, out_dims=240, name='conv_3_2_shuffle',
                              num_groups=3, stride=1, pad='SAME', k_size=3)

            conv_3_3_shuffle = self.shuffle_conv_stage(input_tensor=conv_3_2_shuffle, out_dims=240, name='conv_3_3_shuffle',
                              num_groups=3, stride=1, pad='SAME', k_size=3)

            conv_3_4_shuffle = self.shuffle_conv_stage(input_tensor=conv_3_3_shuffle, out_dims=240, name='conv_3_4_shuffle',
                              num_groups=3, stride=2, pad='SAME', k_size=3, use_add=False, use_avg_pool=True)#(8,18,50,240)
            '''
            # shuffle_conv_stage stage 4
            conv_4_1_shuffle = self.shuffle_conv_stage(input_tensor=conv_3_4_shuffle, out_dims=480, name='conv_4_1_shuffle',
                              num_groups=3, stride=1, pad='SAME', k_size=3)

            conv_4_2_shuffle = self.shuffle_conv_stage(input_tensor=conv_4_1_shuffle, out_dims=480, name='conv_4_2_shuffle',
                              num_groups=3, stride=1, pad='SAME', k_size=3)

            conv_4_3_shuffle = self.shuffle_conv_stage(input_tensor=conv_4_2_shuffle, out_dims=480, name='conv_4_3_shuffle',
                              num_groups=3, stride=1, pad='SAME', k_size=3)

            conv_4_4_shuffle = self.shuffle_conv_stage(input_tensor=conv_4_3_shuffle, out_dims=480, name='conv_4_4_shuffle',
                              num_groups=3, stride=1, pad='SAME', k_size=3)
          
            conv_4_5_shuffle = self.shuffle_conv_stage(input_tensor=conv_4_4_shuffle, out_dims=480, name='conv_4_5_shuffle',
                              num_groups=3, stride=1, pad='SAME', k_size=3)

            conv_4_6_shuffle = self.shuffle_conv_stage(input_tensor=conv_4_5_shuffle, out_dims=480, name='conv_4_6_shuffle',
                              num_groups=3, stride=1, pad='SAME', k_size=3)

            conv_4_7_shuffle = self.shuffle_conv_stage(input_tensor=conv_4_6_shuffle, out_dims=480, name='conv_4_7_shuffle',
                              num_groups=3, stride=1, pad='SAME', k_size=3)

            conv_4_8_shuffle = self.shuffle_conv_stage(input_tensor=conv_4_7_shuffle, out_dims=480, name='conv_4_8_shuffle',
                              num_groups=3, stride=2, pad='SAME', k_size=3, use_add=False, use_avg_pool=True)'''

            ### add dilated convolution ###

            # conv stage 7_1
            conv_7_1 = self._conv_dilated_stage(input_tensor=conv_3_4_shuffle, k_size=3,
                                        out_dims=512, dilation = 2, name='conv7_1') #(8,18,50,1024)

            # conv stage 7_2
            conv_7_2 = self._conv_dilated_stage(input_tensor=conv_7_1, k_size=3,
                                        out_dims=512, dilation = 2, name='conv7_2')#(8,18,50,1024)

            # conv stage 5_3
            conv_7_3 = self._conv_dilated_stage(input_tensor=conv_7_2, k_size=3,
                                        out_dims=512, dilation = 2, name='conv7_3')#(8,18,50,1024)
            
            # added part of SCNN #
            
            # conv stage 7_4
            conv_7_4 = self._conv_dilated_stage(input_tensor=conv_7_3, k_size=3,
                                        out_dims=1024, dilation = 4, name='conv7_4')#(8,18,50,2048)

            # conv stage 7_5
            conv_7_5 = self._conv_stage(input_tensor=conv_7_4, k_size=1,
                                        out_dims=128, name='conv7_5') # 8 x 18 x 50 x 128
            print("conv_7_5=",conv_7_5)
            # add message passing #

            # top to down #
 
            feature_list_old = []
            feature_list_new = []
            for cnt in range(conv_7_5.get_shape().as_list()[1]):
                print("cnt=",cnt)
                feature_list_old.append(tf.expand_dims(conv_7_5[:, cnt, :, :], axis=1))#把conv_7_5从H维从高到低压入feature_list_old
                #print("feature_list_old=",feature_list_old)
            feature_list_new.append(tf.expand_dims(conv_7_5[:, 0, :, :], axis=1))#把conv_7_5从H维第1维压入feature_list_old
            print("feature_list_new=",feature_list_new)
            
            w1 = tf.get_variable('W1', [1, 9, 128, 128], initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5) ) ) )
            with tf.variable_scope("convs_8_1"):
                conv_8_1 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w1, [1, 1, 1, 1], 'SAME')), feature_list_old[1])#conv、relu后与前一个feature map相加
                feature_list_new.append(conv_8_1)

            for cnt in range(2, conv_7_5.get_shape().as_list()[1]):
                with tf.variable_scope("convs_8_1", reuse=True):
                    conv_8_1 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt-1], w1, [1, 1, 1, 1], 'SAME')), feature_list_old[cnt])
                    feature_list_new.append(conv_8_1)
            
            # down to top #
            feature_list_old = feature_list_new
            feature_list_new = []
            length = int(CFG.TRAIN.IMG_HEIGHT / 16) - 1
            feature_list_new.append(feature_list_old[length])

            w2 = tf.get_variable('W2', [1, 9, 128, 128], initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5) ) ) )
            with tf.variable_scope("convs_8_2"):
                conv_8_2 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[17], w2, [1, 1, 1, 1], 'SAME')), feature_list_old[16])
                feature_list_new.append(conv_8_2)

            for cnt in range(2, conv_7_5.get_shape().as_list()[1]):
                with tf.variable_scope("convs_8_2", reuse=True):
                    conv_8_2 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt-1], w2, [1, 1, 1, 1], 'SAME')), feature_list_old[17-cnt])
                    feature_list_new.append(conv_8_2)
                       
            feature_list_new.reverse() 
            
            processed_feature = tf.stack(feature_list_new, axis=1)# 8 x 36 x 1 x 100 x 128
            processed_feature = tf.squeeze(processed_feature)# 8 x 36 x 100 x 128
            print("processed_feature=",processed_feature)
            # left to right #

            feature_list_old = []
            feature_list_new = []
            for cnt in range(processed_feature.get_shape().as_list()[2]):
                feature_list_old.append(tf.expand_dims(processed_feature[:, :, cnt, :], axis=2))
            feature_list_new.append(tf.expand_dims(processed_feature[:, :, 0, :], axis=2))
            
            w3 = tf.get_variable('W3', [9, 1, 128, 128], initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5) ) ) )
            with tf.variable_scope("convs_8_3"):
                conv_8_3 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w3, [1, 1, 1, 1], 'SAME')), feature_list_old[1])
                feature_list_new.append(conv_8_3)
            

            for cnt in range(2, processed_feature.get_shape().as_list()[2]):
                with tf.variable_scope("convs_8_3", reuse=True):
                    conv_8_3 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt-1], w3, [1, 1, 1, 1], 'SAME')), feature_list_old[cnt])
                    feature_list_new.append(conv_8_3)

            # right to left #

            feature_list_old = feature_list_new
            print("feature_list_old=",feature_list_old)
            print("len=",len(feature_list_old))
            feature_list_new = []
            length = int(CFG.TRAIN.IMG_WIDTH / 16) - 1
            print("length=",length)
            feature_list_new.append(feature_list_old[length])

            w4 = tf.get_variable('W4', [9, 1, 128, 128], initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5) ) ) )
            with tf.variable_scope("convs_8_4"):
                conv_8_4 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[49], w4, [1, 1, 1, 1], 'SAME')), feature_list_old[48])
                feature_list_new.append(conv_8_4)

            for cnt in range(2, processed_feature.get_shape().as_list()[2]):
                with tf.variable_scope("convs_8_4", reuse=True):
                    conv_8_4 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt-1], w4, [1, 1, 1, 1], 'SAME')), feature_list_old[49-cnt])
                    feature_list_new.append(conv_8_4)
                       
            feature_list_new.reverse() 
            processed_feature = tf.stack(feature_list_new, axis=2)
            processed_feature = tf.squeeze(processed_feature)

            #######################

            dropout_output = self.dropout(processed_feature, 0.9, is_training=self._is_training, name='dropout') # 0.9 denotes the probability of being kept

            conv_output = self.conv2d(inputdata=dropout_output, out_channel=5,
                               kernel_size=1, use_bias=True, name='conv_9')#(8, 36, 100, 5)

            ret['prob_output'] = tf.image.resize_images(conv_output, [288, 800])#(8, 288, 800, 5)
     

            ### add lane existence prediction branch ###

            # spatial softmax #
            N, H, W, C = conv_output.get_shape().as_list()
            features = conv_output # N x H x W x C   (8, 36, 100, 5)
        
            softmax = tf.nn.softmax(features)#(8, 36, 100, 5)
    
            #avg_pool = self.avgpooling(softmax, kernel_size=2, stride=2)#(8, 18, 50, 5)

            reshape_output = tf.reshape(softmax, [N, -1])  #(8, 4500)

            fc_output = self.fullyconnect(reshape_output, 128)#(8, 128)
    
            relu_output = self.relu(inputdata=fc_output, name='relu6')          
            fc_output = self.fullyconnect(relu_output, 4)
            existence_output = fc_output

            ret['existence_output'] = existence_output #(8,4)
            
           
        return ret

if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.float32, shape=[8, 288, 800, 3], name='input')
    encoder = SHUFFLENETV1Encoder(phase=tf.constant('train', dtype=tf.string))
    ret = encoder.encode(a, name='encode')
    print(ret)
