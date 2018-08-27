"""Contains a model definition for YOLO_v2:

   YOLO9000: Better, Faster, Stronger,
   Joseph Redmon, Ali Farhadi. CVPR 2017.

Usage:

   model_input  = tf.placeholder(tf.float32, shape=(None, 608, 608, 3))
   model_output = yolo_models.build_yolo_v2(model_input, 5, 80)

"""

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

def build_yolo_v2(inp, num_priors=5, num_classes=80):


    xavier = tf.contrib.layers.xavier_initializer()

    def yolo_layer(inp, kernels, biases, moving_mean, moving_variance, gamma, pad, leaky):
    
        conv_kernels = tf.Variable(xavier(kernels.shape), name='ConvKernels')
        conv_biases  = tf.Variable(biases, name='ConvBiases')
    
        x = tf.pad(inp, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        x = tf.nn.conv2d(x, conv_kernels, padding = 'VALID', strides = [1, 1, 1, 1])
    
        if (moving_mean is not None) and (moving_variance is not None) and (gamma is not None):
            conv_bn_initializers = dict({'moving_mean': tf.constant_initializer(moving_mean),
                                         'moving_variance': tf.constant_initializer(moving_variance),
                                         'gamma': tf.constant_initializer(gamma)})
            conv_bn_args = dict({'center' : False, 'scale' : True,
                                 'epsilon': 1e-5, 
                                 'updates_collections' : None,
                                 'is_training': False,
                                 'param_initializers': conv_bn_initializers})
            x = slim.batch_norm(x, **conv_bn_args)
    
        x = tf.nn.bias_add(x, conv_biases)
    
        if leaky:
            x = tf.maximum(.1 * x, x)
    
        return x
    
    
    # ---------------------------------------------------------------------------------------

    with tf.variable_scope('YOLOv2'): 
    
        conv1_kernels = np.zeros((3, 3, 3, 32), dtype=np.float32)
        conv1_biases  = np.zeros((32,), dtype=np.float32)
        conv1_moving_mean = np.zeros((conv1_kernels.shape[-1],))
        conv1_moving_variance = np.ones((conv1_kernels.shape[-1],))
        conv1_gamma = np.ones((conv1_kernels.shape[-1],))
        
        x = yolo_layer(inp, conv1_kernels, conv1_biases, conv1_moving_mean, conv1_moving_variance, conv1_gamma, 1, True)
        x = tf.nn.max_pool(x, padding = 'SAME', ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1])
        
        # ---------------------------------------------------------------------------------------
        
        conv2_kernels = np.zeros(shape=(3, 3, 32, 64), dtype=np.float32)
        conv2_biases  = np.zeros(shape=(64,), dtype=np.float32)
        conv2_moving_mean = np.zeros((conv2_kernels.shape[-1],))
        conv2_moving_variance = np.ones((conv2_kernels.shape[-1],))
        conv2_gamma = np.ones((conv2_kernels.shape[-1],))
        
        x = yolo_layer(x, conv2_kernels, conv2_biases, conv2_moving_mean, conv2_moving_variance, conv2_gamma, 1, True)
        x = tf.nn.max_pool(x, padding = 'SAME', ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1])
        
        # ---------------------------------------------------------------------------------------
        
        conv3_kernels = np.zeros(shape=(3, 3, 64, 128), dtype=np.float32)
        conv3_biases  = np.zeros(shape=(128,), dtype=np.float32)
        conv3_moving_mean = np.zeros((conv3_kernels.shape[-1],))
        conv3_moving_variance = np.ones((conv3_kernels.shape[-1],))
        conv3_gamma = np.ones((conv3_kernels.shape[-1],))
        
        x = yolo_layer(x, conv3_kernels, conv3_biases, conv3_moving_mean, conv3_moving_variance, conv3_gamma, 1, True)
        
        # ---------------------------------------------------------------------------------------
        
        conv4_kernels = np.zeros(shape=(1, 1, 128, 64), dtype=np.float32)
        conv4_biases  = np.zeros(shape=(64,), dtype=np.float32)
        conv4_moving_mean = np.zeros((conv4_kernels.shape[-1],))
        conv4_moving_variance = np.ones((conv4_kernels.shape[-1],))
        conv4_gamma = np.ones((conv4_kernels.shape[-1],))
        
        x = yolo_layer(x, conv4_kernels, conv4_biases, conv4_moving_mean, conv4_moving_variance, conv4_gamma, 0, True)
        
        # ---------------------------------------------------------------------------------------
        
        conv5_kernels = np.zeros(shape=(3, 3, 64, 128), dtype=np.float32)
        conv5_biases  = np.zeros(shape=(128,), dtype=np.float32)
        conv5_moving_mean = np.zeros((conv5_kernels.shape[-1],))
        conv5_moving_variance = np.ones((conv5_kernels.shape[-1],))
        conv5_gamma = np.ones((conv5_kernels.shape[-1],))
        
        x = yolo_layer(x, conv5_kernels, conv5_biases, conv5_moving_mean, conv5_moving_variance, conv5_gamma, 1, True)
        x = tf.nn.max_pool(x, padding = 'SAME', ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1])
        
        # ---------------------------------------------------------------------------------------
        
        conv6_kernels = np.zeros(shape=(3, 3, 128, 256), dtype=np.float32)
        conv6_biases  = np.zeros(shape=(256,), dtype=np.float32)
        conv6_moving_mean = np.zeros((conv6_kernels.shape[-1],))
        conv6_moving_variance = np.ones((conv6_kernels.shape[-1],))
        conv6_gamma = np.ones((conv6_kernels.shape[-1],))
        
        x = yolo_layer(x, conv6_kernels, conv6_biases, conv6_moving_mean, conv6_moving_variance, conv6_gamma, 1, True)
        
        # ---------------------------------------------------------------------------------------
        
        conv7_kernels = np.zeros(shape=(1, 1, 256, 128), dtype=np.float32)
        conv7_biases  = np.zeros(shape=(128,), dtype=np.float32)
        conv7_moving_mean = np.zeros((conv7_kernels.shape[-1],))
        conv7_moving_variance = np.ones((conv7_kernels.shape[-1],))
        conv7_gamma = np.ones((conv7_kernels.shape[-1],))
        
        x = yolo_layer(x, conv7_kernels, conv7_biases, conv7_moving_mean, conv7_moving_variance, conv7_gamma, 0, True)
        
        # ---------------------------------------------------------------------------------------
        
        conv8_kernels = np.zeros(shape=(3, 3, 128, 256), dtype=np.float32)
        conv8_biases  = np.zeros(shape=(256,), dtype=np.float32)
        conv8_moving_mean = np.zeros((conv8_kernels.shape[-1],))
        conv8_moving_variance = np.ones((conv8_kernels.shape[-1],))
        conv8_gamma = np.ones((conv8_kernels.shape[-1],))
        
        x = yolo_layer(x, conv8_kernels, conv8_biases, conv8_moving_mean, conv8_moving_variance, conv8_gamma, 1, True)
        x = tf.nn.max_pool(x, padding = 'SAME', ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1])
        
        # ---------------------------------------------------------------------------------------
        
        conv9_kernels = np.zeros(shape=(3, 3, 256, 512), dtype=np.float32)
        conv9_biases  = np.zeros(shape=(512,), dtype=np.float32)
        conv9_moving_mean = np.zeros((conv9_kernels.shape[-1],))
        conv9_moving_variance = np.ones((conv9_kernels.shape[-1],))
        conv9_gamma = np.ones((conv9_kernels.shape[-1],))
        
        x = yolo_layer(x, conv9_kernels, conv9_biases, conv9_moving_mean, conv9_moving_variance, conv9_gamma, 1, True)
        
        # ---------------------------------------------------------------------------------------
        
        conv10_kernels = np.zeros(shape=(1, 1, 512, 256), dtype=np.float32)
        conv10_biases  = np.zeros(shape=(256,), dtype=np.float32)
        conv10_moving_mean = np.zeros((conv10_kernels.shape[-1],))
        conv10_moving_variance = np.ones((conv10_kernels.shape[-1],))
        conv10_gamma = np.ones((conv10_kernels.shape[-1],))
        
        x = yolo_layer(x, conv10_kernels, conv10_biases, conv10_moving_mean, conv10_moving_variance, conv10_gamma, 0, True)
        
        # ---------------------------------------------------------------------------------------
        
        conv11_kernels = np.zeros(shape=(3, 3, 256, 512), dtype=np.float32)
        conv11_biases  = np.zeros(shape=(512,), dtype=np.float32)
        conv11_moving_mean = np.zeros((conv11_kernels.shape[-1],))
        conv11_moving_variance = np.ones((conv11_kernels.shape[-1],))
        conv11_gamma = np.ones((conv11_kernels.shape[-1],))
        
        x = yolo_layer(x, conv11_kernels, conv11_biases, conv11_moving_mean, conv11_moving_variance, conv11_gamma, 1, True)
        
        # ---------------------------------------------------------------------------------------
        
        conv12_kernels = np.zeros(shape=(1, 1, 512, 256), dtype=np.float32)
        conv12_biases  = np.zeros(shape=(256,), dtype=np.float32)
        conv12_moving_mean = np.zeros((conv12_kernels.shape[-1],))
        conv12_moving_variance = np.ones((conv12_kernels.shape[-1],))
        conv12_gamma = np.ones((conv12_kernels.shape[-1],))
        
        x = yolo_layer(x, conv12_kernels, conv12_biases, conv12_moving_mean, conv12_moving_variance, conv12_gamma, 0, True)
        
        # ---------------------------------------------------------------------------------------
        
        conv13_kernels = np.zeros(shape=(3, 3, 256, 512), dtype=np.float32)
        conv13_biases  = np.zeros(shape=(512,), dtype=np.float32)
        conv13_moving_mean = np.zeros((conv13_kernels.shape[-1],))
        conv13_moving_variance = np.ones((conv13_kernels.shape[-1],))
        conv13_gamma = np.ones((conv13_kernels.shape[-1],))
        
        x13 = yolo_layer(x, conv13_kernels, conv13_biases, conv13_moving_mean, conv13_moving_variance, conv13_gamma, 1, True)
        x = tf.nn.max_pool(x13, padding = 'SAME', ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1])
        
        # ---------------------------------------------------------------------------------------
        
        conv14_kernels = np.zeros(shape=(3, 3, 512, 1024), dtype=np.float32)
        conv14_biases  = np.zeros(shape=(1024,), dtype=np.float32)
        conv14_moving_mean = np.zeros((conv14_kernels.shape[-1],))
        conv14_moving_variance = np.ones((conv14_kernels.shape[-1],))
        conv14_gamma = np.ones((conv14_kernels.shape[-1],))
        
        x = yolo_layer(x, conv14_kernels, conv14_biases, conv14_moving_mean, conv14_moving_variance, conv14_gamma, 1, True)
        
        # ---------------------------------------------------------------------------------------
        
        conv15_kernels = np.zeros(shape=(1, 1, 1024, 512), dtype=np.float32)
        conv15_biases  = np.zeros(shape=(512,), dtype=np.float32)
        conv15_moving_mean = np.zeros((conv15_kernels.shape[-1],))
        conv15_moving_variance = np.ones((conv15_kernels.shape[-1],))
        conv15_gamma = np.ones((conv15_kernels.shape[-1],))
        
        x = yolo_layer(x, conv15_kernels, conv15_biases, conv15_moving_mean, conv15_moving_variance, conv15_gamma, 0, True)
        
        # ---------------------------------------------------------------------------------------
        
        conv16_kernels = np.zeros(shape=(3, 3, 512, 1024), dtype=np.float32)
        conv16_biases  = np.zeros(shape=(1024,), dtype=np.float32)
        conv16_moving_mean = np.zeros((conv16_kernels.shape[-1],))
        conv16_moving_variance = np.ones((conv16_kernels.shape[-1],))
        conv16_gamma = np.ones((conv16_kernels.shape[-1],))
        
        x = yolo_layer(x, conv16_kernels, conv16_biases, conv16_moving_mean, conv16_moving_variance, conv16_gamma, 1, True)
        
        # ---------------------------------------------------------------------------------------
        
        conv17_kernels = np.zeros(shape=(1, 1, 1024, 512), dtype=np.float32)
        conv17_biases  = np.zeros(shape=(512,), dtype=np.float32)
        conv17_moving_mean = np.zeros((conv17_kernels.shape[-1],))
        conv17_moving_variance = np.ones((conv17_kernels.shape[-1],))
        conv17_gamma = np.ones((conv17_kernels.shape[-1],))
        
        x = yolo_layer(x, conv17_kernels, conv17_biases, conv17_moving_mean, conv17_moving_variance, conv17_gamma, 0, True)
        
        # ---------------------------------------------------------------------------------------
        
        conv18_kernels = np.zeros(shape=(3, 3, 512, 1024), dtype=np.float32)
        conv18_biases  = np.zeros(shape=(1024,), dtype=np.float32)
        conv18_moving_mean = np.zeros((conv18_kernels.shape[-1],))
        conv18_moving_variance = np.ones((conv18_kernels.shape[-1],))
        conv18_gamma = np.ones((conv18_kernels.shape[-1],))
        
        x = yolo_layer(x, conv18_kernels, conv18_biases, conv18_moving_mean, conv18_moving_variance, conv18_gamma, 1, True)
        
        # ---------------------------------------------------------------------------------------
        
        conv19_kernels = np.zeros(shape=(3, 3, 1024, 1024), dtype=np.float32)
        conv19_biases  = np.zeros(shape=(1024,), dtype=np.float32)
        conv19_moving_mean = np.zeros((conv19_kernels.shape[-1],))
        conv19_moving_variance = np.ones((conv19_kernels.shape[-1],))
        conv19_gamma = np.ones((conv19_kernels.shape[-1],))
        
        x = yolo_layer(x, conv19_kernels, conv19_biases, conv19_moving_mean, conv19_moving_variance, conv19_gamma, 1, True)
        
        # ---------------------------------------------------------------------------------------
        
        conv20_kernels = np.zeros(shape=(3, 3, 1024, 1024), dtype=np.float32)
        conv20_biases  = np.zeros(shape=(1024,), dtype=np.float32)
        conv20_moving_mean = np.zeros((conv20_kernels.shape[-1],))
        conv20_moving_variance = np.ones((conv20_kernels.shape[-1],))
        conv20_gamma = np.ones((conv20_kernels.shape[-1],))
        
        x20 = yolo_layer(x, conv20_kernels, conv20_biases, conv20_moving_mean, conv20_moving_variance, conv20_gamma, 1, True)
        
        # ---------------------------------------------------------------------------------------
        
        x = tf.concat([x13], 3)
        
        # ---------------------------------------------------------------------------------------
        
        conv21_kernels = np.zeros(shape=(1, 1, 512, 64), dtype=np.float32)
        conv21_biases  = np.zeros(shape=(64,), dtype=np.float32)
        conv21_moving_mean = np.zeros((conv21_kernels.shape[-1],))
        conv21_moving_variance = np.ones((conv21_kernels.shape[-1],))
        conv21_gamma = np.ones((conv21_kernels.shape[-1],))
        
        x = yolo_layer(x, conv21_kernels, conv21_biases, conv21_moving_mean, conv21_moving_variance, conv21_gamma, 0, True)
        
        # ---------------------------------------------------------------------------------------
        
        x = tf.extract_image_patches(x, [1,2,2,1], [1,2,2,1], [1,1,1,1], 'VALID')
        x = tf.concat([x, x20], 3)
        
        # ---------------------------------------------------------------------------------------
        
        conv22_kernels = np.zeros(shape=(3, 3, 1280, 1024), dtype=np.float32)
        conv22_biases  = np.zeros(shape=(1024,), dtype=np.float32)
        conv22_moving_mean = np.zeros((conv22_kernels.shape[-1],))
        conv22_moving_variance = np.ones((conv22_kernels.shape[-1],))
        conv22_gamma = np.ones((conv22_kernels.shape[-1],))
        
        x = yolo_layer(x, conv22_kernels, conv22_biases, conv22_moving_mean, conv22_moving_variance, conv22_gamma, 1, True)
        
        # ---------------------------------------------------------------------------------------
        
        conv23_kernels = np.zeros(shape=(1, 1, 1024, (4+1+num_classes)*num_priors), dtype=np.float32)
        conv23_biases  = np.zeros(shape=((4+1+num_classes)*num_priors,), dtype=np.float32)
        
        x = yolo_layer(x, conv23_kernels, conv23_biases, None, None, None, 0, False)
        
        return x


if __name__ == "__main__":
    model_input  = tf.placeholder(tf.float32, shape=(None, 608, 608, 3))
    model_output = build_yolo_v2(model_input, 5, 80)
    print('In  shape: '+str(model_input.get_shape()))
    print('Out shape: '+str(model_output.get_shape()))
