import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def res_block(input_x, out_channels=64, k=3, s=1, scope='res_block'):
    with tf.variable_scope(scope) as scope:
        x = input_x
        input_x = slim.conv2d_transpose(input_x, out_channels, k, s)
        input_x = slim.batch_norm(input_x, scope='bn1')
        input_x = tf.nn.relu(input_x)
        input_x = slim.conv2d_transpose(input_x, out_channels, k, s)
        input_x = slim.batch_norm(input_x, scope='bn2')
    
    return x+input_x
    
def pixel_shuffle_layer(x, r, n_split):
    def PS(x, r):
        bs, a, b, c = x.get_shape().as_list()
        x = tf.reshape(x, (bs, a, b, r, r))
        x = tf.transpose(x, [0,1,2,4,3])
        x = tf.split(x, a, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        x = tf.split(x, b, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        return tf.reshape(x, (bs, a*r, b*r, 1))

    xc = tf.split(x, n_split, 3)
    return tf.concat([PS(x_, r) for x_ in xc], 3)

def down_sample_layer(input_x):
    K = 4
    arr = np.zeros([K, K, 3, 3])
    arr[:, :, 0, 0] = 1.0 / K ** 2
    arr[:, :, 1, 1] = 1.0 / K ** 2
    arr[:, :, 2, 2] = 1.0 / K ** 2
    weight = tf.constant(arr, dtype=tf.float32)
    downscaled = tf.nn.conv2d(
        input_x, weight, strides=[1, K, K, 1], padding='SAME')
    return downscaled

def leaky_relu(input_x, negative_slop=0.2):
    return tf.maximum(negative_slop*input_x, input_x)

def PSNR(real, fake):
    mse = tf.reduce_mean(tf.square(127.5*(real-fake)+127.5),axis=(-3,-2,-1))
    psnr = tf.reduce_mean(10 * (tf.log(255*255 / tf.sqrt(mse)) / np.log(10)))
    return psnr
