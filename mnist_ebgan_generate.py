# -*- coding: utf-8 -*-
import numpy as np
import sugartensor as tf
import matplotlib.pyplot as plt
import ipdb

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 100
z_dim = 50


#
# create generator
#

# MNIST input tensor ( with QueueRunner )
data = tf.sg_data.Mnist(batch_size=batch_size)

# input images
x = data.train.image
image_shape = [28, 28, 1]

# random uniform seed
z = tf.random_uniform((batch_size, z_dim))

with tf.sg_context(name='generator', size=4, stride=2, act='relu', bn=True):
    # generator network
    gen = (z.sg_dense(dim=1024)
           .sg_dense(dim=7*7*128)
           .sg_reshape(shape=(-1, 7, 7, 128))
           .sg_upconv(dim=64)
           .sg_upconv(dim=1, act='sigmoid', bn=False)
           .sg_squeeze())



with tf.Session() as sess:
    tf.sg_init(sess)
    # restore parameters
    ipdb.set_trace()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))

    # run generator
    imgs = sess.run(gen)

    # plot generated images merged into a single png 
    imgs = imgs.reshape([batch_size] + image_shape)
    save_images(imgs, [13, 8], "outputImgs/generated.png")

    _, ax = plt.subplots(10, 10, sharex=True, sharey=True)
    for i in range(10):
        for j in range(10):
            ax[i][j].imshow(imgs[i * 10 + j], 'gray')
            ax[i][j].set_axis_off()
    plt.savefig('asset/train/sample.png', dpi=600)
    tf.sg_info('Sample image saved to "asset/train/sample.png"')
    plt.close()
