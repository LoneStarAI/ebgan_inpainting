# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import ipdb
import os


import sys
sys.path.append("../dcgan-completion.tensorflow")
from utils import *

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 32   # batch size
z_dim = 50        # noise dimension
margin = 1        # max-margin for hinge loss
pt_weight = 0.1  # PT regularizer's weight, default 0.1
lam = 0.1 # perceptual loss regularizer, default 0.1

#
# inputs
#

# MNIST input tensor ( with QueueRunner )
data = tf.sg_data.Mnist(batch_size=batch_size)

# input images
x = data.train.image
image_shape = [28, 28, 1]
image_size = image_shape[0] 

#
# create generator
#

# random uniform seed
#z = tf.random_uniform((batch_size, z_dim))
z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z')

with tf.sg_context(name='generator', size=4, stride=2, act='relu', bn=True):

    # generator network
    gen = (z.sg_dense(dim=1024)
           .sg_dense(dim=7*7*128)
           .sg_reshape(shape=(-1, 7, 7, 128))
           .sg_upconv(dim=64)
           .sg_upconv(dim=1, act='sigmoid', bn=False))

#
# create discriminator
#


with tf.sg_context(name='discriminator', size=4, stride=2, act='leaky_relu'):
    disc = (gen.sg_conv(dim=64)
            .sg_conv(dim=128)
            .sg_upconv(dim=64)
            .sg_upconv(dim=1, act='linear'))

#
# pull-away term ( PT ) regularizer
#

sample = gen.sg_flatten()
nom = tf.matmul(sample, tf.transpose(sample, perm=[1, 0]))
denom = tf.reduce_sum(tf.square(sample), reduction_indices=[1], keep_dims=True)
pt = tf.square(nom/denom)
pt -= tf.diag(tf.diag_part(pt))
pt = tf.reduce_sum(pt) / (batch_size * (batch_size - 1))

#
# loss & train ops
#

# mean squared errors
mse_fake = tf.reduce_mean(tf.square(disc - gen), reduction_indices=[1, 2, 3])
#mse_real, mse_fake = mse[:batch_size], mse[batch_size:]

#loss_disc = mse_real + tf.maximum(margin - mse_fake, 0)   # discriminator loss
loss_gen = mse_fake + pt * pt_weight   # generator loss + PT regularizer

#train_disc = tf.sg_optim(loss_disc, lr=0.001, category='discriminator')  # discriminator train ops
#train_gen = tf.sg_optim(loss_gen, lr=0.001, category='generator')  # generator train ops

batchSz = batch_size

# +++++++++++++++++   add completion loss  ++++++++++++++++++++++
# perceptual loss
zhat = tf.random_uniform((batch_size, z_dim))
mask_p = tf.placeholder(tf.float32, [None] + image_shape, name='mask')
images = tf.placeholder(tf.float32, [None] + image_shape, name="real_images")
percept_loss = loss_gen
context_loss = tf.reduce_sum(tf.contrib.layers.flatten(tf.abs(tf.mul(mask_p, gen) - tf.mul(mask_p, images))), 1)
complete_loss = context_loss + lam * percept_loss
grad_complete_loss = tf.gradients(complete_loss, z)
# +++++++++++++++++   add completion loss  ++++++++++++++++++++++

# add summary
#tf.sg_summary_loss(tf.identity(loss_disc, name='disc'))
tf.sg_summary_loss(tf.identity(loss_gen, name='gen'))
tf.sg_summary_image(gen)

#
# training
#

# def alternate training func
@tf.sg_train_func
def alt_train(sess, opt):
    l_disc = sess.run([loss_disc, train_disc])[0]  # training discriminator
    l_gen = sess.run([loss_gen, train_gen])[0]  # training generator
    return np.mean(l_disc) + np.mean(l_gen)


# ++++++++++++++  Projected gradient descent on z +++++++++++++++
def complete(sess, maskType="center", lr=0.001, momentum=0.9, outDir="outputImgs", nIter=1000):
   tf.initialize_all_variables().run()
   if os.path.exists(outDir):
       import shutil
       shutil.rmtree(outDir)
    
   os.makedirs(os.path.join(outDir, 'hats_imgs'))
   os.makedirs(os.path.join(outDir, 'completed'))

   saver = tf.train.Saver()
   saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))
   
   # generate masks
   if maskType == 'random':
       fraction_masked = 0.4
       mask = np.ones(image_shape)
       mask[np.random.random(image_shape[:2]) < fraction_masked] = 0.0
   elif maskType == 'center':
       #1-2 * scale are masked:  larger scale masked less pixels
       scale = 0.25
       assert(scale <= 0.5)
       mask = np.ones(image_shape)
       l = int(image_size*scale)
       u = int(image_size*(1.0-scale))
       mask[l:u, l:u, :] = 0.0
   elif maskType == 'left':
       mask = np.ones(image_shape)
       c = image_size // 2
       mask[:,:c,:] = 0.0
   elif maskType == 'full':
       mask = np.ones(image_shape)
   else:
       assert(False)

   # get images from default tensorflow mnist data class
   from tensorflow.examples.tutorials.mnist import input_data
   mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
   t_data = mnist.test.images[:batch_size, :]
   batch_images = t_data.reshape((batch_size, 28, 28, 1))
   nImgs = batch_images.shape[0]

   # get images
   #base_dir = "../dcgan-completion.tensorflow/data/CelebA/test/"
   #img_lst = os.listdir("../dcgan-completion.tensorflow/data/CelebA/test")
   #img_path = [base_dir + img_name for img_name in img_lst]
   #nImgs = len(img_lst)

   batch_idxs = int(np.ceil(nImgs/float(batch_size)))
   batchSz = batch_size
   for idx in xrange(0, batch_idxs):
       #l = idx*batch_size
       #u = min((idx+1)*batch_size, nImgs)
       #batchSz = u-l
       ## load in the test imgs
       #batch = [get_image(batch_file, image_size)
       #         for batch_file in img_path]
       #batch_images = np.array(batch).astype(np.float32)
       ## do padding if samples are not enough to form a batch
       #if batchSz < batch_size:
       #    print(batchSz)
       #    padSz = ((0, int(batch_size-batchSz)), (0,0), (0,0), (0,0))
       #    batch_images = np.pad(batch_images, padSz, 'constant')
       #    batch_images = batch_images.astype(np.float32)

       # initilization
       batch_mask = np.resize(mask, [batch_size] + image_shape)
       zhats = np.random.uniform(0, 1, size=(batch_size, z_dim))
       #zhats = tf.random_uniform((batch_size, z_dim))
       v = 0

       nRows = np.ceil(batchSz/8.)
       nCols = 8
       save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
                   os.path.join(outDir, 'before.png'))
       masked_images = np.multiply(batch_images, batch_mask)
       save_images(masked_images[:batchSz,:,:,:], [nRows,nCols],
                   os.path.join(outDir, 'masked.png'))

       # perform projected gradient descent on zhats
       #z1 = tf.random_uniform((batch_size, z_dim))
       #imgs = sess.run(gen, feed_dict={z: zhats})
       for i in xrange(nIter):
           fd = {
               z: zhats,
               mask_p: batch_mask,
               images: batch_images,
           }
           #run_metadata = tf.RunMetadata()
           run = [complete_loss, grad_complete_loss, gen]
           #loss, g, G_imgs = sess.run(run, feed_dict=fd, 
           #                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
           #                     run_metadata=run_metadata)
           loss, g, G_imgs = sess.run(run, feed_dict=fd)

           #save_images(G_imgs, [4, 8], "outputImgs/generated_zhats.png")
           v_prev = np.copy(v)
           v = momentum*v - lr*g[0]
           zhats += -momentum * v_prev + (1+momentum)*v
           zhats = np.clip(zhats, 0, 1)

           if i % 50 == 0:
               print(i, np.mean(loss[0:batchSz]))
               imgName = os.path.join(outDir,
                                      'hats_imgs/{:04d}.png'.format(i))
               nRows = np.ceil(batchSz/8.)
               nCols = 8
               # obtain and save the generated images
               save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)
               save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], "outputImgs/hats_img_final.png")

               inv_masked_hat_images = np.multiply(G_imgs, 1.0-batch_mask)
               # obtain and save the completed imgs
               completeed = masked_images + inv_masked_hat_images
               imgName = os.path.join(outDir,
                                      'completed/{:04d}.png'.format(i))
               save_images(completeed[:batchSz,:,:,:], [nRows,nCols], imgName)
               save_images(completeed[:batchSz,:,:,:], [nRows,nCols], "outputImgs/completed_final.png")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    tf.sg_init(sess)
    complete(sess, maskType="center", lr=0.001, momentum=0.9, outDir="outputImgs", nIter=6000)
    # run generator
    #imgs = sess.run(gen)

# do training
#alt_train(log_interval=10, max_ep=30, ep_size=data.train.num_batch, early_stop=False)
