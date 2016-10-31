# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import ipdb

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 32   # batch size
z_dim = 50        # noise dimension
margin = 1        # max-margin for hinge loss
pt_weight = 0.1  # PT regularizer's weight

#
# inputs
#

# MNIST input tensor ( with QueueRunner )
data = tf.sg_data.Mnist(batch_size=batch_size)

# input images
x = data.train.image
image_shape = [28, 28]

#
# create generator
#

# random uniform seed
#z = tf.random_uniform((batch_size, z_dim))

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

# create real + fake image input
xx = tf.concat(0, [x, gen])

with tf.sg_context(name='discriminator', size=4, stride=2, act='leaky_relu'):
    disc = (xx.sg_conv(dim=64)
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
mse = tf.reduce_mean(tf.square(disc - xx), reduction_indices=[1, 2, 3])
mse_real, mse_fake = mse[:batch_size], mse[batch_size:]

loss_disc = mse_real + tf.maximum(margin - mse_fake, 0)   # discriminator loss
loss_gen = mse_fake + pt * pt_weight   # generator loss + PT regularizer

train_disc = tf.sg_optim(loss_disc, lr=0.001, category='discriminator')  # discriminator train ops
train_gen = tf.sg_optim(loss_gen, lr=0.001, category='generator')  # generator train ops


# +++++++++++++++++   add completion loss  ++++++++++++++++++++++
# perceptual loss
lam = 0.1
z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z')
zhat = tf.random_uniform((batch_size, z_dim))
mask = tf.placeholder(tf.float32, [None] + image_shape, name='mask')
images = tf.placeholder(tf.float32, [None] + image_shape, name="real_images")
percept_loss = loss_gen
context_loss = tf.reduce_sum(tf.contrib.layers.flatten(tf.abs(tf.mul(mask, gen) - tf.mul(mask, images))), 1)
complet_loss = context_loss + lam * percept_loss
# +++++++++++++++++   add completion loss  ++++++++++++++++++++++

# add summary
tf.sg_summary_loss(tf.identity(loss_disc, name='disc'))
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
#for i in xrange(nIter):
#  fd = {z: zhats, mask: mask, images: images} 
#  run = complete_loss, grad_complete_loss, G
#            # perform projected gradient descent on zhats
#            for i in xrange(config.nIter):
#                fd = {
#                    self.z: zhats,
#                    self.mask: batch_mask,
#                    self.images: batch_images,
#                }
#                run = [self.complete_loss, self.grad_complete_loss, self.G]
#                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)
#
#                v_prev = np.copy(v)
#                v = config.momentum*v - config.lr*g[0]
#                zhats += -config.momentum * v_prev + (1+config.momentum)*v
#                zhats = np.clip(zhats, -1, 1)
#
#                if i % 50 == 0:
#                    print(i, np.mean(loss[0:batchSz]))
#                    imgName = os.path.join(config.outDir,
#                                           'hats_imgs/{:04d}.png'.format(i))
#                    nRows = np.ceil(batchSz/8)
#                    nCols = 8
#                    save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)
#
#                    inv_masked_hat_images = np.multiply(G_imgs, 1.0-batch_mask)
#                    completeed = masked_images + inv_masked_hat_images
#                    imgName = os.path.join(config.outDir,
#                                           'completed/{:04d}.png'.format(i))
#                    save_images(completeed[:batchSz,:,:,:], [nRows,nCols], imgName)
# ++++++++++++++  Projected gradient descent on z +++++++++++++++

with tf.Session() as sess:
    tf.sg_init(sess)
    # restore parameters
    ipdb.set_trace()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))

    # run generator
    imgs = sess.run(gen)

    # plot result
    _, ax = plt.subplots(10, 10, sharex=True, sharey=True)
    for i in range(10):
        for j in range(10):
            ax[i][j].imshow(imgs[i * 10 + j], 'gray')
            ax[i][j].set_axis_off()
    plt.savefig('asset/train/sample.png', dpi=600)
    tf.sg_info('Sample image saved to "asset/train/sample.png"')
    plt.close()
# do training
#alt_train(log_interval=10, max_ep=30, ep_size=data.train.num_batch, early_stop=False)
