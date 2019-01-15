import tensorflow as tf


def down_sample_conv(x, filter_size):
    out = tf.layers.conv2d(
        x,
        filters=filter_size,
        kernel_size=[1, 1],
        strides=(2, 2),
        padding='same'
    )
    out = tf.layers.batch_normalization(out)
    return out


def residual_block(x, filter_size, down_sample=None):
    residual = x
    # downscale
    strides = (1, 1)
    if down_sample:
        strides = (2, 2)
        residual = down_sample_conv(residual, filter_size)

    # 1st conv
    out = tf.layers.conv2d(
        x,
        filters=filter_size,
        kernel_size=[3, 3],
        strides=strides,
        padding='same'
    )
    out = tf.layers.batch_normalization(out)
    out = tf.nn.leaky_relu(out)

    # 2nd conv
    out = tf.layers.conv2d(
        out,
        filters=filter_size,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same'
    )
    out = tf.layers.batch_normalization(out)
    # Resnet shortcut
    out += residual
    out = tf.nn.leaky_relu(out)

    return out


def up_sample_conv(x, filter_size):
    out = tf.layers.conv2d_transpose(
        x,
        filters=filter_size,
        kernel_size=[1, 1],
        strides=(2, 2),
        padding='same'
    )
    out = tf.layers.batch_normalization(out)
    return out


def residual_block_transpose(x, filter_size, up_sample=None):
    residual = x
    # upscale
    strides = (1, 1)
    if up_sample:
        strides = (2, 2)
        residual = up_sample_conv(residual, filter_size)

    # 1st conv_transpose
    out = tf.layers.conv2d_transpose(
        x,
        filters=filter_size,
        kernel_size=[3, 3],
        strides=strides,
        padding='same'
    )
    out = tf.layers.batch_normalization(out)
    out = tf.nn.leaky_relu(out)

    # 2nd conv_transpose
    out = tf.layers.conv2d_transpose(
        out,
        filters=filter_size,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same'
    )
    out = tf.layers.batch_normalization(out)
    # Resnet shortcut
    out += residual
    out = tf.nn.leaky_relu(out)

    return out


def discriminator(img, reuse_variable=None):
    with tf.variable_scope('discriminator', reuse=reuse_variable) as scope:
        # ResNet
        out = tf.layers.conv2d(
            img,
            filters=64,
            kernel_size=[7, 7],
            strides=(2, 2),
            padding='same'
        )
        out = tf.layers.batch_normalization(out)
        out = tf.nn.leaky_relu(out)

        ###
        for i in range(0, 2):
            out = residual_block(out, 64)
        ###
        for i in range(0, 2):
            out = residual_block(out, 128, i == 0)
        ###
        for i in range(0, 2):
            out = residual_block(out, 256, i == 0)
        ###
        for i in range(0, 2):
            out = residual_block(out, 512, i == 0)

        # global average pooling
        out = tf.reduce_mean(out, axis=[1, 2])

        out_flat = tf.layers.flatten(out)
        out_flat = tf.layers.dense(out_flat, 1024, activation=tf.nn.leaky_relu)
        out_flat = tf.layers.dense(out_flat, 1)

        return out_flat


def generator(img):
    with tf.variable_scope('generator') as scope:
        # Convolutional autoencoder
        with tf.variable_scope('encoder'):
            out = tf.layers.conv2d(
                img,
                filters=64,
                kernel_size=[3, 3],
                strides=(1, 1),
                padding='same'
            )
            out = tf.layers.batch_normalization(out)
            out = tf.nn.leaky_relu(out)

            ###
            for i in range(0, 2):
                out = residual_block(out, 64)
            ###
            for i in range(0, 2):
                out = residual_block(out, 128, i == 0)
            ###
            for i in range(0, 2):
                out = residual_block(out, 256, i == 0)
            ###
            for i in range(0, 2):
                out = residual_block(out, 512, i == 0)

        with tf.variable_scope('decoder') as scope:
            ###
            for i in range(0, 2):
                out = residual_block_transpose(out, 512, i == 0)
            ###
            for i in range(0, 2):
                out = residual_block_transpose(out, 256, i == 0)
            ###
            for i in range(0, 2):
                out = residual_block_transpose(out, 128, i == 0)
            ###
            for i in range(0, 2):
                out = residual_block_transpose(out, 64, i == 0)

            ###
            for i in range(0, 2):
                out = residual_block_transpose(out, 64)

            # output image
            decode = tf.layers.conv2d(
                out,
                filters=3,
                kernel_size=[3, 3],
                strides=(1, 1),
                padding='same',
                activation=tf.nn.tanh
            )
            return decode