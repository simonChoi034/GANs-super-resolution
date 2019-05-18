import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Concatenate, Dropout, Input, ZeroPadding2D
from tensorflow.keras import Sequential

OUTPUT_CHANNELS = 3


def downsample(filters, size, apply_batchnorm=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = Sequential()
    result.add(
        Conv2D(filters, size, strides=2, padding='same',
               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False, apply_batchnorm=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = Sequential()
    result.add(
        Conv2DTranspose(filters, size, strides=2,
                        padding='same',
                        kernel_initializer=initializer,
                        use_bias=False))

    if apply_batchnorm:
        result.add(BatchNormalization())

    if apply_dropout:
        result.add(Dropout(0.5))

    result.add(LeakyReLU())

    return result


def pixel_shuffle(x, filters, size):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = Conv2D(filters, size, strides=1,
                    padding='same',
                    kernel_initializer=initializer,
                    use_bias=False)(x)

    result = tf.nn.depth_to_space(result, 2)
    result = LeakyReLU()(result)

    return result


def conv2d(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = Sequential()
    result.add(Conv2D(filters, size, strides=1, padding='same', kernel_initializer=initializer, use_bias=False))

    if apply_dropout:
        result.add(Dropout(0.5))

    result.add(LeakyReLU())

    return result


def Generator():
    down_stack = [
        downsample(64, 4),  # (bs, 180, 320, 128)
        downsample(256, 4),  # (bs, 90, 160, 256)
        downsample(512, 4),  # (bs, 45, 80, 512)
        conv2d(512, 4),
        conv2d(512, 4),
    ]

    up_stack = [
        conv2d(512, 4, apply_dropout=True),
        conv2d(512, 4, apply_dropout=True),
        upsample(256, 4),  # (bs, 90, 160, 256)
        upsample(128, 4),  # (bs, 180, 320, 128)
    ]

    concat = Concatenate()

    inputs = Input(shape=[None, None, 3])
    x = inputs

    # Downsampling
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    # sub-pixel convolution
    x = pixel_shuffle(x, 64, 4)
    x = pixel_shuffle(x, 64, 4)

    initializer = tf.random_normal_initializer(0., 0.02)
    x = Conv2D(OUTPUT_CHANNELS, 9, strides=1, padding='same', kernel_initializer=initializer, use_bias=False,
               activation='tanh')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = Input(shape=[None, None, 3], name='input_image')
    tar = Input(shape=[None, None, 3], name='target_image')

    x = Concatenate()([inp, tar])

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = ZeroPadding2D()(down3)
    conv = Conv2D(512, 4, strides=1,
                  kernel_initializer=initializer,
                  use_bias=False)(zero_pad1)

    batchnorm1 = BatchNormalization()(conv)

    leaky_relu = LeakyReLU()(batchnorm1)

    zero_pad2 = ZeroPadding2D()(leaky_relu)

    last = Conv2D(1, 4, strides=1,
                  kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
