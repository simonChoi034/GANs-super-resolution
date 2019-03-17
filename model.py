import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, LeakyReLU, Dense, Flatten


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

    def call(self, x, training=None, mask=None):
        ## encoder
        x = Conv2D(filters=64, kernel_size=9, padding='same')(x)
        x = LeakyReLU()(x)

        x = self.residual_block(x, filters=128, strides=2)
        x = self.residual_block(x, filters=256)

        ## decoder
        x = self.residual_block(x, filters=256)
        x = tf.nn.depth_to_space(x, 2)
        x = self.residual_block(x, filters=128)
        x = tf.nn.depth_to_space(x, 2)
        x = self.residual_block(x, filters=64)

        ## output conv
        output_layer = Conv2D(filters=3, kernel_size=9, padding='same', activation='tanh')(x)
        return output_layer

    def conv2d(self, input, filters, kernel_size, strides=1):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(input)
        x = LeakyReLU()(x)
        return x

    def residual_block(self, input, filters, strides=1):
        residual = self.conv2d(input, filters=filters, kernel_size=1, strides=strides)
        x = self.conv2d(input, filters=filters, kernel_size=3, strides=strides)

        x = Conv2D(filters=filters, kernel_size=3, padding='same')(x)
        x += residual
        x = LeakyReLU()(x)

        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

    def call(self, x, training=None, mask=None):
        x = self.conv2d(input=x, filters=64, kernel_size=3)
        x = self.conv2d(input=x, filters=64, kernel_size=3, strides=2)
        x = self.conv2d(input=x, filters=128, kernel_size=3)
        x = self.conv2d(input=x, filters=128, kernel_size=3, strides=2)
        x = self.conv2d(input=x, filters=256, kernel_size=3)
        x = self.conv2d(input=x, filters=256, kernel_size=3, strides=2)
        x = self.conv2d(input=x, filters=512, kernel_size=3)
        x = self.conv2d(input=x, filters=512, kernel_size=3, strides=2)

        x = Flatten()(x)

        x = Dense(32, activation='relu')(x)
        x = Dense(1)(x)

        return x

    def conv2d(self, input, filters, kernel_size, strides=1):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

