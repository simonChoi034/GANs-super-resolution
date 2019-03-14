import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dense, Flatten


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

    def call(self, x, training=None, mask=None):
        ## input conv
        x = self.conv2d(input=x, filters=64, kernel_size=9)

        x = self.conv2d(input=x, filters=64, kernel_size=3)
        x = self.conv2d(input=x, filters=128, kernel_size=3)
        x = self.conv2d(input=x, filters=128, kernel_size=3)
        x = self.conv2d(input=x, filters=256, kernel_size=3)
        x = self.conv2d(input=x, filters=256, kernel_size=3)
        x = self.conv2d(input=x, filters=512, kernel_size=3)
        x = self.conv2d(input=x, filters=512, kernel_size=3)

        ## upacale image
        x = self.conv2d_transpose(input=x, filters=64, kernel_size=9, strides=2)
        x = self.conv2d(input=x, filters=64, kernel_size=9)
        x = self.conv2d(input=x, filters=64, kernel_size=9)

        ## output conv
        output_layer = Conv2D(filters=3, kernel_size=9, padding='same', activation='sigmoid')(x)
        return output_layer

    def conv2d(self, input, filters, kernel_size, strides=1):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    def conv2d_transpose(self, input, filters, kernel_size, strides=1):
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(input)
        x = BatchNormalization()(x)
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

