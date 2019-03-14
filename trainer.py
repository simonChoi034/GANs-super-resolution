import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as cross_entropy
import numpy as np
import skimage
from skimage import io
from skimage.transform import downscale_local_mean
from skimage.color import rgba2rgb, gray2rgb
import matplotlib.pyplot as plt
import os
from glob import glob
from model import Generator, Discriminator

DATASET_PATH = '/Users/Simon/tf_files/dataset/image/anime'
images_in_memory = 32
learning_rate = 1e-5


def expand_dims(array):
    # batch size = 1
    if array.ndim == 3:
        return np.expand_dims(array, axis=0)
    return array


def image2rgb(image):
    if image.ndim == 2:
        image = gray2rgb(image)
    if image.shape[2] == 4:
        image = rgba2rgb(image)
    return image


def preprocess_dataset(image_dirs):
    images = []
    for image_dir in image_dirs:
        try:
            image = skimage.io.imread(image_dir)
            if image.shape[0] % 2 == 0 and image.shape[1] % 2 == 0:
                images.append(image)
        except:
            continue

    images = np.array(images)

    # transform rgba/grey-scale image to rgb image
    for i, image in enumerate(images):
        images[i] = image2rgb(image)

    # create downscaled image
    downscaled_images = []
    for image in images:
        downscaled_images.append(downscale_local_mean(image, (2, 2, 1)))

    downscaled_images = np.array(downscaled_images)

    return images, downscaled_images


def convert_types(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    label = tf.cast(label, tf.float32)
    label /= 255
    return image, label


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(from_logits=True)(tf.ones_like(fake_output), fake_output)


def srresnet_loss(real_output, fake_output):
    return cross_entropy(from_logits=True)(real_output, fake_output)


def train_model(input_dir_queue):
    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Pre-train generator with supervised learning
    for step in range(1000):
        # random select 32 images
        selected_image_dirs = np.random.choice(input_dir_queue, images_in_memory, replace=False)
        images, downscaled_images = preprocess_dataset(selected_image_dirs)
        dataset_size = len(images)
        if dataset_size == 0:
            continue

        print("Train generator")
        for i in range(images_in_memory):
            offset = i % dataset_size
            downscaled_image, image = convert_types(expand_dims(downscaled_images[offset]), expand_dims(images[offset]))

            with tf.GradientTape() as gen_tape:
                generated_image = generator(downscaled_image, training=True)

                # calculate loss
                loss = srresnet_loss(image, generated_image)

            gradients_of_generator = gen_tape.gradient(loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

            # print image
            if i == images_in_memory - 1:
                print(loss)
                print("Period: ", i)
                show_img(generated_image)

    # train GAN
    for step in range(10000):
        # random select 32 images
        selected_image_dirs = np.random.choice(input_dir_queue, images_in_memory, replace=False)
        images, downscaled_images = preprocess_dataset(selected_image_dirs)
        dataset_size = len(images)
        if dataset_size == 0:
            continue

        print("Training GAN")
        for i in range(images_in_memory):
            offset = i % dataset_size
            downscaled_image, image = convert_types(expand_dims(downscaled_images[offset]), expand_dims(images[offset]))

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_image = generator(downscaled_image, training=True)

                real_output = discriminator(image, training=True)
                fake_output = discriminator(downscaled_image, training=True)

                # calculate loss
                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            # Back propagation
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            # print image
            if i == images_in_memory - 1:
                print(disc_loss)
                print("Period: ", i)
                show_img(generated_image)


def show_img(generated_image):
    img = generated_image[0] * 255
    img = np.array(img).astype('int')
    plt.imshow(img)
    plt.show()
    plt.clf()


def get_file_list():
    input_dir_queue = []
    input_dir_queue.extend(glob(DATASET_PATH + '/*/*.jpg'))
    input_dir_queue.extend(glob(DATASET_PATH + '/*/*.png'))

    return np.array(input_dir_queue)


def main():
    input_dir_queue = get_file_list()
    train_model(input_dir_queue)


if __name__ == '__main__':
    main()
