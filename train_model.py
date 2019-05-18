import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as loss_object
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from model import Generator, Discriminator
from helper import preprocess_dataset, get_file_list, show_img
import argparse

images_in_memory = 32
batch_size = 8
learning_rate = 2e-4
LAMBDA = 100

# Global object
generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)


def convert_types(image, label):
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss


def srresnet_loss(real_output, fake_output):
    return MeanSquaredError()(real_output, fake_output)


def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([target, target], training=True)
        disc_generated_output = discriminator([target, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target) # + srresnet_loss(target, gen_output)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        # mean squared loss
        mse = srresnet_loss(target, gen_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))

    return mse


def train_model(input_dir_queue):
    # train GAN
    print("Training GAN")
    for step in range(10000):
        # random select 32 images
        selected_image_dirs = np.random.choice(input_dir_queue, images_in_memory, replace=False)
        images, downscaled_images = preprocess_dataset(selected_image_dirs)
        images, downscaled_images = convert_types(images, downscaled_images)

        dataset_size = len(images)
        if dataset_size == 0:
            continue

        print("Step ", step)

        loss = 0
        for i in range(10):
            offset = (i * batch_size) % dataset_size

            loss = train_step(images[offset:offset + batch_size], downscaled_images[offset:offset + batch_size])

        # print mse for reference
        print("Mean squared loss: ", loss)
        # show generated image
        generated_images = generator(downscaled_images, training=False)
        show_img(generated_images)


def main(DATASET_PATH):
    input_dir_queue = get_file_list(DATASET_PATH)
    train_model(input_dir_queue)


if __name__ == '__main__':
    # phase argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', metavar='folder_path', required=True,
                        dest='path', action='store', help='Path of dataset folder')
    args = parser.parse_args()

    main(DATASET_PATH=args.path)
