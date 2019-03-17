import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as cross_entropy
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from model import Generator, Discriminator
from helper import preprocess_dataset, get_file_list, expand_dims, show_img

DATASET_PATH = '/Users/Simon/tf_files/dataset/image/anime'
images_in_memory = 32
learning_rate = 1e-5


def convert_types(image, label):
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(from_logits=True)(tf.ones_like(fake_output), fake_output)


def srresnet_loss(real_output, fake_output):
    return MeanSquaredError()(real_output, fake_output)


def train_model(input_dir_queue):
    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

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
                fake_output = discriminator(generated_image, training=True)

                # calculate loss
                gen_loss = generator_loss(fake_output) + srresnet_loss(image, generated_image)
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


def main():
    input_dir_queue = get_file_list(DATASET_PATH)
    train_model(input_dir_queue)


if __name__ == '__main__':
    main()
