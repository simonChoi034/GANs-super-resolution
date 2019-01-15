import tensorflow as tf
import numpy as np
import skimage
from skimage import io
from skimage.transform import downscale_local_mean
from skimage.color import rgba2rgb, gray2rgb
import matplotlib.pyplot as plt
import os
from model import generator, discriminator

DATASET_PATH = 'dataset/image/anime/'
batch_size = 1
images_in_memory = 32
learning_rate = 0.0001


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

    # normalizing image
    images = (images / 127.5) - 1
    downscaled_images = (downscaled_images / 127.5) - 1

    return images, downscaled_images


def train_model(input_dir_queue):
    graph = tf.Graph()
    with graph.as_default():
        # Graph
        downscaled_img = tf.placeholder(tf.float32, shape=[batch_size, None, None, 3], name='downscaled_img')
        original_img = tf.placeholder(tf.float32, shape=[batch_size, None, None, 3], name='original_img')

        Gz = generator(downscaled_img)

        Dx = discriminator(original_img)

        Dg = discriminator(Gz, reuse_variable=True)

        # Two Loss Functions for discriminator
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))

        # Loss function for generator
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

        # Get the variables for different network
        d_vars = tf.trainable_variables(scope='discriminator')
        g_vars = tf.trainable_variables(scope='generator')

        # Train the discriminator
        d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
        d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

        # Train the generator
        g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

        # evaluation score
        pixel_error = tf.losses.mean_squared_error(labels=original_img, predictions=Gz)

        # From this point forward, reuse variables
        tf.get_variable_scope().reuse_variables()

    # config Gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #
    pixel_errors = []

    with tf.Session(graph=graph, config=config) as sess:
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()

        ## pre-train discrimiator
        # random select 32 images
        selected_image_dirs = np.random.choice(input_dir_queue, images_in_memory, replace=False)
        images, downscaled_images = preprocess_dataset(selected_image_dirs)
        dataset_size = len(images)

        print("Training discriminator")
        for i in range(1000):
            offset = (i * batch_size) % (dataset_size - batch_size)
            feet_dict = {'original_img:0': expand_dims(images[offset:offset + batch_size]),
                         'downscaled_img:0': expand_dims(downscaled_images[offset:offset + batch_size])}
            _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                                   feed_dict=feet_dict)

        # saving check point
        saver.save(sess, "./tf_model")

        for step in range(1000):
            # load 32 images into memory
            print("Load batch of images into memory")
            queue_offset = (step * images_in_memory) % (len(input_dir_queue) - images_in_memory)
            selected_image_dirs = input_dir_queue[queue_offset:queue_offset + images_in_memory]
            images, downscaled_images = preprocess_dataset(selected_image_dirs)
            dataset_size = len(images)

            # Train generator and discriminator together
            print("Start training")
            for i in range(300):
                offset = (i * batch_size) % (dataset_size - batch_size)
                feet_dict = {'original_img:0': expand_dims(images[offset:offset + batch_size]),
                             'downscaled_img:0': expand_dims(downscaled_images[offset:offset + batch_size])}

                # Train discriminator on both real and fake images
                _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                                       feed_dict=feet_dict)

                # Train generator
                _, generated_img, error = sess.run([g_trainer, Gz, pixel_error], feed_dict={
                    'downscaled_img:0': np.expand_dims(downscaled_images[i % dataset_size], axis=0)})

                error = error[0]

                if i % 60 == 0:
                    print("Period: ", i)
                    print("Pixel error: ", error)
                    pixel_errors.append(error)

                    # saving check point
                    saver.save(sess, "./tf_model")

        plt.ylabel("Error")
        plt.xlabel("Periods")
        plt.title("Error vs. Periods")
        plt.plot(pixel_errors, label="training")
        plt.legend()
        plt.show()

        # save model
        saver.save(sess, './tf_model', global_step=1)
        sess.close()


def get_file_list():
    input_dir_queue = []
    for dir in os.listdir(DATASET_PATH):
        try:
            for file in os.listdir(DATASET_PATH + dir):
                file_dir = DATASET_PATH + dir + '/' + file
                input_dir_queue.append(file_dir)
        except:
            continue

    return np.array(input_dir_queue)


def main():
    input_dir_queue = get_file_list()
    train_model(input_dir_queue)


if __name__ == '__main__':
    main()
