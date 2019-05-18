from glob import glob
import numpy as np
import skimage
from skimage import io
from skimage.transform import downscale_local_mean
from skimage.color import rgba2rgb, gray2rgb
import matplotlib.pyplot as plt


def get_file_list(dataset_path):
    input_dir_queue = []
    input_dir_queue.extend(glob(dataset_path + '/*.jpg'))
    input_dir_queue.extend(glob(dataset_path + '/*.png'))
    input_dir_queue.extend(glob(dataset_path + '/*/*.jpg'))
    input_dir_queue.extend(glob(dataset_path + '/*/*.png'))

    return np.array(input_dir_queue)


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
            if image.shape[0] == 1080 and image.shape[1] == 1920:
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

    # Normalize input
    images = (images - 127.5) / 127.5
    downscaled_images = (downscaled_images - 127.5) / 127.5

    return images, downscaled_images


def show_img(generated_image):
    img = (generated_image[0] + 1) * 127.5
    img = np.array(img).astype('int')
    plt.imshow(img)
    plt.show()
    plt.clf()
