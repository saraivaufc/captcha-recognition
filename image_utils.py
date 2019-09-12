# -*- encoding=UTF-8-*-

import os

import h5py
import numpy as np
from PIL import Image
from skimage.transform import rotate
from tqdm import tqdm
from matplotlib import pyplot as plt

import settings
import time

def load_file(captcha_image_file, resize_to=None):
    if isinstance(captcha_image_file, str):
        image = Image.open(captcha_image_file)
    else:
        image = captcha_image_file

    image = image.convert('LA')

    image_array = np.array(image.getdata())\
        .reshape(image.size[1], image.size[0], 2) \
        [:, :, :1]

    plt.imshow(image_array.reshape((image_array.shape[0], image_array.shape[
        1])))
    plt.show()

    return image_array

def normalize(image):
    mean = np.mean(image)
    std = np.std(image)
    normalized = ((image - mean) / float(std))
    return normalized

def load_dataset(dataset, read_only=False):
    if read_only:
        dataset = h5py.File(dataset, 'r')
    else:
        dataset = h5py.File(dataset, 'r+')
    x_data = dataset["x"]
    y_data = dataset["y"]
    return dataset, x_data, y_data


def make_dataset(filename, x_shape, y_shape):
    dataset = h5py.File(filename, 'w')
    x_data = dataset.create_dataset("x", (0,
                                          x_shape[0],
                                          x_shape[1],
                                          x_shape[2]),
                                    'f',
                                    maxshape=(None,
                                              x_shape[0],
                                              x_shape[1],
                                              x_shape[2]),
                                    chunks=True)
    y_data = dataset.create_dataset("y", (0, y_shape[0]),
                                    'f',
                                    maxshape=(None, y_shape[0]),
                                    chunks=True)

    return dataset, x_data, y_data


def save_dataset(batch, output_path, x_shape, y_shape):
    if os.path.isfile(output_path):
        dataset, x_data, y_data = load_dataset(output_path)
    else:
        dataset, x_data, y_data = make_dataset(output_path,
                                               x_shape,
                                               y_shape)

    length = len(batch)

    x_data_size = x_data.len()
    y_data_size = y_data.len()

    x_data.resize((x_data_size + length, x_shape[0], x_shape[1],
                   x_shape[2]))
    y_data.resize((y_data_size + length, y_shape[0]))

    for index in tqdm(iterable=range(length), miniters=10, unit=" samples"):
        x_data[x_data_size + index] = batch[index][0]
        y_data[y_data_size + index] = batch[index][1]

    dataset.close()


def text_to_labels(text):
    labels = []
    for i in range(settings.CAPTCHA_SIZE):
        if i < len(text):
            text_caracter = text[i]
        else:
            text_caracter = None
        for c in settings.CHARACTERS:
            if i < len(text):
                if text_caracter == c:
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                labels.append(0)
    return labels


def labels_to_text(labels):
    text = ""

    for index in range(0, settings.CAPTCHA_SIZE):
        labels_index = labels[index * len(settings.CHARACTERS):
                              index * len(settings.CHARACTERS) + len(
                                  settings.CHARACTERS)]

        try:
            i = list(labels_index).index(1)
            text += settings.CHARACTERS[i]
        except Exception as e:
            pass

    return text


def generate_dataset(images_dir,
                     output_path,
                     image_shape,
                     caracters,
                     captcha_size):
    files = os.listdir(images_dir)
    batch = []
    for filename in files:
        if filename.find(".png") != -1 or filename.find(".jpg") != -1:
            image_path = images_dir + "/" + filename

            train = load_file(image_path, image_shape)
            train = normalize(train)

            text = os.path.splitext(filename)[0]

            labels = text_to_labels(text)

            batch.append((train, labels))

    save_dataset(batch,
                 output_path,
                 image_shape, (captcha_size * len(caracters),))
    del batch
