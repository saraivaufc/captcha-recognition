import numpy as np
import tensorflow as tf

import image_utils


class Classifier(object):
    def __init__(self, model, model_dir):
        self.__model = model
        self.__model.summary()

        checkpoint_path = "{dir}/model.ckpt".format(dir=model_dir)

        latest = tf.train.latest_checkpoint(model_dir)

        if latest:
            model.load_weights(latest)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=model_dir)

        self.__callbacks = [cp_callback, tensorboard_callback]

    def train(self, input_train, input_test, epochs, batch_size):
        train_file, train_data, train_labels = image_utils.load_dataset(
            input_train, read_only=True)
        test_file, test_data, test_labels = image_utils.load_dataset(
            input_test, read_only=True)

        train_size = train_data.len()
        test_size = test_data.len()

        print("\nTrain: {0}\nTest:{1}\n".format(train_size, test_size))

        train_images = np.asarray(train_data,
                                  dtype=np.float32)

        train_labels = np.asarray(train_labels,
                                  dtype=np.int8)

        test_images = np.asarray(test_data,
                                 dtype=np.float32)

        test_labels = np.asarray(test_labels,
                                 dtype=np.int8)

        self.__model.fit(x=train_images, y=train_labels,
                         epochs=epochs,
                         batch_size=batch_size,
                         callbacks=self.__callbacks,
                         shuffle=True,
                         validation_data=(test_images, test_labels))

    def predict(self, images_path):
        images_batch = []
        for image_path in images_path:
            image = image_utils.load_file(image_path, resize_to=(50,150))
            normalized_image = image_utils.normalize(image)
            images_batch.append(normalized_image)

        pred = self.__model.predict(np.array(images_batch, dtype=np.float32))

        text_list = []
        for p in pred:
            p[p > 0.5] = 1
            p[p <= 0.5] = 0
            text = image_utils.labels_to_text(p)
            text_list.append(text)

        return text_list
