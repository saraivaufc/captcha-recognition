import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *


def model_fn(input_shape, characters, captcha_size):
    x_inputs = Input(input_shape)

    x = Conv2D(32, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(x_inputs)
    x = BatchNormalization(momentum=0.0)(x)
    x = Conv2D(32, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(x_inputs)
    x = BatchNormalization(momentum=0.0)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization(momentum=0.0)(x)
    x = Conv2D(64, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization(momentum=0.0)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization(momentum=0.0)(x)
    x = Conv2D(128, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization(momentum=0.0)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization(momentum=0.0)(x)
    x = Dropout(0.5)(x)

    x = Dense(captcha_size * (len(characters)), activation="sigmoid")(x)

    model = Model(inputs=x_inputs, outputs=x)

    # optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=0.045)

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    return model
