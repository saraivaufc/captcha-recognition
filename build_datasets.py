from image_utils import generate_dataset
import settings

# https://dominikschmidt.xyz/mnist_captcha/

# https://www.kaggle.com/genesis16/captcha-4-letter#captcha.zip

# https://www.kaggle.com/bongq417/captcha#train.zip

generate_dataset(images_dir="data/train",
                 output_path="data/train.h5",
                 image_shape=settings.IMAGE_SHAPE,
                 caracters=settings.CHARACTERS,
                 captcha_size=settings.CAPTCHA_SIZE)

generate_dataset(images_dir="data/test",
                 output_path="data/test.h5",
                 image_shape=settings.IMAGE_SHAPE,
                 caracters=settings.CHARACTERS,
                 captcha_size=settings.CAPTCHA_SIZE)