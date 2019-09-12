from classifier import Classifier
from model import model_fn
import settings

input_train = "data/train.h5"
input_test = "data/test.h5"
batch_size = 128
epochs = 1000
model_dir = "data/logs"

model = model_fn(settings.IMAGE_SHAPE, settings.CHARACTERS, settings.CAPTCHA_SIZE)

classifier = Classifier(model=model, model_dir=model_dir)

classifier.train(
    input_train=input_train,
    input_test=input_test,
    epochs=epochs,
    batch_size=batch_size
)
