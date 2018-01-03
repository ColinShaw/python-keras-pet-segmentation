import cv2
import numpy as np
from   keras.applications import imagenet_utils
from   loader             import Loader
from   model              import Model


class Predict(Loader):

    def __init__(self):
        Loader.__init__(self)

    def segmentation(self):
        self.__model = Model().skip_layer_vgg16()
        self.__model.load_weights('model_weights.h5')
        self.__image = self.load_image('images/test.jpg')
        image = self.__image.astype(np.float32)
        scaled = imagenet_utils.preprocess_input(image)
        masked = self.__model.predict([image])

        # Overlay the result onto the original image, save...

