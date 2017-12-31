import cv2
from   keras.applications.imagenet_utils import preprocess_input
from   loader                            import Loader
from   model                             import Model


class Verify(Loader):

    def __init__(self):
        Loader.__init__(self)

    def verify(self):
        self.__m = Model().skip_layer_vgg16()
        self.__m.load_weights('model_weights.h5')
        self.__image = self.load_image('images/test.jpg')
        scaled = preprocess_input(image.astype(np.float32))
        masked = m.predict([image])

        # Overlay the result onto the original image, save...

