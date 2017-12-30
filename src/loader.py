from glob       import glob
from scipy.misc import imread, imresize


class Loader(object):

    def __load_image(self, path):
        image = imread(path)
        image = imresize(image, (224,224))
        return image

    def load_images(self, path):
        images = []
        paths  = glob(path)
        for path in paths:
            image = self.__load_image(path)
            images.append(image)
        return images

