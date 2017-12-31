from scipy.misc import imread, imresize


class Loader(object):

    def load_image(self, path):
        image = imread(path)
        image = imresize(image, (224,224))
        return image

