from scipy.misc import imread, imresize


class Loader(object):

    def load_image(self, path):
        try:
            image = imread(path)
            image = imresize(image, (224,224))
        except:
            print(path)
        return image

