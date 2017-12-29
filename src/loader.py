from keras.preprocessing import image


class Loader(object):

    def load_image(self, path):
        return image.load_img(path, target_size=(224,224))

    def train(self):
        pass

    def valid(self):
        pass

