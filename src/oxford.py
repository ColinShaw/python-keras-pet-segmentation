import os
import numpy  as np
from   loader import Loader


class Oxford(Loader):

    def __init__(self):
        Loader.__init__(self)
        self.__data = []
        self.__load_data()

    def __load_data(self):
        dirlist = os.listdir('data/oxford_iiit/images')
        images = [f for f in dirlist if 'jpg' in f]
        for image in images:
            image   = os.path.splitext(image)[0]
            feature = self.load_image('data/oxford_iiit/images/{}.jpg'.format(image))
            label   = self.load_image('data/oxford_iiit/trimaps/{}.png'.format(image))
            self.__data.append((feature, label))

    def __scale_label(self, label):
        label[label > 2]   = 1.0
        label[label < 255] = 0.0
        return label.astype(np.float32)

    def train(self):
        data     = [] 
        features = [] 
        labels   = []
        for i in range(0, 200):
            feature = self.__data[i][0]
            label   = self.__data[i][1]
            label   = self.__scale_label(label)
            data.append((feature, label))
        return data

    def valid(self):
        data     = []
        features = [] 
        labels   = []
        for i in range(200, 250):
            feature = self.__data[i][0]
            label   = self.__data[i][1]
            label   = self.__scale_label(label)
            data.append((feature, label))
        return data

