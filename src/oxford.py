import os
import random
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
            label   = self.load_image('data/oxford_iiit/trimaps/{}.png'.format(image))
            feature = self.load_image('data/oxford_iiit/images/{}.jpg'.format(image))
            if len(feature.shape) < 3:
                feature = np.dstack((feature,feature,feature))
            self.__data.append((feature, label))
        random.shuffle(self.__data)

    def __label_threshold(self, label):
        label[label > 2]   = 255
        label[label < 255] = 0
        return label

    def train(self):
        data     = [] 
        features = [] 
        labels   = []
        for i in range(0, 5000):
            feature = self.__data[i][0]
            label   = self.__label_threshold(self.__data[i][1])
            data.append((feature, label))
        return data

    def valid(self):
        data     = []
        features = [] 
        labels   = []
        for i in range(5000, 6000):
            feature = self.__data[i][0]
            label   = self.__label_threshold(self.__data[i][1])
            data.append((feature, label))
        return data

