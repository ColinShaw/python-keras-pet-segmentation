from loader import Loader
import random


class Oxford(Loader):

    def __init__(self):
        Loader.__init__(self)
        self.__load_data()

    def __load_data(self):
        features = self.load_images('data/oxford_iiit/images/*.jpg')
        labels   = self.load_images('data/oxford_iiit/trimaps/*.jpg')
        self.__data = []
        for i in range(len(features)):
            self.__data.append((features[i],labels[i]))
        random.shuffle(self.__data)

    def __scale_label(self, label):
        return label

    def train(self):
        data = features = labels = []
        for i in range(0,200):
            feature = self.__data[i][0]
            label   = self.__data[i][1]
            label   = self.__scale_label(label)
            data.append((feature,label))
        return data

    def valid(self):
        data = features = labels = []
        for i in range(200,250):
            feature = self.__data[i][0]
            label   = self.__data[i][1]
            label   = self.__scale_label(label)
            data.append((feature,label))
        return data

