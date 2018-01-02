import random
import cv2 
import numpy as np
from   keras.applications.imagenet_utils import preprocess_input
from   scipy.misc                        import imrotate
from   random                            import shuffle
from   oxford                            import Oxford


class Generator(object):

    def __init__(self):
        self.__train = Oxford().train()
        self.__valid = Oxford().valid()

    def __augment_rotation(self, image):
        angle = np.random.uniform(-45, 45)
        image = imrotate(image, angle) 
        return image

    def __augment_luminance(self, image):
        scale        = np.random.uniform(0.25, 0.75)
        image        = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HLS)
        image[:,:,1] = image[:,:,1] * (0.25 + scale)
        image        = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
        return image

    def __augment_translation(self, image):
        rand_x = np.random.uniform(-80, 80)
        rand_y = np.random.uniform(-80, 80)
        scale  = np.float32([[1,0,rand_x], [0,rand_y,0]])
        image  = cv2.warpAffine(image, scale, (224,224))
        return image

    def __augment_feature(self, image):
        image = self.__augment_rotation(image)
        image = self.__augment_translation(image)
        image = self.__augment_luminance(image)
        image = np.reshape(image, (224,224,3))
        image = preprocess_input(image.astype(np.float32))
        return image

    def __preprocess_label(self, image):
        image = image.astype(np.float32)
        image = image / 255.0
        return image
 
    def __augment_label(self, image):
        image = self.__augment_rotation(image)
        image = self.__augment_translation(image)
        image = self.__preprocess_label(image)
        image = np.reshape(image, (224,224,1))
        return image

    def __make_zeros(self, batch_size):
        features = np.zeros((batch_size,224,224,3))
        labels   = np.zeros((batch_size,224,224,1))
        return features, labels

    def __augmented_train_feature(self, i):
        feature = self.__train[i][0]
        feature = self.__augment_feature(feature)
        return feature    

    def __augmented_train_label(self, i):
        label = self.__train[i][1]
        label = self.__augment_label(label)
        return label

    def __augmented_valid_feature(self, i):
        feature = self.__valid[i][0]
        feature = self.__augment_feature(feature)
        return feature

    def __augmented_valid_label(self, i):
        label = self.__valid[i][1]
        label = self.__augment_label(label)
        return label

    def train(self, batch_size):
        while True:
            random.shuffle(self.__train)
            features, labels = self.__make_zeros(batch_size)
            for i in range(batch_size):
                features[i] = self.__augmented_train_feature(i)
                labels[i]   = self.__augmented_train_label(i)
            yield features, labels

    def valid(self, batch_size):
        while True:
            random.shuffle(self.__valid)
            features, labels = self.__make_zeros(batch_size)
            for i in range(batch_size):
                features[i] = self.__augmented_valid_feature(i)
                labels[i]   = self.__augmented_valid_label(i)
            yield features, labels

