from keras.callbacks import ModelCheckpoint
from model           import Model
from generator       import Generator


class Train(object):

    @staticmethod
    def oxford():
        m = Model().skip_layer_vgg16()
        g = Generator()
        c = ModelCheckpoint(
            filepath       = 'model_weights.h5', 
            verbose        = True, 
            save_best_only = True
        )
        m.fit_generator(
            generator        = g.train(32),
            steps_per_epoch  = 100,
            epochs           = 30,
            validation_data  = g.valid(32),
            validation_steps = 10,
            callbacks        = [c]
        )


