from keras.callbacks import ModelCheckpoint
from model           import Model
from generator       import Generator


class Train(object):

    @staticmethod
    def oxford():
        model      = Model().skip_layer_vgg16()
        generator  = Generator()
        checkpoint = ModelCheckpoint(
            filepath       = 'model_weights.h5', 
            verbose        = True, 
            save_best_only = True
        )
        model.fit_generator(
            generator        = generator.train(32),
            steps_per_epoch  = 100,
            epochs           = 10,
            validation_data  = generator.valid(32),
            validation_steps = 2,
            callbacks        = [checkpoint]
        )


