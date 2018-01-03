from keras.callbacks import ModelCheckpoint
from os.path         import isfile
from model           import Model
from generator       import Generator


MODEL_WEIGHTS_FILE = 'model_weights.h5'


class Train(object):

    @staticmethod
    def oxford():
        model      = Model().skip_layer_vgg16()
        if isfile(MODEL_WEIGHTS_FILE):
            model.load_weights(MODEL_WEIGHTS_FILE)
        generator  = Generator()
        checkpoint = ModelCheckpoint(
            filepath       = MODEL_WEIGHTS_FILE, 
            verbose        = True, 
            save_best_only = True
        )
        model.fit_generator(
            generator        = generator.train(128),
            steps_per_epoch  = 200,
            epochs           = 50,
            validation_data  = generator.valid(128),
            validation_steps = 5,
            callbacks        = [checkpoint]
        )


