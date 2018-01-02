import os


class Clean(object):
    
    @staticmethod
    def clean():
        os.unlink('model_weights.h5')

