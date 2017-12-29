from oxford import Oxford


class Generator(object):

    def __init__(self):
        self.training_data   = Oxford().train()
        self.validation_data = Oxford().valid()

    def train(self):
        pass

    def valid(self):
        pass
