from keras import backend as K


class Metric(object):

    '''
    Related to Dice coefficient
    '''
    def dice_init(self, constant):
        self.constant = constant
        return self

    def dice(self, truth, prediction):
        flat_truth      = K.flatten(truth)
        flat_prediction = K.flatten(prediction)
        intersection    = K.sum(flat_truth * flat_prediction)
        numerator       = 2 * intersection + self.constant
        sum_truth       = K.sum(flat_truth)
        sum_prediction  = K.sum(flat_prediction)
        denominator     = sum_truth + sum_prediction + self.constant
        dice            = numerator / denominator
        return dice

    def dice_loss(self, truth, prediction):
        return -self.dice(truth, prediction)


    '''
    Related to cross entropy
    '''
    def cross(self, truth, prediction):
        flat_truth      = K.flatten(truth)
        flat_prediction = K.flatter(prediction)
        pass 

    def cross_loss(self, truth, prediction):
        pass 

