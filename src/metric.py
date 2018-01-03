from keras import backend as K


class Metric(object):

    def dice_init(self, constant):
        self.constant = constant
        return self

    def dice_coef(self, truth, prediction):
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
        return -self.dice_coef(truth, prediction)


    def __cross_flatten(self, truth, prediction):
        const           = 0.5 * K.ones((224*224,1))
        flat_truth      = K.flatten(truth)
        flat_prediction = K.flatten(prediction)
        bool_truth      = K.greater(flat_truth, const)
        bool_prediction = K.greater(flat_prediction, const)
        truth           = K.cast_to_floatx(bool_truth)
        prediction      = K.cast_to_floatx(bool_prediction)
        return truth, prediction 
    
    def cross_loss(self, truth, prediction):
        truth, prediction = self.__cross_flatten(truth, prediction)
        cross = K.categorical_crossentropy(truth, prediction)
        return cross


