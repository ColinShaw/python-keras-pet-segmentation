from keras import backend as K


class IOU(object):

    def __init__(self, constant):
        self.constant = constant

    def iou(self, truth, prediction):
        flat_truth      = K.flatten(truth)
        flat_prediction = K.flatten(prediction)
        intersection    = K.sum(flat_truth * flat_prediction)
        numerator       = 2 * intersection + self.constant
        sum_truth       = K.sum(flat_truth)
        sum_prediction  = K.sum(flat_prediction)
        denominator     = sum_truth + sum_prediction + self.constant
        iou             = numerator / denominator
        return iou

    def loss(self, truth, prediction):
        return -self.iou(truth, prediction)

