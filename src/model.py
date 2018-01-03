from keras.applications import VGG16
from keras.layers       import Conv2D, Conv2DTranspose, Add
from keras.models       import Model as KerasModel
from keras.optimizers   import Adam
from metric             import Metric


class Model(object):

    @staticmethod
    def skip_layer_vgg16():
        vgg16 = VGG16(
            weights     = 'imagenet', 
            include_top = False
        )

        for layer in vgg16.layers:
            layer.trainable = False

        v5 = vgg16.get_layer('block5_pool')
        c5 = Conv2D(
            filters     = 1, 
            kernel_size = (1,1), 
            activation  = 'relu',
            padding     = 'same'
        )(v5.output)
        t5 = Conv2DTranspose(
            filters     = 1, 
            kernel_size = (2,2), 
            strides     = (2,2), 
            activation  = 'relu',
            padding     = 'same'
        )(c5)

        v4 = vgg16.get_layer('block4_pool')
        c4 = Conv2D(
            filters     = 1 ,
            kernel_size = (1,1), 
            activation  = 'relu',
            padding     = 'same'
        )(v4.output)
        a4 = Add()([t5, c4])
        t4 = Conv2DTranspose(
            filters     = 1, 
            kernel_size = (2,2), 
            strides     = (2,2), 
            activation  = 'relu',
            padding     = 'same'
        )(a4)

        v3 = vgg16.get_layer('block3_pool')
        c3 = Conv2D(
            filters     = 1, 
            kernel_size = (1,1), 
            activation  = 'relu',
            padding     = 'same'
        )(v3.output)
        a3 = Add()([t4, c3])
        t3 = Conv2DTranspose(
            filters     = 1, 
            kernel_size = (2,2), 
            strides     = (2,2), 
            activation  = 'relu',
            padding     = 'same'
        )(a3)

        v2 = vgg16.get_layer('block2_pool')
        c2 = Conv2D(
            filters     = 1, 
            kernel_size = (1,1), 
            activation  = 'relu',
            padding     = 'same'
        )(v2.output)
        a2 = Add()([t3, c2])
        t2 = Conv2DTranspose(
            filters     = 1, 
            kernel_size = (2,2), 
            strides     = (2,2), 
            activation  = 'relu',
            padding     = 'same'
        )(a2)

        v1 = vgg16.get_layer('block1_pool')
        c1 = Conv2D(
            filters     = 1, 
            kernel_size = (1,1), 
            activation  = 'relu',
            padding     = 'same'
        )(v1.output)
        a1 = Add()([t2, c1])
        t1 = Conv2DTranspose(
            filters     = 1, 
            kernel_size = (2,2), 
            strides     = (2,2), 
            activation  = 'relu',
            padding     = 'same'
        )(a1)

        c0 = Conv2D(
            filters     = 1, 
            kernel_size = (1,1), 
            padding     = 'same',
            activation  = 'tanh'
        )(t1)

        model = KerasModel(
            inputs  = [vgg16.input], 
            outputs = [c0]
        )
        metric = Metric().dice_init(1e-3)

        model.compile(
            optimizer = Adam(lr=1e-5), 
            loss      = 'mse',    # metric.dice_loss
            metrics   = ['mse']   # metric.dice_coef
        )

        return model

