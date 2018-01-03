from keras.applications import VGG16
from keras.layers       import Conv2D, Conv2DTranspose, Dropout, Add
from keras.models       import Model as KerasModel
from keras.optimizers   import Adam
from metric             import Metric


DROPOUT_VALUE = 0.2


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
        d5 = Dropout(DROPOUT_VALUE)(v5.output)
        c5 = Conv2D(
            filters     = 1, 
            kernel_size = (1,1), 
            activation  = 'relu',
            padding     = 'same'
        )(d5)
        d5 = Dropout(DROPOUT_VALUE)(c5)
        t5 = Conv2DTranspose(
            filters     = 1, 
            kernel_size = (2,2), 
            strides     = (2,2), 
            activation  = 'relu',
            padding     = 'same'
        )(d5)

        v4 = vgg16.get_layer('block4_pool')
        d4 = Dropout(DROPOUT_VALUE)(v4.output)
        c4 = Conv2D(
            filters     = 1 ,
            kernel_size = (1,1), 
            activation  = 'relu',
            padding     = 'same'
        )(d4)
        a4 = Add()([t5, c4])
        d4 = Dropout(DROPOUT_VALUE)(a4)
        t4 = Conv2DTranspose(
            filters     = 1, 
            kernel_size = (2,2), 
            strides     = (2,2), 
            activation  = 'relu',
            padding     = 'same'
        )(d4)

        v3 = vgg16.get_layer('block3_pool')
        d3 = Dropout(DROPOUT_VALUE)(v3.output)
        c3 = Conv2D(
            filters     = 1, 
            kernel_size = (1,1), 
            activation  = 'relu',
            padding     = 'same'
        )(d3)
        a3 = Add()([t4, c3])
        d3 = Dropout(DROPOUT_VALUE)(a3)
        t3 = Conv2DTranspose(
            filters     = 1, 
            kernel_size = (2,2), 
            strides     = (2,2), 
            activation  = 'relu',
            padding     = 'same'
        )(d3)

        v2 = vgg16.get_layer('block2_pool')
        d2 = Dropout(DROPOUT_VALUE)(v2.output)
        c2 = Conv2D(
            filters     = 1, 
            kernel_size = (1,1), 
            activation  = 'relu',
            padding     = 'same'
        )(d2)
        a2 = Add()([t3, c2])
        d2 = Dropout(DROPOUT_VALUE)(a2)
        t2 = Conv2DTranspose(
            filters     = 1, 
            kernel_size = (2,2), 
            strides     = (2,2), 
            activation  = 'relu',
            padding     = 'same'
        )(d2)

        v1 = vgg16.get_layer('block1_pool')
        d1 = Dropout(DROPOUT_VALUE)(v1.output)
        c1 = Conv2D(
            filters     = 1, 
            kernel_size = (1,1), 
            activation  = 'relu',
            padding     = 'same'
        )(d1)
        a1 = Add()([t2, c1])
        d1 = Dropout(DROPOUT_VALUE)(a1)
        t1 = Conv2DTranspose(
            filters     = 1, 
            kernel_size = (2,2), 
            strides     = (2,2), 
            activation  = 'relu',
            padding     = 'same'
        )(d1)

        d0 = Dropout(DROPOUT_VALUE)(t1)
        c0 = Conv2D(
            filters     = 1, 
            kernel_size = (1,1), 
            padding     = 'same',
            activation  = 'tanh'
        )(d0)

        model = KerasModel(
            inputs  = [vgg16.input], 
            outputs = [c0]
        )
        
        #metric = Metric().dice_init(1e-3)

        model.compile(
            optimizer = Adam(lr=1e-5), 
            loss      = 'mse',    # metric.dice_loss
            metrics   = ['mse']   # metric.dice_coef
        )

        return model

