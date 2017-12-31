from keras.applications import VGG16
from keras.layers       import Conv2D, Conv2DTranspose, Add
from keras.models       import Model as KerasModel
from keras.optimizers   import Adam
from iou                import IOU


class Model(object):

    @staticmethod
    def skip_layer_vgg16():
        vgg16 = VGG16(weights='imagenet', include_top=False)
        for layer in vgg16.layers:
            layer.trainable = False

        v5 = vgg16.get_layer('block5_pool')
        c5 = Conv2D(1,(1,1))(v5.output)
        t5 = Conv2DTranspose(1,4,strides=(2,2),padding='same')(c5)

        v4 = vgg16.get_layer('block4_pool')
        c4 = Conv2D(1,(1,1))(v4.output)
        a4 = Add()([t5, c4])
        t4 = Conv2DTranspose(1,4,strides=(8,8),padding='same')(a4)

        v1 = vgg16.get_layer('block1_pool')
        c1 = Conv2D(1,(1,1))(v1.output)
        a1 = Add()([t4, c1])
        t1 = Conv2DTranspose(1,16,strides=(2,2),padding='same')(a1)

        m = KerasModel(inputs=vgg16.input, outputs=t1)
        i = IOU(1.0)
        m.compile(
            optimizer = Adam(lr=1e-5), 
            loss      = i.loss, 
            metrics   = [i.iou]
        )
        return m

