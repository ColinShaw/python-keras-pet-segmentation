import numpy as np
from   keras.applications        import VGG16
from   keras.layers              import Conv2D, Conv2DTranspose, Add
from   keras.models              import Model
from   keras.optimizers          import Adam
from   keras.preprocessing       import image
from   keras.preprocessing.image import ImageDataGenerator



def generate_model():
    vgg16 = VGG16(weights='imagenet', include_top=False)

    c5  = Conv2D(1,(1,1))(vgg16.layers[18].output)
    u5  = Conv2DTranspose(1,4,strides=(2,2))(c5)
    c4  = Conv2D(1,(1,1))(vgg16.layers[14].output)
    s4  = Add()([u5, c4])
    u4  = Conv2DTranspose(1,4,strides=(2,2))(s4)
    c1  = Conv2D(1,(1,1))(vgg16.layers[3].output)
    s1  = Add()([u4, c1])
    out = Conv2DTranspose(1,16,strides=(8,8))(s1)

    m = Model(inputs=vgg16.input, outputs=out)

    for i in range(19):
        m.layers[i].trainable = False

    m.compile(
        optimizer = Adam(lr=1e-5), 
        loss      = 'categorical_crossentropy', 
        metrics   = ['accuracy']
    )

    return m


def load_image(path):
    return image.load_img(path, target_size=(224,224))


def load_data():
    pass





model = generate_model()
model.summary()

