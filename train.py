import numpy as np
from   keras.applications        import VGG16
from   keras.layers              import Conv2D, Conv2DTranspose, Add
from   keras.models              import Model
from   keras.optimizers          import Adam
from   keras.preprocessing       import image
from   keras.preprocessing.image import ImageDataGenerator


def generate_model():
    vgg16 = VGG16(weights='imagenet', include_top=False)
    for layer in vgg16.layers:
        layer.trainable = False

    l5 = vgg16.get_layer('block5_pool').output
    c5 = Conv2D(1,(1,1))(l5)
    t5 = Conv2DTranspose(1,4,strides=(2,2))(c5)
    l4 = vgg16.get_layer('block4_pool').output
    c4 = Conv2D(1,(1,1))(l4)
    a4 = Add()([t5, c4])
    t4 = Conv2DTranspose(1,4,strides=(2,2))(a4)
    l1 = vgg16.get_layer('block1_pool').output
    c1 = Conv2D(1,(1,1))(l1)
    a1 = Add()([t4, c1])
    t1 = Conv2DTranspose(1,16,strides=(8,8))(a1)

    model = Model(inputs=vgg16.input, outputs=t1)

    model.compile(
        optimizer = Adam(lr=1e-5), 
        loss      = 'categorical_crossentropy', 
        metrics   = ['accuracy']
    )
    model.summary()

    return model


def load_image(path):
    return image.load_img(path, target_size=(224,224))


def load_data():
    pass


gen = ImageDataGenerator(
    rescale            = 1.0 / 255,
    rotation_range     = 90,
    width_shift_range  = 0.2,
    height_shift_range = 0.2,
    shear_range        = 0.2,
    zoom_range         = 0.2,
    horizontal_flip    = True,
    fill_mode          = 'nearest'
)


model = generate_model()

