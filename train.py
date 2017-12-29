from src.model     import Model
from src.generator import Generator


m = Model().skip_layer_vgg16()
g = Generator()

hist = m.fit_generator(
    generator        = g.train(),
    steps_per_epoch  = 100,
    epochs           = 30,
    validation_data  = g.valid(),
    validation_steps = 10,
    verbose          = 2
)

