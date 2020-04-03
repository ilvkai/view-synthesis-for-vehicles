from keras import backend as K
from keras.optimizers import Adam
from gan import GAN

class LSGAN(GAN):
    """
        Class for representing LSGAN (https://arxiv.org/pdf/1611.04076.pdf)
    """
    def __init__(self, generator, discriminator,
                 generator_optimizer=Adam(0.0001, beta_1=.5, beta_2=0.9),
                 discriminator_optimizer=Adam(0.0001, beta_1=.5, beta_2=0.9),
                    **kwargs):
        super(LSGAN, self).__init__(generator, discriminator, generator_optimizer=generator_optimizer,
                                      discriminator_optimizer = discriminator_optimizer, **kwargs)

    def _compile_generator_loss(self):
        def generator_least_square_loss(y_true, y_pred):
            return K.mean((y_pred - 1) ** 2)
        return generator_least_square_loss, []

    def _compile_discriminator_loss(self):
        def true_loss(y_true, y_pred):
            y_true = y_pred[:self._batch_size]
            return K.mean((y_true - 1) ** 2)

        def fake_loss(y_true, y_pred):
            y_fake = y_pred[self._batch_size:]
            return K.mean(y_fake ** 2)

        def discriminator_least_square_loss(y_true, y_pred):
            return fake_loss(y_true, y_pred) + true_loss(y_true, y_pred)

        return discriminator_least_square_loss, [true_loss, fake_loss]
