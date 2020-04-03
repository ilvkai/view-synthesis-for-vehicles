from keras import backend as K
from keras.optimizers import RMSprop
from gan import GAN
from keras.constraints import Constraint


class WeightClip(Constraint):
    def __init__(self, c=0.01):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__, 'c': self.c}


class WGAN(GAN):
    """
        Class for representing WGAN (https://arxiv.org/pdf/1701.07875.pdf).

    """
    def __init__(self, generator, discriminator,
                    generator_optimizer=RMSprop(5e-5),
                    discriminator_optimizer=RMSprop(5e-5),
                    **kwargs):
        """
        :param discriminator: All convolutions and dense layers should be created with kernel_constraint = WeightClip()
        """
        super(WGAN, self).__init__(generator, discriminator, generator_optimizer=generator_optimizer,
                                      discriminator_optimizer=discriminator_optimizer, **kwargs)

    def _compile_generator_loss(self):
        def generator_wasserstein_loss(y_true, y_pred):
            return -K.mean(y_pred)
        return generator_wasserstein_loss, []

    def _compile_discriminator_loss(self):
        def true_loss(y_true, y_pred):
            y_true = y_pred[:self._batch_size]
            return -K.mean(y_true)

        def fake_loss(y_true, y_pred):
            y_fake = y_pred[self._batch_size:]
            return K.mean(y_fake)

        def discriminator_wasserstein_loss(y_true, y_pred):
            return fake_loss(y_true, y_pred) + true_loss(y_true, y_pred)

        return discriminator_wasserstein_loss, [true_loss, fake_loss]

