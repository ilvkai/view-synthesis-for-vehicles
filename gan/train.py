import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import keras.backend as K
assert K.image_data_format() == 'channels_last', "Backend should be tensorflow and data_format channel_last"
from keras.backend import tf as ktf
config = ktf.ConfigProto()
config.gpu_options.allow_growth = True
session = ktf.Session(config=config)
K.set_session(session)
from tqdm import tqdm

class Trainer(object):
    def __init__(self, dataset, gan, output_dir = 'output/generated_samples',
                 checkpoints_dir='output/checkpoints', training_ratio=5,
                 display_ratio=1, checkpoint_ratio=10, start_epoch=0,
                 number_of_epochs=100, batch_size=64, **kwargs):
        self.dataset = dataset
        self.current_epoch = start_epoch
        self.last_epoch = start_epoch + number_of_epochs
        self.generator = gan.get_generator()
        self.discriminator = gan.get_discriminator()
        self.gan = gan
        
        generator_model, discriminator_model = gan.compile_models()        
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        
        self.batch_size = batch_size        
        self.output_dir = output_dir
        self.checkpoints_dir = checkpoints_dir
        self.training_ratio = training_ratio
        self.display_ratio = display_ratio
        self.checkpoint_ratio = checkpoint_ratio
        self.debug=kwargs['debug']
        
        
    def save_generated_images(self):
        if hasattr(self.dataset, 'next_generator_sample_test'):
            batch = self.dataset.next_generator_sample_test()
        else:
            batch = self.dataset.next_generator_sample() 
        image = self.dataset.display(self.generator.predict_on_batch(batch), batch)
        title = "epoch_{}.png".format(str(self.current_epoch).zfill(3))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        plt.imsave(os.path.join(self.output_dir, title), image,  cmap=plt.cm.gray)
        
    def make_checkpoint(self):
        g_title = "epoch_{}_generator.h5".format(str(self.current_epoch).zfill(3))
        d_title = "epoch_{}_discriminator.h5".format(str(self.current_epoch).zfill(3))
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.discriminator.save(os.path.join(self.checkpoints_dir, d_title))
        self.generator.save(os.path.join(self.checkpoints_dir, g_title))        
    
    def train_one_step(self, discriminator_loss_list, generator_loss_list):
        for j in range(self.training_ratio):
            discrimiantor_batch = self.dataset.next_discriminator_sample()
            generator_batch = self.dataset.next_generator_sample()
            #All zeros as ground truth because it`s not used
            loss = self.discriminator_model.train_on_batch(
                            discrimiantor_batch + generator_batch, np.zeros([self.batch_size]))
            discriminator_loss_list.append(loss)

        generator_batch = self.dataset.next_generator_sample()
        loss = self.generator_model.train_on_batch(generator_batch, np.zeros([self.batch_size]))
        generator_loss_list.append(loss)
    
    def train_one_epoch(self, validation_epoch=False):
        print("Epoch: %i" % self.current_epoch)
        discriminator_loss_list = []
        generator_loss_list = []

        
        for _ in tqdm(range(int(self.dataset.number_of_batches_per_epoch()))):
            try:
                self.train_one_step(discriminator_loss_list, generator_loss_list)
            except ktf.errors.InvalidArgumentError as err:
                print (err)

       
        g_loss_str, d_loss_str = self.gan.get_losses_as_string(np.mean(np.array(generator_loss_list), axis = 0),
                                                               np.mean(np.array(discriminator_loss_list), axis = 0))        
        print (g_loss_str)
        print (d_loss_str)
        
        if hasattr(self.dataset, 'next_generator_sample_test') and validation_epoch:
            print ("Validation...")
            validation_loss_list = []
            for _ in tqdm(range(int(self.dataset.number_of_batches_per_validation()))):
                    generator_batch = self.dataset.next_generator_sample_test()
                    loss = self.generator_model.test_on_batch(generator_batch, np.zeros([self.batch_size]))
                    validation_loss_list.append(loss)
            val_loss_str, d_loss_str = self.gan.get_losses_as_string(np.mean(np.array(validation_loss_list), axis=0),
                                                                      np.mean(np.array(discriminator_loss_list), axis=0))
            print (val_loss_str.replace('Generator loss', 'Validation loss'))
        
    def train(self):
        while (self.current_epoch < self.last_epoch):            
            if (self.current_epoch + 1) % self.display_ratio == 0:
                self.save_generated_images()
            if (self.current_epoch + 1) % self.checkpoint_ratio == 0:
                self.make_checkpoint()     
            self.train_one_epoch((((self.current_epoch + 1) % self.checkpoint_ratio == 0) or self.current_epoch==0))
            self.current_epoch += 1
