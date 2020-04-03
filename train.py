from conditional_gan import make_generator, make_discriminator, CGAN
import cmd
from gan.train import Trainer

from pose_dataset import PoseHMDataset
from utils import Logger
import sys
import os.path as osp
import os
import datetime
import json

def main():
    args = cmd.args()

    date_str = '{}'.format(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
    sys.stdout = Logger(osp.join(args.output_dir, 'log_{}.txt'.format(date_str)))
    # save opts
    with open(osp.join(args.output_dir, 'args_{}.json'.format(date_str)), 'w') as fp:
        json.dump(vars(args), fp, indent=1)


    generator = make_generator(args.image_size, args.use_input_pose, args.warp_skip, args.disc_type, args.warp_agg,args.debug)
    if args.generator_checkpoint is not None:
        generator.load_weights(args.generator_checkpoint)
    
    discriminator = make_discriminator(args.image_size, args.use_input_pose, args.warp_skip, args.disc_type, args.warp_agg)
    if args.discriminator_checkpoint is not None:
        discriminator.load_weights(args.discriminator_checkpoint)
    
    dataset = PoseHMDataset(test_phase=False, **vars(args))
    
    gan = CGAN(generator, discriminator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))
    
    trainer.train()
    
if __name__ == "__main__":
    main()
