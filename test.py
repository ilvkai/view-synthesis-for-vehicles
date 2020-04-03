import os

from conditional_gan import make_generator
import cmd
from pose_dataset import PoseHMDataset

from gan.inception_score import get_inception_score

from skimage.io import imread, imsave
from skimage.measure import compare_ssim

import numpy as np
import pandas as pd

from tqdm import tqdm
import re
from gan.fid_score import FID_score

def l1_score(generated_images, reference_images):
    score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        score = np.abs(2 * (reference_image/255.0 - 0.5) - 2 * (generated_image/255.0 - 0.5)).mean()
        score_list.append(score)
    return np.mean(score_list)


def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        ssim = compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    return np.mean(ssim_score_list)


def save_images(input_images, target_images, generated_images, names, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for images in zip(input_images, target_images, generated_images, names):
        res_name = str('_'.join(images[-1])) + '.png'
        imsave(os.path.join(output_folder, res_name), np.concatenate(images[:-1], axis=1))





def load_generated_images(images_folder):
    input_images = []
    target_images = []
    generated_images = []
    names = []
    count=0
    countTotal = len(os.listdir(images_folder))
    for img_name in os.listdir(images_folder):
        img = imread(os.path.join(images_folder, img_name))
        h = img.shape[1] / 3
        input_images.append(img[:, :h])
        target_images.append(img[:, h:2*h])
        generated_images.append(img[:, 2*h:])

        m = re.match(r'([A-Za-z0-9_]*.jpg)_([A-Za-z0-9_]*.jpg)', img_name)
        fr = m.groups()[0]
        to = m.groups()[1]
        names.append([fr, to])
        count = count +1
        print('count is {} / {}'.format(count, countTotal))

    return input_images, target_images, generated_images, names


def generate_images(dataset, generator,  use_input_pose):
    input_images = []
    target_images = []
    generated_images = []
    names = []

    def deprocess_image(img):
        return (255 * ((img + 1) / 2.0)).astype(np.uint8)

    for _ in tqdm(range(dataset._pairs_file_test.shape[0])):
        batch, name = dataset.next_generator_sample_test(with_names=True)
        out = generator.predict(batch)
        input_images.append(deprocess_image(batch[0]))
        out_index = 2 if use_input_pose else 1
        target_images.append(deprocess_image(batch[out_index]))
        generated_images.append(deprocess_image(out[out_index]))
        names.append([name.iloc[0]['from'], name.iloc[0]['to']])


    input_array = np.concatenate(input_images, axis=0)
    target_array = np.concatenate(target_images, axis=0)
    generated_array = np.concatenate(generated_images, axis=0)
    return input_array, target_array, generated_array, names


def test():
    args = cmd.args()
    if args.load_generated_images:
        print ("Loading images...")
        input_images, target_images, generated_images, names = load_generated_images(args.generated_images_dir)
    else:
        print ("Generate images...")
        from keras import backend as K
        if args.use_dropout_test:
            K.set_learning_phase(1)
        dataset = PoseHMDataset(test_phase=True, **vars(args))
        generator = make_generator(args.image_size, args.use_input_pose, args.warp_skip, args.disc_type, args.warp_agg)
        assert (args.generator_checkpoint is not None)
        generator.load_weights(args.generator_checkpoint)
        input_images, target_images, generated_images, names = generate_images(dataset, generator, args.use_input_pose)
        print ("Save images to %s..." % (args.generated_images_dir, ))
        save_images(input_images, target_images, generated_images, names,
                        args.generated_images_dir)

    if args.is_test_real_data:
        generated_images=target_images

    print('Compute FID score')
    fid = FID_score()
    fid_score = fid.calculate_fid_images(generated_images, target_images)
    print ("FID score %s" % fid_score)


    print ("Compute inception score...")
    inception_score = get_inception_score(generated_images)
    print ("Inception score %s" % inception_score[0])

    print ("Compute structured similarity score (SSIM)...")
    structured_score = ssim_score(generated_images, target_images)
    print ("SSIM score %s" % structured_score)

    print ("Compute l1 score...")
    norm_score = l1_score(generated_images, target_images)
    print ("L1 score %s" % norm_score)


if __name__ == "__main__":
    test()


