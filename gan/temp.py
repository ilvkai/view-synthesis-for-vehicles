# example of calculating the frechet inception distance in Keras
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize

class FID_score(object):
    def __init__(self):
        self.model = InceptionV3(include_top=False, pooling='avg', input_shape=(256, 256, 3))

    def calculate_fid_one_image(self, reference_images, generated_images):
        reference_images = preprocess_input(reference_images)
        generated_images = preprocess_input(generated_images)
        # calculate activations
        act1 = self.model.predict(reference_images)
        act2 = self.model.predict(generated_images)
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def calculate_fid_images(self, reference_images, generated_images):
        # reference_images = self.scale_images(reference_images, (256,256,3))
        # generated_images = self.scale_images(generated_images, (256,256,3))
        # reference_images = preprocess_input(reference_images)
        # generated_images = preprocess_input(generated_images)
        fid_score_list = []
        for reference_image, generated_image in zip(reference_images, generated_images):
            reference_image = np.expand_dims(reference_image, axis=0)
            generated_image = np.expand_dims(generated_image, axis=0)
            ssim = self.calculate_fid_one_image(reference_image, generated_image)
            fid_score_list.append(ssim)
        return np.mean(fid_score_list)

    def scale_images(self, images, new_shape):
        images_list = list()
        for image in images:
            # resize with nearest neighbor interpolation
            new_image = resize(image, new_shape, 0)
            # store
            images_list.append(new_image)
        return asarray(images_list)



# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
# define two fake collections of images
images1 = randint(0, 255, 1*32*32*3)
images1 = images1.reshape((1,32,32,3))
images2 = randint(0, 255, 1*32*32*3)
images2 = images2.reshape((1,32,32,3))
print('Prepared', images1.shape, images2.shape)
# convert integer to floating point values
images1 = images1.astype(np.uint8)
images2 = images2.astype(np.uint8)
# resize images
images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))
print('Scaled', images1.shape, images2.shape)
# pre-process images
images1 = preprocess_input(images1)
images2 = preprocess_input(images2)
# fid between images1 and images1
fid = calculate_fid(model, images1, images1)
print('FID (same): %.3f' % fid)
# fid between images1 and images2
fid = calculate_fid(model, images1, images2)
print('FID (different): %.3f' % fid)
