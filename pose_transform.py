from keras.models import Input, Model
from keras.engine.topology import Layer
from keras.backend import tf as ktf


import pose_utils
import pylab as plt
import numpy as np
from skimage.io import imread
from skimage.transform import warp_coords

import skimage.measure
import skimage.transform


from pose_utils import LABELS, MISSING_VALUE
from tensorflow.contrib.image import transform as tf_perspective_transform


import cv2
import numpy
import scipy.misc as scm


class PerspectiveTransformLayer(Layer):
    def __init__(self, number_of_transforms, aggregation_fn, init_image_size,debug, **kwargs):
        assert aggregation_fn in ['none', 'max', 'avg']
        self.aggregation_fn = aggregation_fn
        self.number_of_transforms = number_of_transforms
        self.init_image_size = init_image_size
        self.debug = debug
        super(PerspectiveTransformLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.image_size = list(input_shape[0][1:])
        self.perspective_mul = [1, 1, self.init_image_size[0] / self.image_size[0],
                           1, 1, self.init_image_size[1] / self.image_size[1],
                           1.0*self.image_size[0] / self.init_image_size[0],
                           1.0*self.image_size[1] / self.init_image_size[1]]
        self.perspective_mul = np.array(self.perspective_mul).reshape((1, 1, 8))

    def call(self, inputs):
        expanded_tensor = ktf.expand_dims(inputs[0], -1)
        print('expanded_tensor:', expanded_tensor.shape)
        multiples = [1, self.number_of_transforms, 1, 1, 1]
        tiled_tensor = ktf.tile(expanded_tensor, multiples=multiples)
        print('tiled_tensor:', tiled_tensor.shape)
        repeated_tensor = ktf.reshape(tiled_tensor, ktf.shape(inputs[0]) * np.array([self.number_of_transforms, 1, 1, 1]))
        print('repeated_tensor:', repeated_tensor.shape)


        perspective_transforms = inputs[1] / self.perspective_mul

        perspective_transforms = ktf.reshape(perspective_transforms, (-1, 8))
        tranformed = tf_perspective_transform(repeated_tensor, perspective_transforms)
        res = ktf.reshape(tranformed, [-1, self.number_of_transforms] + self.image_size)
        res = ktf.transpose(res, [0, 2, 3, 1, 4])

        #Use masks
        if len(inputs) == 3:
            mask = ktf.transpose(inputs[2], [0, 2, 3, 1])
            mask = ktf.image.resize_images(mask, self.image_size[:2], method=ktf.image.ResizeMethod.NEAREST_NEIGHBOR)
            res = res * ktf.expand_dims(mask, axis=-1)


        if self.aggregation_fn == 'none':
            res = ktf.reshape(res, [-1] + self.image_size[:2] + [self.image_size[2] * self.number_of_transforms])
        elif self.aggregation_fn == 'max':
            res = ktf.reduce_max(res, reduction_indices=[-2])
        elif self.aggregation_fn == 'avg':
            counts = ktf.reduce_sum(mask, reduction_indices=[-1])
            counts = ktf.expand_dims(counts, axis=-1)
            res = ktf.reduce_sum(res, reduction_indices=[-2])
            res /= counts
            res = ktf.where(ktf.is_nan(res), ktf.zeros_like(res), res)
        return res

    def compute_output_shape(self, input_shape):
        if self.aggregation_fn == 'none':
            return tuple([input_shape[0][0]] + self.image_size[:2] + [self.image_size[2] * self.number_of_transforms])
        else:
            return input_shape[0]

    def get_config(self):
        config = {"number_of_transforms": self.number_of_transforms,
                  "aggregation_fn": self.aggregation_fn}
        base_config = super(PerspectiveTransformLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def give_name_to_keypoints(array):
    res = {}
    for i, name in enumerate(LABELS):
        if array[i][0] != MISSING_VALUE and array[i][1] != MISSING_VALUE:
            res[name] = array[i][::-1]
    return res


def check_valid(kp_array):
    kp = give_name_to_keypoints(kp_array)
    result= check_keypoints_present(kp, ['F_license_plate', 'RF_roof', 'LF_roof', 'R_fog_lamp', 'L_fog_lamp', 'L_head_light', 'R_head_light'])  # lk

    #check different oritation , determined by two wheels: LF_wheel and RF_wheel
    opp_result=check_keypoints_present(kp,['LF_wheel','RF_wheel'])
    if opp_result:
        result = False
    return result


def check_keypoints_present(kp, kp_names):
    result = True
    for name in kp_names:
        result = result and (name in kp)
    return result


def compute_st_distance(kp):
    st_distance1 = np.sum((kp['R_fog_lamp'] - kp['LF_roof']) ** 2)
    st_distance2 = np.sum((kp['RF_roof'] - kp['L_fog_lamp']) ** 2)
    return np.sqrt((st_distance1 + st_distance2)/2.0)


def mask_from_kp_array(kp_array, border_inc, img_size):
    min = np.min(kp_array, axis=0)
    max = np.max(kp_array, axis=0)
    min -= int(border_inc)
    max += int(border_inc)

    min = np.maximum(min, 0)
    max = np.minimum(max, img_size[::-1])

    mask = np.zeros(img_size)
    mask[min[1]:max[1], min[0]:max[0]] = 1
    return mask


def get_array_of_points(kp, names):
    return np.array([kp[name] for name in names],dtype=np.int)

# 1 	LF_wheel	11 L_mirror
# 2 	LB_wheel 	12 R_mirror
# 3 	RF_wheel 	13 RF_roof
# 4 	RB_wheel 	14 LF_roof
# 5 	R_fog_lamp 	15 LB_roof
# 6 	L_fog_lamp 	16 RB_roof
# 7 	R_head_light 	17 	LB_lamp
# 8 	L_head_light 	18 	RB_lamp
# 9 	F_auto_logo 	19 	B_auto_logo
# 10 	F_license_plate 20 	B_license_plate
def pose_masks(array2, img_size):
    kp2 = give_name_to_keypoints(array2)
    masks = []
    # st2 = compute_st_distance(kp2)
    empty_mask = np.zeros(img_size) #should always be 0 and should not be modified

    # roof
    # body_mask = np.ones(img_size)# mask_from_kp_array(get_array_of_points(kp2, ['Rhip', 'Lhip', 'Lsho', 'Rsho']), 0.1 * st2, img_size)
    # masks.append(body_mask)

    # roof 13 14 15 16 in order**************************
    roof_candidate_names = {'RF_roof', 'LF_roof', 'LB_roof', 'RB_roof'}
    roof_kp_names = set()
    for cn in roof_candidate_names:
        if cn in kp2:
            roof_kp_names.add(cn)

    if len(roof_kp_names) == 4:
        temp_empty_mask = np.zeros(img_size) #should be set 0 everytime used lk
        points=get_array_of_points(kp2, ['RF_roof', 'LF_roof', 'LB_roof', 'RB_roof'])
        roofmask=cv2.fillConvexPoly(temp_empty_mask, points, 1)
        masks.append(roofmask)
    else:
        masks.append(empty_mask)


    #head_up   7 8 14 13       **************************
    head_up_names = {'R_head_light', 'L_head_light', 'LF_roof', 'RF_roof'}
    head_up_kp_names = set()
    for cn in head_up_names:
        if cn in kp2:
            head_up_kp_names.add(cn)
    if len(head_up_kp_names)==4:
        # tttt=cv2.imread("test.jpg")
        temp_empty_mask = np.zeros(img_size)  # should be set 0 everytime used lk
        points = get_array_of_points(kp2, ['R_head_light', 'L_head_light', 'LF_roof', 'RF_roof'])
        head_mask = cv2.fillConvexPoly(temp_empty_mask, points, 1)
        masks.append(head_mask)
        # scm.imsave('head_down_test.jpg', head_mask) #for test
    else:
        masks.append(empty_mask)


    # head_down   5 6 8 7        **************************
    head_candidate_names = {'R_fog_lamp', 'L_fog_lamp', 'L_head_light', 'R_head_light'}
    head_kp_names = set()
    for cn in head_candidate_names:
        if cn in kp2:
            head_kp_names.add(cn)
    if len(head_kp_names)==4:
        temp_empty_mask = np.zeros(img_size)  # should be set 0 everytime used lk
        points = get_array_of_points(kp2, ['R_fog_lamp', 'L_fog_lamp', 'L_head_light', 'R_head_light'])
        head_mask = cv2.fillConvexPoly(temp_empty_mask, points, 1)
        masks.append(head_mask)
    else:
        masks.append(empty_mask)



    #back_up 17 18 16 15   **************************
    candidate_names = {'LB_lamp', 'RB_lamp', 'RB_roof', 'LB_roof'}
    kp_names = set()
    for cn in candidate_names:
        if cn in kp2:
            kp_names.add(cn)
    if len(kp_names) == 4:
        temp_empty_mask = np.zeros(img_size)  # should be set 0 everytime used lk
        points = get_array_of_points(kp2, ['LB_lamp', 'RB_lamp', 'RB_roof', 'LB_roof'])
        part_mask = cv2.fillConvexPoly(temp_empty_mask, points, 1)
        masks.append(part_mask)
    else:
        masks.append(empty_mask)
    #left-1   1 2 15 14   *************************
    candidate_names = {'LF_wheel', 'LB_wheel', 'LB_roof', 'LF_roof'}
    kp_names = set()
    for cn in candidate_names:
        if cn in kp2:
            kp_names.add(cn)
    if len(kp_names) == 4:
        temp_empty_mask = np.zeros(img_size)  # should be set 0 everytime used lk
        points = get_array_of_points(kp2, ['LF_wheel', 'LB_wheel', 'LB_roof', 'LF_roof'])
        part_mask = cv2.fillConvexPoly(temp_empty_mask, points, 1)
        masks.append(part_mask)
    else:
        masks.append(empty_mask)

    # 1 	LF_wheel	11 L_mirror
    # 2 	LB_wheel 	12 R_mirror
    # 3 	RF_wheel 	13 RF_roof
    # 4 	RB_wheel 	14 LF_roof
    # 5 	R_fog_lamp 	15 LB_roof
    # 6 	L_fog_lamp 	16 RB_roof
    # 7 	R_head_light 	17 	LB_lamp
    # 8 	L_head_light 	18 	RB_lamp
    # 9 	F_auto_logo 	19 	B_auto_logo
    # 10 	F_license_plate 20 	B_license_plate
    # right-1   3 4 16 13   *************************
    candidate_names = {'RF_wheel', 'RB_wheel', 'RB_roof', 'RF_roof'}
    kp_names = set()
    for cn in candidate_names:
        if cn in kp2:
            kp_names.add(cn)
    if len(kp_names) == 4:
        temp_empty_mask = np.zeros(img_size)  # should be set 0 everytime used lk
        points = get_array_of_points(kp2, ['RF_wheel', 'RB_wheel', 'RB_roof', 'RF_roof'])
        part_mask = cv2.fillConvexPoly(temp_empty_mask, points, 1)
        masks.append(part_mask)
    else:
        masks.append(empty_mask)


    # def mask_joint(fr, to, inc_to):
    #     if not check_keypoints_present(kp2, [fr, to]):
    #         return empty_mask
    #     return skimage.measure.grid_points_in_poly(img_size, estimate_polygon(kp2[fr], kp2[to], st2, inc_to, 0.1, 0.2, 0.2)[:, ::-1])

    #masks.append(mask_joint('Rhip', 'Rkne', 0.1))
    #masks.append(mask_joint('Lhip', 'Lkne', 0.1))

    #masks.append(empty_mask)
    #masks.append(empty_mask)

    masks.append(empty_mask)
    masks.append(empty_mask)

    masks.append(empty_mask)
    masks.append(empty_mask)

    return np.array(masks)


def estimate_polygon(fr, to, st, inc_to, inc_from, p_to, p_from):
    fr = fr + (fr - to) * inc_from
    to = to + (to - fr) * inc_to

    norm_vec = fr - to
    norm_vec = np.array([-norm_vec[1], norm_vec[0]])
    norm = np.linalg.norm(norm_vec)
    if norm == 0:
        return np.array([
            fr + 1,
            fr - 1,
            to - 1,
            to + 1,
        ])
    norm_vec = norm_vec / norm
    vetexes = np.array([
        fr + st * p_from * norm_vec,
        fr - st * p_from * norm_vec,
        to - st * p_to * norm_vec,
        to + st * p_to * norm_vec
    ])

    return vetexes

def perspective_transforms(array1, array2):
    kp1 = give_name_to_keypoints(array1)
    kp2 = give_name_to_keypoints(array2)

    no_point_tr = np.array([[1, 0, 1000], [0, 1, 1000], [0, 0, 1]])

    transforms = []
    def to_transforms(tr):
        from numpy.linalg import LinAlgError
        try:
            np.linalg.inv(tr)
            transforms.append(tr)
        except LinAlgError:
            transforms.append(no_point_tr)

    #roof    **************************
    roof_candidate_names = {'RF_roof', 'LF_roof', 'LB_roof', 'RB_roof'}  # lk
    roof_kp_names = set()
    for cn in roof_candidate_names:
        if cn in kp1 and cn in kp2:
            roof_kp_names.add(cn)
    if len(roof_kp_names) == 4:
        # if len(head_kp_names) < 3:
        # head_kp_names.add('Lsho')
        # head_kp_names.add('Rsho')
        roof_poly_1 = get_array_of_points(kp1, ['RF_roof', 'LF_roof', 'LB_roof', 'RB_roof'])
        roof_poly_2 = get_array_of_points(kp2, ['RF_roof', 'LF_roof', 'LB_roof', 'RB_roof'])
        tr = cv2.getPerspectiveTransform(np.float32(roof_poly_2), np.float32(roof_poly_1))
        to_transforms(tr)

    else:
        to_transforms(no_point_tr)

    #head_up 7 8 14 13    **************************
    head_up_candidate_names = {'R_head_light', 'L_head_light', 'LF_roof', 'RF_roof'}  # lk
    head_up_kp_names = set()
    for cn in head_up_candidate_names:
        if cn in kp1 and cn in kp2:
            head_up_kp_names.add(cn)
    if len(head_up_kp_names) == 4:
        # if len(head_kp_names) < 3:
        # head_kp_names.add('Lsho')
        # head_kp_names.add('Rsho')
        head_up_poly_1 = get_array_of_points(kp1, ['R_head_light', 'L_head_light', 'LF_roof', 'RF_roof'])
        head_up_poly_2 = get_array_of_points(kp2, ['R_head_light', 'L_head_light', 'LF_roof', 'RF_roof'])
        # tr = skimage.transform.estimate_transform('projective', src=head_poly_2, dst=head_poly_1)
        # to_transforms(tr.params)
        tr = cv2.getPerspectiveTransform(np.float32(head_up_poly_2), np.float32(head_up_poly_1))
        to_transforms(tr)
    else:
        to_transforms(no_point_tr)


    #head_down    **************************
    head_candidate_names = {'R_fog_lamp', 'L_fog_lamp', 'L_head_light', 'R_head_light'}#lk
    head_kp_names = set()
    for cn in head_candidate_names:
        if cn in kp1 and cn in kp2:
            head_kp_names.add(cn)
    if len(head_kp_names) == 4:
        #if len(head_kp_names) < 3:
        #head_kp_names.add('Lsho')
        #head_kp_names.add('Rsho')
        head_poly_1 = get_array_of_points(kp1, ['R_fog_lamp', 'L_fog_lamp', 'L_head_light', 'R_head_light'])
        head_poly_2 = get_array_of_points(kp2, ['R_fog_lamp', 'L_fog_lamp', 'L_head_light', 'R_head_light'])
        # tr = skimage.transform.estimate_transform('projective', src=head_poly_2, dst=head_poly_1)
        # to_transforms(tr.params)
        tr = cv2.getPerspectiveTransform(np.float32(head_poly_2), np.float32(head_poly_1))
        to_transforms(tr)
    else:
        to_transforms(no_point_tr)


    #back_up    **************************
    candidate_names = {'LB_lamp', 'RB_lamp', 'RB_roof', 'LB_roof'}  # lk
    kp_names = set()
    for cn in candidate_names:
        if cn in kp1 and cn in kp2:
            kp_names.add(cn)
    if len(kp_names) == 4:
        # if len(head_kp_names) < 3:
        # head_kp_names.add('Lsho')
        # head_kp_names.add('Rsho')
        part_poly_1 = get_array_of_points(kp1, ['LB_lamp', 'RB_lamp', 'RB_roof', 'LB_roof'])
        part_poly_2 = get_array_of_points(kp2, ['LB_lamp', 'RB_lamp', 'RB_roof', 'LB_roof'])
        # tr = skimage.transform.estimate_transform('projective', src=head_poly_2, dst=head_poly_1)
        # to_transforms(tr.params)
        tr = cv2.getPerspectiveTransform(np.float32(part_poly_2), np.float32(part_poly_1))
        to_transforms(tr)
    else:
        to_transforms(no_point_tr)

    #'LF_wheel', 'LB_wheel', 'LB_roof', 'LF_roof'
    #left-1   1 2 15 14   *************************
    candidate_names = {'LF_wheel', 'LB_wheel', 'LB_roof', 'LF_roof'}  # lk
    kp_names = set()
    for cn in candidate_names:
        if cn in kp1 and cn in kp2:
            kp_names.add(cn)
    if len(kp_names) == 4:
        # if len(head_kp_names) < 3:
        # head_kp_names.add('Lsho')
        # head_kp_names.add('Rsho')
        part_poly_1 = get_array_of_points(kp1, ['LF_wheel', 'LB_wheel', 'LB_roof', 'LF_roof'])
        part_poly_2 = get_array_of_points(kp2, ['LF_wheel', 'LB_wheel', 'LB_roof', 'LF_roof'])
        # tr = skimage.transform.estimate_transform('projective', src=head_poly_2, dst=head_poly_1)
        # to_transforms(tr.params)
        tr = cv2.getPerspectiveTransform(np.float32(part_poly_2), np.float32(part_poly_1))
        to_transforms(tr)
    else:
        to_transforms(no_point_tr)

    #'RF_wheel', 'RB_wheel', 'RB_roof', 'RF_roof'
    # right-1   3 4 16 13   *************************
    candidate_names = {'RF_wheel', 'RB_wheel', 'RB_roof', 'RF_roof'}  # lk
    kp_names = set()
    for cn in candidate_names:
        if cn in kp1 and cn in kp2:
            kp_names.add(cn)
    if len(kp_names) == 4:
        # if len(head_kp_names) < 3:
        # head_kp_names.add('Lsho')
        # head_kp_names.add('Rsho')
        part_poly_1 = get_array_of_points(kp1, ['RF_wheel', 'RB_wheel', 'RB_roof', 'RF_roof'])
        part_poly_2 = get_array_of_points(kp2, ['RF_wheel', 'RB_wheel', 'RB_roof', 'RF_roof'])
        # tr = skimage.transform.estimate_transform('projective', src=head_poly_2, dst=head_poly_1)
        # to_transforms(tr.params)
        tr = cv2.getPerspectiveTransform(np.float32(part_poly_2), np.float32(part_poly_1))
        to_transforms(tr)
    else:
        to_transforms(no_point_tr)

    # def estimate_join(fr, to, inc_to):
    #     if not check_keypoints_present(kp2, [fr, to]):
    #         return no_point_tr
    #     poly_2 = estimate_polygon(kp2[fr], kp2[to], st2, inc_to, 0.1, 0.2, 0.2)
    #     if check_keypoints_present(kp1, [fr, to]):
    #         poly_1 = estimate_polygon(kp1[fr], kp1[to], st1, inc_to, 0.1, 0.2, 0.2)
    #     else:
    #         if fr[0]=='R':
    #             fr = fr.replace('R', 'L')
    #             to = to.replace('R', 'L')
    #         else:
    #             fr = fr.replace('L', 'R')
    #             to = to.replace('L', 'R')
    #         if check_keypoints_present(kp1, [fr, to]):
    #             poly_1 = estimate_polygon(kp1[fr], kp1[to], st1, inc_to, 0.1, 0.2, 0.2)
    #         else:
    #             return no_point_tr
    #     return skimage.transform.estimate_transform('projective', dst=poly_1, src=poly_2).params

    to_transforms(no_point_tr)
    to_transforms(no_point_tr)

    to_transforms(no_point_tr)
    to_transforms(no_point_tr)

    return np.array(transforms).reshape((-1, 9))[..., :-1]


