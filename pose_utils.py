import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.draw import circle, line_aa, polygon
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import skimage.measure, skimage.transform
import sys
import scipy.misc
import cv2

LIMB_SEQ = [[12,15], [15,14], [14,13], [13,12], [12,6], [6,7], [4,6], [5,7],
           [4,5], [16,17], [14,16], [15,17], [8,9], [18,19], [7,16],
           [7,13], [16,14], [6,17], [6,12],[15,17],[7,5],[5,1],[1,16],[6,4],[4,3],[3,17]]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85], [255, 0, 85], [255, 0, 85], [255, 0, 85], [255, 0, 85], [255, 0, 85], [255, 0, 85]]

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

LABELS = ['LF_wheel', 'LB_wheel', 'RF_wheel', 'RB_wheel', 'R_fog_lamp', 'L_fog_lamp', 'R_head_light', 'L_head_light',
               'F_auto_logo', 'F_license_plate', 'L_mirror', 'R_mirror', 'RF_roof', 'LF_roof', 'LB_roof', 'RB_roof', 'LB_lamp', 'RB_lamp','B_auto_logo','B_license_plate']

MISSING_VALUE = -1


def map_to_cord(pose_map, threshold=0.1):
    all_peaks = [[] for i in range(20)]
    pose_map = pose_map[..., :20]

    y, x, z = np.where(np.logical_and(pose_map == pose_map.max(axis = (0, 1)),
                                     pose_map > threshold))
    for x_i, y_i, z_i in zip(x, y, z):
        all_peaks[z_i].append([x_i, y_i])

    x_values = []
    y_values = []

    for i in range(20):
        if len(all_peaks[i]) != 0:
            x_values.append(all_peaks[i][0][0])
            y_values.append(all_peaks[i][0][1])
        else:
            x_values.append(MISSING_VALUE)
            y_values.append(MISSING_VALUE)

    return np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)


def cords_to_map(cords, img_size, sigma=6):
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
    return result


def draw_pose_from_cords(pose_joints, img_size, radius=2, draw_joints=True):
    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    mask = np.zeros(shape=img_size, dtype=bool)

    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            colors[yy, xx] = np.expand_dims(val, 1) * 255
            mask[yy, xx] = True

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]
        mask[yy, xx] = True

    return colors, mask

def draw_pose_from_cords_on_img(pose_joints, img, img_size, radius=2, draw_joints=True):
    # colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    colors = cv2.resize(img, img_size)
    mask = np.zeros(shape=img_size, dtype=bool)

    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            colors[yy, xx] = np.expand_dims(val, 1) * 255
            mask[yy, xx] = True

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]
        mask[yy, xx] = True

    return colors, mask


def draw_pose_from_map(pose_map, threshold=0.1, **kwargs):
    cords = map_to_cord(pose_map, threshold=threshold)
    return draw_pose_from_cords(cords, pose_map.shape[:2], **kwargs)


def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def mean_inputation(X):
    X = X.copy()
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
            val = np.mean(X[:, i, j][X[:, i, j] != -1])
            X[:, i, j][X[:, i, j] == -1] = val
    return X

def draw_legend():
    handles = [mpatches.Patch(color=np.array(color) / 255.0, label=name) for color, name in zip(COLORS, LABELS)]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def produce_ma_mask(kp_array, img_size, point_radius=4):
    x1=255
    y1=255
    mask = np.zeros(shape=img_size, dtype=bool)
    from skimage.morphology import dilation, erosion, square
    for i in range(20):
        to_missing = kp_array[i][0] == MISSING_VALUE or kp_array[i][1] == MISSING_VALUE
        if to_missing:
            continue
        if kp_array[i][0]<x1:
            x1=kp_array[i][0]
        if kp_array[i][1]<y1:
            y1=kp_array[i][1]


    x2= np.max(kp_array[:, 0])
    y2= np.max(kp_array[:, 1])
    yy, xx = polygon(range(x1,x2), range(y1,y2), shape=img_size)
    mask[x1:x2, y1:y2] = True

    mask = dilation(mask, square(5))
    mask = erosion(mask, square(5))

    return mask

