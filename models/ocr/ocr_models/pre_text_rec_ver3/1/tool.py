import math
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image


def resize_norm_img(img, max_wh_ratio):
    imgC, imgH, imgW = [3, 48, 320]
    assert imgC == img.shape[2]
    imgW = int((32 * max_wh_ratio))

    w = 100
    if w is not None and w > 0:
        imgW = w
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype("float32")
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def resize_norm_img_srn(img, image_shape):
    imgC, imgH, imgW = image_shape

    img_black = np.zeros((imgH, imgW))
    im_hei = img.shape[0]
    im_wid = img.shape[1]

    if im_wid <= im_hei * 1:
        img_new = cv2.resize(img, (imgH * 1, imgH))
    elif im_wid <= im_hei * 2:
        img_new = cv2.resize(img, (imgH * 2, imgH))
    elif im_wid <= im_hei * 3:
        img_new = cv2.resize(img, (imgH * 3, imgH))
    else:
        img_new = cv2.resize(img, (imgW, imgH))

    img_np = np.asarray(img_new)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    img_black[:, 0 : img_np.shape[1]] = img_np
    img_black = img_black[:, :, np.newaxis]

    row, col, c = img_black.shape
    c = 1

    return np.reshape(img_black, (c, row, col)).astype(np.float32)


def srn_other_inputs(image_shape, num_heads, max_text_length):

    imgC, imgH, imgW = image_shape
    feature_dim = int((imgH / 8) * (imgW / 8))

    encoder_word_pos = (
        np.array(range(0, feature_dim))
        .reshape((feature_dim, 1))
        .astype("int64")
    )
    gsrm_word_pos = (
        np.array(range(0, max_text_length))
        .reshape((max_text_length, 1))
        .astype("int64")
    )

    gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
    gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
        [-1, 1, max_text_length, max_text_length]
    )
    gsrm_slf_attn_bias1 = np.tile(
        gsrm_slf_attn_bias1, [1, num_heads, 1, 1]
    ).astype("float32") * [-1e9]

    gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
        [-1, 1, max_text_length, max_text_length]
    )
    gsrm_slf_attn_bias2 = np.tile(
        gsrm_slf_attn_bias2, [1, num_heads, 1, 1]
    ).astype("float32") * [-1e9]

    encoder_word_pos = encoder_word_pos[np.newaxis, :]
    gsrm_word_pos = gsrm_word_pos[np.newaxis, :]

    return [
        encoder_word_pos,
        gsrm_word_pos,
        gsrm_slf_attn_bias1,
        gsrm_slf_attn_bias2,
    ]


def process_image_srn(img, image_shape, num_heads, max_text_length):
    norm_img = resize_norm_img_srn(img, image_shape)
    norm_img = norm_img[np.newaxis, :]

    [
        encoder_word_pos,
        gsrm_word_pos,
        gsrm_slf_attn_bias1,
        gsrm_slf_attn_bias2,
    ] = srn_other_inputs(image_shape, num_heads, max_text_length)

    gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1.astype(np.float32)
    gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2.astype(np.float32)
    encoder_word_pos = encoder_word_pos.astype(np.int64)
    gsrm_word_pos = gsrm_word_pos.astype(np.int64)

    return (
        norm_img,
        encoder_word_pos,
        gsrm_word_pos,
        gsrm_slf_attn_bias1,
        gsrm_slf_attn_bias2,
    )


def resize_norm_img_sar(img, image_shape, width_downsample_ratio=0.25):
    imgC, imgH, imgW_min, imgW_max = image_shape
    h = img.shape[0]
    w = img.shape[1]
    valid_ratio = 1.0
    # make sure new_width is an integral multiple of width_divisor.
    width_divisor = int(1 / width_downsample_ratio)
    # resize
    ratio = w / float(h)
    resize_w = math.ceil(imgH * ratio)
    if resize_w % width_divisor != 0:
        resize_w = round(resize_w / width_divisor) * width_divisor
    if imgW_min is not None:
        resize_w = max(imgW_min, resize_w)
    if imgW_max is not None:
        valid_ratio = min(1.0, 1.0 * resize_w / imgW_max)
        resize_w = min(imgW_max, resize_w)
    resized_image = cv2.resize(img, (resize_w, imgH))
    resized_image = resized_image.astype("float32")
    # norm
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    resize_shape = resized_image.shape
    padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
    padding_im[:, :, 0:resize_w] = resized_image
    pad_shape = padding_im.shape

    return padding_im, resize_shape, pad_shape, valid_ratio


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and (
            _boxes[i + 1][0][0] < _boxes[i][0][0]
        ):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img
