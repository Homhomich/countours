import math

import cv2 as cv
import numpy as np
import pandas as pd
import os
import random as random
from sklearn import preprocessing

def normalize(width, height, area, length, eccentricity):
    x_array = np.array([width, height, area, length, eccentricity])
    n_arr = preprocessing.normalize([x_array])
    return n_arr[0][0], n_arr[0][1], n_arr[0][2], n_arr[0][3], n_arr[0][4]


def rotate_image(image, angle):
    height, width = image.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv.warpAffine(image, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def resize_img(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


def to_exel(file_name, width_arr, height_arr, area_arr, length_arr, eccentricity_arr, class_value):
    inner_df = pd.read_excel(file_name)
    df = pd.DataFrame({'Ширина': width_arr,
                       'Высота': height_arr,
                       'Площадь': area_arr,
                       'Периметр': length_arr,
                       'Эксцентриситет': eccentricity_arr,
                       'Результат': class_value,
                       })

    frames = [inner_df, df]
    result_df = pd.concat(frames, sort=False)
    result_df.to_excel(file_name, index=False)


def find_counts(img, class_value):
    width_arr = list()
    height_arr = list()
    area_arr = list()
    length_arr = list()
    cx_arr = list()
    cy_arr = list()
    eccentricity_arr = list()
    roundness_arr = list()

    for i in range(1, len(img)):
        for j in range(len(img[i])):
            value_to_minus = img[0][j]
            for k in range(len(img[i][j])):
                if int(img[i][j][k]) < int(value_to_minus[k]):
                    img[i][j][k] = int(value_to_minus[k]) - int(img[i][j][k])
                else:
                    img[i][j][k] = int(img[i][j][k]) - int(value_to_minus[k])
            if img[i][j][2] < 4:
                img[i][j] = [255, 255, 255]

    cv.imwrite('/results_pics/result.bmp', img)

    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = 100
    ret, thresh_img = cv.threshold(img_grey, thresh, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    updated_contours = []

    for contour in contours:
        area = cv.contourArea(contour)
        length = cv.arcLength(contour, True)
        eccentricity = 0
        roundness = 0

        if len(contour) >= 5:
            ellipse = cv.fitEllipse(contour)
            width_ellipse = ellipse[1][0]
            height_ellipse = ellipse[1][1]
            majorAxis = height_ellipse if height_ellipse > width_ellipse else width_ellipse
            minorAxis = width_ellipse if height_ellipse > width_ellipse else height_ellipse
            eccentricity = math.sqrt(1 - pow(minorAxis / majorAxis, 2))
            roundness = pow(length, 2) / (2 * 3.1416 * area)  # Гладкость

        M = cv.moments(contour)
        rect = cv.minAreaRect(contour)
        (x, y), (width, height), angle = rect
        cx = 0
        cy = 0
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

        if (area > 10) & (area < 10000):
            n_width, n_height, n_area, n_length, n_eccentricity = normalize(width, height, area, length, eccentricity)
            width_arr.append(n_width)
            length_arr.append(n_length)
            height_arr.append(n_height)
            area_arr.append(n_area)
            eccentricity_arr.append(n_eccentricity)

            #  Без нормализации
            cx_arr.append(cx)
            cy_arr.append(cy)
            roundness_arr.append(roundness)
            updated_contours.append(contour)

    to_exel('./excels/rice_withOUT_spreading.xlsx', width_arr, height_arr, area_arr, length_arr, eccentricity_arr,
            class_value)
    img_contours = np.zeros(img.shape)
    cv.drawContours(img_contours, updated_contours, -1, (0, 255, 0), 1)
    cv.imwrite('result_countours.bmp', img_contours)
    print('hi')


if __name__ == '__main__':
    dir_name = "good_pics"
    pics_dir = os.listdir(dir_name)
    randomNumber = random.randint(1, 10)

    for item in [1, 2, 3, 4, 5, 6, 7]:
        image = cv.imread('good_pics/' + pics_dir[item])
        if item > 200:
            print('SPEADING')
            resizeAmount = random.randint(50, 70)
            angle = random.randint(30, 50)

            changed_img = resize_img(image, resizeAmount)
            changed_img = rotate_image(changed_img, angle)
            find_counts(changed_img, 1)
        find_counts(image, 1)
    print('hi1')
