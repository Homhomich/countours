import cv2 as cv
import numpy as np
import pandas as pd


def to_exel(width_arr, height_arr, area_arr, length_arr, cx_arr, cy_arr):
    df = pd.DataFrame({'Ширина': width_arr,
                       'Высота': height_arr,
                       'Площать': area_arr,
                       'Периметр': length_arr,
                       'CX': cx_arr,
                       'CY': cy_arr
                       })
    df.to_excel('./ricegood.xlsx')


def find_counts():
    width_arr = list()
    height_arr = list()
    area_arr = list()
    length_arr = list()
    cx_arr = list()
    cy_arr = list()

    img = cv.imread('good_rice.bmp')
    for i in range(1, len(img)):
        for j in range(len(img[i])):
            value_to_minus = img[0][j]
            for k in range(len(img[i][j])):
                if int(img[i][j][k]) < int(value_to_minus[k]):
                    img[i][j][k] = int(value_to_minus[k]) - int(img[i][j][k])
                else:
                    img[i][j][k] = int(img[i][j][k]) - int(value_to_minus[k])
            if img[i][j][2] < 7:
                img[i][j] = [255, 255, 255]

    cv.imwrite('result.bmp', img)

    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = 100
    ret, thresh_img = cv.threshold(img_grey, thresh, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv.contourArea(contour)
        length = cv.arcLength(contour, True)
        M = cv.moments(contour)
        rect = cv.minAreaRect(contour)
        (x, y), (width, height), angle = rect
        cx = 0
        cy = 0
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

        if (area > 10) & (area < 10000):
            width_arr.append(width)
            length_arr.append(length)
            height_arr.append(height)
            area_arr.append(area)
            cx_arr.append(cx)
            cy_arr.append(cy)

    to_exel(width_arr, height_arr, area_arr, length_arr, cx_arr, cy_arr)
    img_contours = np.zeros(img.shape)
    cv.drawContours(img_contours, contours, -1, (0, 255, 0), 1)
    cv.imwrite('result_countours.bmp', img_contours)


if __name__ == '__main__':
    find_counts()
