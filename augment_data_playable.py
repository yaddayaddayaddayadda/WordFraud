import numpy as np
import cv2
import glob
import os
import copy

SRC_DIRECTORY = "training_data_orig_playable_2532"
WHITE = 255.0
c=0
for file in glob.glob(SRC_DIRECTORY+"/*.jpg", recursive=True):
    letter = file.split("/")[1][0]
    img = cv2.imread(file)
    images = []
    images.append(img)
    for blur in range(1, 10, 2):
        blurred_img = cv2.GaussianBlur(img, (blur, blur), blur)
        images.append(blurred_img)
    X, Y = img.shape[1], img.shape[0]
    for image in images:
        for x_shift in range(-7, 7):
            for y_shift in range(-7,7):
                M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
                shifted = cv2.warpAffine(image, M, (X, Y))
                if x_shift < 0:
                    shifted[:, X + x_shift:] = 255.0
                else:
                    shifted[:, :x_shift] = 255.0
                if y_shift < 0:
                    shifted[Y+y_shift:, :] = 255.0
                else:
                    shifted[:y_shift,:] = 255.0
                shifted = cv2.normalize(shifted, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(f'training_data_playable_2532_augmented/{letter}_{c}.jpg', shifted)
                c+=1   