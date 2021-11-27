import cv2

from skimage import measure, color

from skimage.measure import regionprops

import numpy as np
import matplotlib.pyplot as plt
import os

import copy

def skimageFilter(gray):

    binary_warped = copy.copy(gray)
    plt.imshow(gray) 


    binary_warped[binary_warped > 0.1] = 255
    gray = 2*gray

    return binary_warped, gray



def moveImageTodir(path,targetPath,name):

    if os.path.isdir(path):

        image_name = "gt_image/"+str(name)+".png"

        binary_name = "gt_binary_image/"+str(name)+".png"

        instance_name = "gt_instance_image/"+str(name)+".png"

        train_rows = image_name + " " + binary_name + " " + instance_name + "\n"

        origin_img = cv2.imread(path+"/img.png")

        origin_img = cv2.resize(origin_img, (1280,720))

        cv2.imwrite(targetPath+"/"+image_name, origin_img)

        img = cv2.imread(path+'/label.png')

        img = cv2.resize(img, (1280,720))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        binary_warped, instance = skimageFilter(gray)

        cv2.imwrite(targetPath+binary_name, binary_warped)

        cv2.imwrite(targetPath+instance_name, instance)

        cv2.waitKey()

        cv2.destroyAllWindows()

        print("success create data name is : ", train_rows)

        return train_rows

    # return train_rows

if __name__ == "__main__":



    count = 1

    with open("./train.txt", 'w+') as file:
        dir_name = "./annotations"
        tool = os.listdir(dir_name)
        for annotations_dir in os.listdir(dir_name):

            json_dir = os.path.join(dir_name, annotations_dir)

            if os.path.isdir(json_dir):

                train_rows = moveImageTodir(json_dir, "./", str(count).zfill(4))

                file.write(train_rows)

                count += 1
