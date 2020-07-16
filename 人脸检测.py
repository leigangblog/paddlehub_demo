# -*- coding: utf-8 -*-
import paddlehub as hub
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def show_img(res_img_path):
    img = mpimg.imread(res_img_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


mask_detector = hub.Module(name="pyramidbox_lite_mobile_mask")
res = mask_detector.face_detection(paths=['imgs/mask.jpg'], visualization=True, output_dir='detection_output')
res_img_path = 'detection_output/mask.jpg'
show_img(res_img_path)


