# -*- coding: utf-8 -*-
# 导入需要的库
import paddlehub as hub
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')
def show_img(res_img_path):
    img = mpimg.imread(res_img_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
module = hub.Module(name="deeplabv3p_xception65_humanseg")
res = module.segmentation(paths=["imgs/image_seg.jpg"], visualization=True, output_dir='humanseg_output')
res_img_path = 'humanseg_output/image_seg.png'
show_img(res_img_path)
