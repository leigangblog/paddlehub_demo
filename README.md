<p align="center">
 <img src="https://gitee.com/leigangblog/images/raw/master/static/20200716210654.jpg" align="middle" >
</p>
PaddleHub是飞桨生态的预训练模型应用工具，开发者可以便捷地使用高质量的预训练模型结合Fine-tune API快速完成模型迁移到部署的全流程工作。PaddleHub提供的预训练模型涵盖了图像分类、目标检测、词法分析、语义模型、情感分析、视频分类、图像生成、图像分割、文本审核、关键点检测等主流模型。更多详情可查看官网：https://www.paddlepaddle.org.cn/hub

PaddleHub以预训练模型应用为核心具备以下特点：  

* **模型即软件**，通过Python API或命令行实现模型调用，可快速体验或集成飞桨特色预训练模型。

* **易用的迁移学习**，通过Fine-tune API，内置多种优化策略，只需少量代码即可完成预训练模型的Fine-tuning。

* **一键模型转服务**，简单一行命令即可搭建属于自己的深度学习模型API服务完成部署。

* **自动超参优化**，内置AutoDL Finetuner能力，一键启动自动化超参搜索。

## 目录

* [安装](#安装)
* [使用](#使用)
* [联系作者](#联系作者)

## 安装

### 环境依赖

* Python >= 3.6
* PaddlePaddle >= 1.7.0
* 操作系统: Windows/Mac/Linux

### 安装命令

在安装PaddleHub之前，请先安装PaddlePaddle深度学习框架，更多安装说明请查阅[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick)

```shell
pip install paddlehub
```

除上述依赖外，预训练模型和数据集的下载需要网络连接，请确保机器可以**正常访问网络**。若本地已存在相关预训练模型目录，则可以离线使用PaddleHub。


## 使用
### 图像分类
```python
# -*- coding: utf-8 -*-
import paddlehub as hub

classifier = hub.Module(name="resnet50_vd_animals")
result = classifier.classification(paths=['imgs/img_cls.jpg'])
print(result)

```
参考链接：https://paddlepaddle.org.cn/hubdetail?name=resnet50_vd_animals&en_category=ImageClassification

### 人脸检测
```python
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

```
参考链接：https://paddlepaddle.org.cn/hubdetail?name=pyramidbox_lite_mobile_mask&en_category=FaceDetection

### 文字识别
```python
# -*- coding: utf-8 -*-
import paddlehub as hub

ocr = hub.Module(name="chinese_ocr_db_crnn_server")
result = ocr.recognize_text(paths=['imgs/text_detection.jpg'],visualization=True,output_dir='ocr_result')
# print(result)
print(result[0]['data'])
```
参考链接：https://paddlepaddle.org.cn/hubdetail?name=chinese_ocr_db_crnn_server&en_category=TextRecognition

### 关键点检测
```python
# -*- coding: utf-8 -*-
import paddlehub as hub
pose_estimation = hub.Module(name="human_pose_estimation_resnet50_mpii")
result = pose_estimation.keypoint_detection(paths=['imgs/pose_estimation.jpg'],visualization=True, output_dir='output_pose')
print(result)
```
参考链接：https://paddlepaddle.org.cn/hubdetail?name=human_pose_estimation_resnet50_mpii&en_category=KeyPointDetection

### 目标检测
```python
# -*- coding: utf-8 -*-
import paddlehub as hub

object_detector = hub.Module(name="yolov3_resnet50_vd_coco2017")
result = object_detector.object_detection(paths=['imgs/object_detector.jpg'], visualization=True,
                                          output_dir='detection_result')
print(result)
```
参考链接：https://paddlepaddle.org.cn/hubdetail?name=yolov3_resnet50_vd_coco2017&en_category=ObjectDetection

### 图像生成
```python
# -*- coding: utf-8 -*-
import paddlehub as hub

stylepro_artistic = hub.Module(name="stylepro_artistic")
result = stylepro_artistic.style_transfer(
    paths=[{
        'content': 'imgs/style_content1.jpg',
        'styles': ['imgs/style1.jpg']
    }], visualization=True, output_dir='transfer_result')

```
参考链接：https://www.paddlepaddle.org.cn/hubdetail?name=stylepro_artistic&en_category=GANs

### 图像分割
```python
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

```
参考链接：https://paddlepaddle.org.cn/hubdetail?name=deeplabv3p_xception65_humanseg&en_category=ImageSegmentation


##  联系作者
* 作者GitHub：https://github.com/leigangblog
* Github开源代码：https://github.com/leigangblog/paddlehub_demo

**FAQ**

当安装或者使用遇到问题时，可以通过[FAQ](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-FAQ)查找解决方案。
如果在FAQ中没有找到解决方案，欢迎您将问题以[Github Issues](https://github.com/PaddlePaddle/PaddleHub/issues)的形式提交给我们，我们会第一时间进行跟进。

## 用户交流群

* 飞桨PaddlePaddle 交流群：796771754（QQ群）
* 飞桨ERNIE交流群：760439550（QQ群）



