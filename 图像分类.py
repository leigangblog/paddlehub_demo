# -*- coding: utf-8 -*-
import paddlehub as hub

classifier = hub.Module(name="resnet50_vd_animals")
result = classifier.classification(paths=['imgs/img_cls.jpg'])
print(result)
