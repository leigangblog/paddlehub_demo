# -*- coding: utf-8 -*-
import paddlehub as hub

object_detector = hub.Module(name="yolov3_resnet50_vd_coco2017")
result = object_detector.object_detection(paths=['imgs/object_detector.jpg'], visualization=True,
                                          output_dir='detection_result')
print(result)
