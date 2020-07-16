# -*- coding: utf-8 -*-
import paddlehub as hub
pose_estimation = hub.Module(name="human_pose_estimation_resnet50_mpii")
result = pose_estimation.keypoint_detection(paths=['imgs/pose_estimation.jpg'],visualization=True, output_dir='output_pose')
print(result)