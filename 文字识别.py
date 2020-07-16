# -*- coding: utf-8 -*-
import paddlehub as hub

ocr = hub.Module(name="chinese_ocr_db_crnn_server")
result = ocr.recognize_text(paths=['imgs/text_detection.jpg'],visualization=True,output_dir='ocr_result')
# print(result)
print(result[0]['data'])