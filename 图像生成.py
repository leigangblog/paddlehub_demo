# -*- coding: utf-8 -*-
import paddlehub as hub

stylepro_artistic = hub.Module(name="stylepro_artistic")
result = stylepro_artistic.style_transfer(
    paths=[{
        'content': 'imgs/style_content1.jpg',
        'styles': ['imgs/style1.jpg']
    }], visualization=True, output_dir='transfer_result')
