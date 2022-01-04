# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 00:36:38 2021

@author: admin
"""

import os
from PIL import Image

if __name__ == '__main__':
    
    new_path = "E:/code_csz/UNet_Dplan/data/cells/"
    for classes in ['train','valid','test']:
        f_path = "{}/{}".format(new_path, classes)
        dirlist = os.listdir(f_path)
        for name in dirlist:
            if(name.split('_')[-1] == "mask..png"):
                new_name = name.replace("mask..png","mask.png")
                os.rename("{}/{}".format(f_path, name), "{}/{}".format(f_path,new_name))
        