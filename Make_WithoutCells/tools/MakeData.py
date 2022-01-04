# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 00:15:37 2021

@author: admin
"""

import os
import PIL.Image as Image

def ReadTxt(path):
    with open(path,'r') as txt:
        return txt.readlines()

def txt2mess(txt):
    img_dir = "E:/code_csz/UNet_Dplan/data/cells/masks"
    mess = []
    for line in txt:
        line = "{}/{}_mask.png".format(img_dir,line.replace('''\n''',''))
        mess.append(line)
    return mess

def Read2Save(Mess,classes):
    new_path = "E:/code_csz/UNet_Dplan/data/cells/{}".format(classes)
    for img_path in Mess:
        img_name = img_path.split('/')[-1].replace("png","")
        img = Image.open(img_path)
        img.save("{}/{}.png".format(new_path,img_name))

if __name__ == '__main__':
    train_path = "E:/code_csz/UNet_Bplan/datasets/ImageSets/Segmentation/train.txt"
    val_path = "E:/code_csz/UNet_Bplan/datasets/ImageSets/Segmentation/val.txt"
    test_path = "E:/code_csz/UNet_Bplan/datasets/ImageSets/Segmentation/test.txt"
    
    train_mess = txt2mess(ReadTxt(train_path))
    Read2Save(train_mess,'train')
    
    val_mess = txt2mess(ReadTxt(val_path))
    Read2Save(val_mess,'valid')
    
    test_mess = txt2mess(ReadTxt(test_path))
    Read2Save(test_mess,'test')
    