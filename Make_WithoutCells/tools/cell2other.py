from PIL import Image
import shutil
import os
import numpy as np

def Get_up(path):
    path_group = path.split('\\')
    del path_group[-1]
    path_str = ""
    for i in path_group:
        path_str += "{}/".format(i)
    return path_str

def make_new_mask(path):
    imgs = os.listdir(path)
    img_pakcet = []
    for img in imgs:
        if("_mask" in img):
            img_path = "{}/{}".format(path,img)
            img = Image.open(img_path)
            img_np = np.array(img)
            img_np[img_np == 0] = 2
            img_np[img_np == 255] = 0
            img_np[img_np == 2] = 255
            img = Image.fromarray(img_np)
            img.save(img_path)


if __name__ == '__main__':
    path = "{}{}".format(Get_up(os.getcwd()),"data/cells_old")
    for u_dir in os.listdir(path):
        full_dir_path = "{}/{}".format(path,u_dir)
        make_new_mask(full_dir_path)

