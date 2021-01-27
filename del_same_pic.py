import PIL.Image as Image
import numpy as np
import os

def img_2_np(img):
    img_np = np.array(img)
    return img_np

def np_total(np):
    num = 0
    for y in np:
        for x in y:
                for n in x:
                    num += n
    return num

def find_same(packet_box):
    mark_box = []
    same_pic_path = []
    for num, packet in enumerate(packet_box):
        data_box = []
        for n in range(len(packet_box)):
            if(num != n):
                if(n in mark_box):
                    continue
                else:
                    data = packet[-1] / packet_box[n][-1]
                    data_box.append(data)
        if(1.0 in data_box):
            mark_box.append(num)
    for mark in mark_box:
        same_pic_path.append(packet_box[mark][0])
    return same_pic_path

if __name__ == '__main__':
    dir_path = "path" # dir path for your image
    path_box = []
    packet_box = []
    for pic_name in (os.listdir(dir_path)):
        if(pic_name.split('.')[-1] in ["JPG" , "png" ,"jpeg","jpg","JPEG","PNG"]):
            full_path = dir_path + "/" + pic_name
            path_box.append(full_path)
        else:
            continue
    for path in path_box:
        img = Image.open(path)
        img_np = img_2_np(img)
        np_total_data = np_total(img_np)
        singe_packet = (path,img,img_np,np_total_data)
        packet_box.append(singe_packet)
    same = find_same(packet_box)
    for del_path in same:
        os.remove(del_path)
    print("finish, removed {} same pic in all {} pic, same percent is {}%".format(len(same),len(path_box),(len(same)/len(path_box))*100))

    ##if this code can help your work, pls give me a star! God bless you!
    ##if you want to know that how to use this code,you can read my blog(chinese) or the Readme
