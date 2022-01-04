import os
import time
import random
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.optim as optim
import torchvision.models as models
from tools.set_seed import set_seed
from tools.my_dataset import MyDataset
from tools.unet import UNet
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
set_seed()  # 设置随机种子

def compute_dice(y_pred, y_true):
    """
    :param y_pred: 4-d tensor, value = [0,1]
    :param y_true: 4-d tensor, value = [0,1]
    :return:
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

def get_img_name(img_dir, type_arg):
    """
    获取文件夹下format格式的文件名
    :param img_dir: str
    :param format: str
    :return: list
    """
    file_names = os.listdir(img_dir)
    img_names = list(filter(lambda x: x.endswith(type_arg), file_names))
    img_names = list(filter(lambda x: not x.endswith("matte.png"), img_names))

    if len(img_names) < 1:
        raise ValueError("{}下找不到{}格式数据".format(img_dir, type_arg))
    return img_names

def get_model(m_path):
    unet = UNet(in_channels=3, out_channels=1, init_features=32)
    checkpoint = torch.load(m_path)
    print(type(checkpoint))
    # remove module.
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        namekey = k[7:] if k.startswith('module.') else k
        new_state_dict[namekey] = v

    unet.load_state_dict(new_state_dict)

    return unet

def Inference(args,device):
    print("start Inference")
    img_dir = "{}/data/{}".format(BASE_DIR,args.p)
    model_path = "{}/{}".format(BASE_DIR,args.m)
    output_path = "{}/{}".format(BASE_DIR, args.o)
    time_total = 0
    num_infer = 99999
    mask_thres = .5

    # 1. data
    img_names = get_img_name(img_dir, type_arg=args.i)
    random.shuffle(img_names)
    num_img = len(img_names)

    # 2. model
    unet = get_model(model_path)
    unet.to(device)
    unet.eval()

    for idx, img_name in enumerate(img_names):
        if idx > num_infer:
            break

        path_img = os.path.join(img_dir, img_name)

        # step 1/4 : path --> img_chw
        img_hwc = Image.open(path_img).convert('RGB')
        img_save = img_hwc.copy()
        img_hwc = img_hwc.resize((args.s, args.s))
        img_arr = np.array(img_hwc)
        img_chw = img_arr.transpose((2, 0, 1))

        # step 2/4 : img --> tensor
        img_tensor = torch.tensor(img_chw).to(torch.float)
        img_tensor.unsqueeze_(0)
        img_tensor = img_tensor.to(device)

        # step 3/4 : tensor --> features
        time_tic = time.time()
        with torch.no_grad():
            outputs = unet(img_tensor)

        time_toc = time.time()

        # step 4/4 : visualization
        mask_pred = outputs.ge(0.5).cpu().data.numpy().astype("uint8")
        Image.fromarray(mask_pred.squeeze()).resize((1920, 1200)).save(
            "{}/{}/{}_mask.png".format(BASE_DIR,args.o,img_name.split('.')[0]))

        img_save.save("{}/{}".format(output_path,img_name))

        time_s = time_toc - time_tic
        time_total += time_s

        print('{:d}/{:d}: {} {:.3f}s '.format(idx + 1, num_img, img_name, time_s))

    return output_path


def ReadImgPath(path, file_names,type_arg):
    pre_imgs = {}
    for file_name in file_names:
        if ('_mask' in file_name.split('.')[0]):
            jpg_name = file_name.replace("_mask.png", type_arg)
            pre_imgs["{}/{}".format(path, jpg_name)] = "{}/{}".format(path, file_name)
    return pre_imgs

def DelCells(img_np, label_np,ColorCode):
    r_img_np = img_np * label_np
    r_img_np[r_img_np == 0] = ColorCode
    return Image.fromarray(r_img_np)

def Paths2Img(paths_dict,save_path,ColorCode):
    total = len(paths_dict)
    for num, img in enumerate(paths_dict):
        jpg = Image.open(img).convert('L')
        mask = Image.open(paths_dict[img]).convert('L')
        jpg_np = np.array(jpg)
        mask_np = np.array(mask)
        CellsDeled = DelCells(jpg_np, mask_np,ColorCode)
        img_name = img.split('/')[-1]
        CellsDeled.convert('L').save("{}/{}".format(save_path,img_name))
        print("{}/{} {}".format(num,total,img_name))

def Go2DelCells(args,outpath):
    ColorCode = args.c
    print("now start to delete cells")
    dir_path = outpath
    save_path = "{}/withoutcells".format(BASE_DIR)
    pre_mess = ReadImgPath(dir_path, os.listdir(dir_path),type_arg=args.i)
    Paths2Img(pre_mess,save_path,ColorCode)


if __name__ == '__main__':
    base_dir = os.getcwd()
    print("l")
    parsers = argparse.ArgumentParser()
    parsers.add_argument('-d',type=str,default='cuda',help="cpu or cuda")
    parsers.add_argument('-i',type=str,default='.jpg',help=".jpg or .png or .jpeg")
    parsers.add_argument('-p',type=str,default='data',help="path of datas' dir")
    parsers.add_argument('-m',type=str,default='models/cells_100_type2.pkl',help="path of you want to use model")
    parsers.add_argument('-o',type=str,default='outputs')
    parsers.add_argument('-s',type=int,default=512,help="masks imgs' size")
    parsers.add_argument('-c',type=int,default=255,help="color to input del part")
    args = parsers.parse_args()

    if(args.d == 'cpu'):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OutputPath = Inference(args,device)
    print("Inference over")
    Go2DelCells(args,OutputPath)
    print("Delete cells over")


