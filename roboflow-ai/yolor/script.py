%matplotlib inline
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.google_utils import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, clip_coords, set_logging, increment_path
from utils.loss import compute_loss
from utils.metrics import ap_per_class
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized

from models.models import *

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

# You must run this one, the follwoing code relys on this opt object
from argparse import Namespace
opt = Namespace(augment=False, batch_size=32, cfg='./cfg/yolor_p6.cfg', conf_thres=0.5, data='../zero-waste-1/data.yaml', device='0', exist_ok=False, img_size=416, iou_thres=0.65, name='exp', names='./data/zerowaste.names', project='../runs/test', save_conf=True, save_json=True, save_txt=False, single_cls=False, task='test', verbose=True, weights=['../runs/train/yolor_p6_2022_03_26-10_44_07/weights/best_overall.pt'])

# ===========================> The following are some params will be passed to test() funciton
# weights=["../runs/train/yolor_p6_2022_03_26-10_44_07/weights/best_overall.pt"]
data = "../zero-waste-1/data.yaml"	# Because this file will load some utils module, so this file must be put under yolor folder
batch_size=32

imgsz=640
# conf_thres=0.5	# object confidence threshold
# iou_thres=0.6,  # IOU threshold for NMS, or .65
# task='test'
device='0'	# help='cuda device, i.e. 0 or 0,1,2,3 or cpu'
single_cls=False
# augment=False
verbose=True
# save_txt=False	# for auto-labelling
# save_conf=True
# save_json=True

project='../runs/test'
name='exp'
# exist_ok=False
# cfg='./cfg/yolor_p6.cfg'
names='./data/zerowaste.names'

# model=None
log_imgs=0  	# number of logged imag
save_dir=Path('./')  # for saving images
plots=True
dataloader=None

set_logging()
device = select_device(opt.device, batch_size=opt.batch_size)
save_txt = opt.save_txt  # save *.txt labels

# Directories: where all the logging file will be saved for this experiment
save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

# Load model
model = Darknet(opt.cfg).to(device)

try:
    ckpt = torch.load(opt.weights[0], map_location=device)  # load checkpoint
    ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)
except:
    load_darknet_weights(model, opt.weights[0])
imgsz = check_img_size(imgsz, s=64)  # check img_size

# Half: half precision if we are running on GPU
half = device.type != 'cpu'
if half:
    model.half()

# Configure
yaml_dir='../zero-waste-1/data.yaml'
model.eval()
is_coco = yaml_dir.endswith('coco.yaml')  # is COCO dataset
with open(yaml_dir) as f:
    data = yaml.load(f, Loader=yaml.FullLoader)  # model dict

print(data)	# Making sure you are running this file under yolor folder

check_dataset(data)  # check
nc = 1 if single_cls else int(data['nc'])  # number of classes
iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
niou = iouv.numel()
