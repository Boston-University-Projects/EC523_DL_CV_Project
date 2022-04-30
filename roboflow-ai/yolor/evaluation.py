from glob import glob
import cv2, skimage, os
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time

from tqdm import tqdm

anno_label_path = "/runs/test/exp10/labels"
pred_label_path = "../zero-waste-10/test/labels"

with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
    lines = f.readlines()
    self.focal_length = float(lines[0].strip().split()[-1])
    lines[1] = lines[1].strip().split()
    self.pp = (float(lines[1][1]), float(lines[1][2]))
with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
    self.pose = [line.strip().split() for line in f.readlines()]


