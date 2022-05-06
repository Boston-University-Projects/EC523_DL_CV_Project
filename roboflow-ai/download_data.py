# !pip install roboflow --user


# =======================Version 1: No Augmentation, no tiling, resized to 416x416 ==============
# from roboflow import Roboflow
# rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
# project = rf.workspace("drago1234").project("zero-waste")
# dataset = project.version(1).download("yolov5")

# from roboflow import Roboflow
# rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
# project = rf.workspace("drago1234").project("zero-waste")
# dataset = project.version(1).download("coco")


# =======================Version 15: No Augmentation, resized to 448x448==============
# from roboflow import Roboflow
# rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
# project = rf.workspace("drago1234").project("zero-waste")
# dataset = project.version(15).download("yolov5")

# from roboflow import Roboflow
# rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
# project = rf.workspace("drago1234").project("zero-waste")
# dataset = project.version(15).download("coco")

# =======================Version 10: No Augmentation, no tiling, resized to 640x640 ==============
from roboflow import Roboflow
rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
project = rf.workspace("drago1234").project("zero-waste")
dataset = project.version(10).download("yolov5")

# You need coco formated ground truth label for evaluation purpose
from roboflow import Roboflow
rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
project = rf.workspace("drago1234").project("zero-waste")
dataset = project.version(10).download("coco")


# =======================Version 11: Augmented no tilling, x1 ==============
# from roboflow import Roboflow
# rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
# project = rf.workspace("drago1234").project("zero-waste")
# dataset = project.version(11).download("yolov5")

# # You need coco formated ground truth label for evaluation purpose
# from roboflow import Roboflow
# rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
# project = rf.workspace("drago1234").project("zero-waste")
# dataset = project.version(11).download("coco")


# =======================Version 12: Augmented no tilling, x3 ==============

# from roboflow import Roboflow
# rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
# project = rf.workspace("drago1234").project("zero-waste")
# dataset = project.version(12).download("coco")

# from roboflow import Roboflow
# rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
# project = rf.workspace("drago1234").project("zero-waste")
# dataset = project.version(12).download("yolov5")


# =======================Version 13: No Augmentation, resized to 1088x1088==============
# from roboflow import Roboflow
# rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
# project = rf.workspace("drago1234").project("zero-waste")
# dataset = project.version(13).download("yolov5")

# from roboflow import Roboflow
# rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
# project = rf.workspace("drago1234").project("zero-waste")
# dataset = project.version(13).download("coco")


# =======================Version 14: No Augmentation, resized to 1280x1280==============
# from roboflow import Roboflow
# rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
# project = rf.workspace("drago1234").project("zero-waste")
# dataset = project.version(14).download("yolov5")

# from roboflow import Roboflow
# rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
# project = rf.workspace("drago1234").project("zero-waste")
# dataset = project.version(14).download("coco")

# =======================Version 16: No Augmentation, with tiling, resize to 640x640 ==============
# from roboflow import Roboflow
# rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
# project = rf.workspace("drago1234").project("zero-waste")
# dataset = project.version(16).download("yolov5")

# from roboflow import Roboflow
# rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
# project = rf.workspace("drago1234").project("zero-waste")
# dataset = project.version(16).download("coco")



