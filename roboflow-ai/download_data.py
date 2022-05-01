

# =======================Version 11: Augmented no tilling, x1 ==============
from roboflow import Roboflow
rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
project = rf.workspace("drago1234").project("zero-waste")
dataset = project.version(11).download("yolov5")

from roboflow import Roboflow
rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
project = rf.workspace("drago1234").project("zero-waste")
dataset = project.version(11).download("coco")


# =======================Version 11: Augmented no tilling, x3 ==============

from roboflow import Roboflow
rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
project = rf.workspace("drago1234").project("zero-waste")
dataset = project.version(12).download("coco")

from roboflow import Roboflow
rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
project = rf.workspace("drago1234").project("zero-waste")
dataset = project.version(12).download("yolov5")