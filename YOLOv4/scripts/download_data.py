from roboflow import Roboflow
rf = Roboflow(api_key="j5T2cCGBbriYBlLyNw5I")
project = rf.workspace("drago1234").project("zero-waste")
dataset = project.version(10).download("coco")