# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:17:04 2024

@author: Ajitesh.Srivastava
"""

from ultralytics import YOLO
import os
from roboflow import Roboflow

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Stops the OpemMp Duplicate error from ocurring


rf = Roboflow(api_key="A4OjNJfPp9QssIW9164L")
project = rf.workspace("personal-workspace-ifdhz").project("truck-object-detection")
version = project.version(1)
dataset = version.download("yolov8")


model = YOLO("yolov8n.pt")
# Train for the first time
# model.train(data = dataset.location + '/data.yaml', epochs = 100, imgsz = 640, plots = True)

# Train with previous Model Weights Specified
# model.train(model = r"C:\Users\Ajitesh.Srivastava\Downloads\Crack_Detection_AI\runs\detect\train2\weights\best.pt", data = dataset.location + '/data.yaml', epochs = 100, imgsz = 640, plots = True)

# Deploy Model to Roboflow
version = project.version(1)
version.deploy("yolov8", r"J:\temp\Ajitesh\04_Ultralytics_AI\04_Truck_Yolov8_Testing\runs\detect\train3", r"J:\temp\Ajitesh\04_Ultralytics_AI\04_Truck_Yolov8_Testing\runs\detect\train3\weights\best.pt")
