# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:37:36 2024

@author: Ajitesh.Srivastava
"""


from inference import get_model
import supervision as sv
import cv2

import os
from os import path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Stops the OpemMp Duplicate error from ocurring


# # define the image url to use for inference
# image_file = r"C:\Users\Ajitesh.Srivastava\Downloads\Truck_model_test.jfif"


def annotate(image_file):
    image = cv2.imread(image_file)
    
    # load a pre-trained yolov8n model
    model = get_model(model_id="truck-object-detection/1", api_key="A4OjNJfPp9QssIW9164L")
    
    # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
    results = model.infer(image)[0]
    
    # load the results into the supervision Detections api
    detections = sv.Detections.from_inference(results)
    
    # create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale = 1, text_thickness=2)
    
    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    
    # display the image
    sv.plot_image(annotated_image)


folder_path = r"C:\Users\Ajitesh.Srivastava\Downloads\Personal Trucks Project\Inference_Test\01_Source_Images"

for image in os.listdir(os.fsencode(folder_path)):
    file_path = path.join(folder_path, os.fsdecode(image))
    annotate(file_path)
