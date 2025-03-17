# Kangaroo Detection & YOLO Model Retraining
This project is a Django-based web application that:

Uploads images and detects kangaroos using a YOLOv8 model.

Allows users to manually annotate missing kangaroos and send the images for retraining.

Iteratively(here i=3) retrains the model using Google Colab and updates the YOLO model for better performance.

Visualizes the improvement in performance over multiple retraining iterations.
# Features

Upload images for kangaroo detection.

Automatically detect kangaroos using a pre-trained YOLOv8 model.

Manually add bounding boxes for missed detections using Roboflow.

Retrain the YOLO model in Google Colab and update weights in Django.

Performance Visualization: Overlay past & current performance metrics on a graph.
#Usage Guide

🔹 Predict an Image

Navigate to http://127.0.0.1:8000/predict.

Click on the Upload Image button.

Select an image and upload it.

The application will detect kangaroos using the YOLOv8 model.

🔹 Manual Annotation (If Needed)

Navigate to http://127.0.0.1:8000/upload

If some kangaroos are not detected, manually draw bounding boxes.

Submit the corrected image for retraining.

🔹 Retrain Model
Navigate to http://127.0.0.1:8000/retrain

Click Retrain Model.

The model is retrained in Google Colab (automatically redirected).

The new weights are stored and downloaded into Django.

The model performance is visualized in a graph.

Navigate to http://127.0.0.1:8000/plot
