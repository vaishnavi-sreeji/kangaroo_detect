import os
import cv2
import numpy as np
import requests
from django.shortcuts import render,redirect
from django.core.files.storage import default_storage
from django.conf import settings
from django.http import JsonResponse
from ultralytics import YOLO
from roboflow import Roboflow
import json
import matplotlib.pyplot as plt
from django.http import HttpResponse


model = YOLO(os.path.join(settings.BASE_DIR,'kangaroo_app', 'best.pt'))

rf = Roboflow(api_key="bj6ZqTUTkbLUEG1wNz1C")
project = rf.workspace("trash-jcnv1").project("kangaroo_detection-9ektj")
dataset = project.version(2)

# COLAB_NOTEBOOK_URL = "https://colab.research.google.com/drive/1ThYvuBHf8xbD_vMdbdGj5f5YJtwks8xv"
COLAB_NOTEBOOK_URL ="https://colab.research.google.com/drive/12KUdvygmwRTrRj9cJ4ZYSpWOpwbwQMtv"


def detect_kangaroos(image_path):
    results = model(image_path)  
    image = cv2.imread(image_path)

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):  
            x1, y1, x2, y2 = map(int, box.tolist())
            label = model.names[int(cls)] 
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = os.path.join(settings.MEDIA_ROOT, 'detected_kangaroo.jpg')
    cv2.imwrite(output_path, image)
    return output_path


def upload_predict_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        file_path = default_storage.save(image_file.name, image_file)
        image_url = settings.MEDIA_URL + file_path  

        detected_image = detect_kangaroos(os.path.join(settings.MEDIA_ROOT, file_path))

        return render(request, 'result.html', {'image_path': detected_image.replace(settings.MEDIA_ROOT, settings.MEDIA_URL)})

    return render(request, 'upload.html')

def upload_image_for_annotation(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        file_path = os.path.join(settings.MEDIA_ROOT, image_file.name)

        with open(file_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        try:
            print(f"Uploading {file_path} to Roboflow...")
            upload_response = project.upload_image(file_path, num_retry=3)
            print("Raw Upload Response:", upload_response)

            if isinstance(upload_response, tuple):
                response_dict = upload_response[0]  
            else:
                response_dict = upload_response  

            if isinstance(response_dict, dict) and ('id' in response_dict or response_dict.get('duplicate', False)):
                image_url = settings.MEDIA_URL + image_file.name
                return render(request, 'upload_success.html', {
                    'image_path': image_url,
                    'message': "Image uploaded successfully to Roboflow."
                })

            return render(request, 'upload_failed.html', {
                'error': f'Failed to upload image: {upload_response}'
            })

        except Exception as e:
            print("Roboflow Upload Error:", str(e))
            return render(request, 'upload_failed.html', {
                'error': f'Roboflow Upload Error: {str(e)}'
            })

    return render(request, 'upload_annimage.html')


MAX_ITERATIONS = 3 

def retrain_model(request):
    iteration = request.session.get('retrain_iteration', 0)
    if iteration < MAX_ITERATIONS:
        
        request.session['retrain_iteration'] = iteration + 1
        print(f"Starting retraining iteration {iteration + 1}...")

        return redirect(COLAB_NOTEBOOK_URL)

    else:
        
        request.session['retrain_iteration'] = 0
        print("Retraining complete after 3 iterations.")
        return redirect('/')  

def plot_model_performance(request):
    
    metrics_path = os.path.join(settings.MEDIA_ROOT, "model_metrics.json")

    if not os.path.exists(metrics_path):
        return HttpResponse("Metrics file not found.", status=404)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    
    categories = ["mAP 50", "Precision", "Recall"]
    values = [metrics["mAP_50"], metrics["precision"], metrics["recall"]]

   
    plt.figure(figsize=(6, 4))
    plt.bar(categories, values, color=['blue', 'green', 'red'])
    plt.ylim(0, 1) 
    plt.title("YOLO Model Performance")
    plt.ylabel("Score")

   
    response = HttpResponse(content_type="image/png")
    plt.savefig(response, format="png")
    plt.close()
    return response














