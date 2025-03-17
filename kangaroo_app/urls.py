from django.urls import path
from .views import upload_predict_image 
from .views import upload_image_for_annotation
from .views import retrain_model
from .views import plot_model_performance


urlpatterns = [
    path('predict/', upload_predict_image, name='upload_image'),
    path('upload/', upload_image_for_annotation, name='upload_image'),
    path('retrain/', retrain_model, name='retrain_model'),
    path("plot/", plot_model_performance, name="plot_model_performance"),
]
