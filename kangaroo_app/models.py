from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to="uploads/")
    predicted_boxes = models.JSONField(default=list)  # Store YOLO's bounding box results
    corrected_boxes = models.JSONField(default=list, blank=True, null=True)  # Manual corrections
    processed = models.BooleanField(default=False)  # Indicates if retrained

    def __str__(self):
        return f"Image {self.id}"

