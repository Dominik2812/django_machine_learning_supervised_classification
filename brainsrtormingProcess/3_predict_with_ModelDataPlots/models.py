from django.db import models
from django.conf import settings
from picklefield.fields import PickledObjectField
import numpy


class BaseData(models.Model):
    url= models.CharField(max_length=30, default='baseData')
    data = PickledObjectField(default='SOME STRING')

class CrossVal(models.Model):
    url= models.CharField(max_length=30, default='CrossVal')
    cv_scores = PickledObjectField()
    cv_names = PickledObjectField()
    dimRed = models.CharField(max_length=30, default='LDA')
    pred_frames=PickledObjectField(default='pred_frames')
    baseData= models.ForeignKey(BaseData, on_delete=models.CASCADE, related_name='CrossVal')

class ProjectionIn2D(models.Model):
    url= models.CharField(max_length=30, default='URL')
    plot = PickledObjectField()
    model = models.CharField(max_length=30, default='Modelname')
    dimRed = models.CharField(max_length=30, default='LDA')
    baseData= models.ForeignKey(BaseData, on_delete=models.CASCADE, related_name='ProjectionIn2D')





