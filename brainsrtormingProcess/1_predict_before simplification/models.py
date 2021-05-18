from django.db import models
from django.conf import settings
from picklefield.fields import PickledObjectField
import numpy


# class PredResults(models.Model):
#     sepal_length = models.FloatField()
#     sepal_width = models.FloatField()
#     petal_length = models.FloatField()
#     petal_width = models.FloatField()
#     classification = models.CharField(max_length=30)

#     def __str__(self):
#         return self.classification


class BaseData(models.Model):
    url= models.CharField(max_length=30, default='baseData')
    data = PickledObjectField(default='SOME STRING')

class CrossVal(models.Model):
    url= models.CharField(max_length=30, default='CrossVal')
    cv_scores = PickledObjectField()
    cv_names = PickledObjectField()
    dimRed = PickledObjectField()
    baseData= models.ForeignKey(BaseData, on_delete=models.CASCADE, related_name='CrossVal')

class OptimizedHyperParameters(models.Model):
    url= models.CharField(max_length=30, default='OptimizedHyperParameters')
    optimizedScores = PickledObjectField()
    modelNames = PickledObjectField()
    bestSingle = PickledObjectField()
    bestParameters = PickledObjectField(default='OptimizedHyperParameters')
    baseData= models.ForeignKey(BaseData, on_delete=models.CASCADE, related_name='OptimizedHyperParameters')

class MajorityVote(models.Model):
    url= models.CharField(max_length=30, default='MajorityVote')
    mvc_scores = PickledObjectField()
    baseData= models.ForeignKey(BaseData, on_delete=models.CASCADE, related_name='MajorityVote')



