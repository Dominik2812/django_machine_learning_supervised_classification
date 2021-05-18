from django.urls import path
from . import views

app_name = 'predict'

urlpatterns = [
    path('', views.loadData, name='loadData'),

    path('simpleAccuracy', views.simpleAccuracy, name='simpleAccuracy'),


]