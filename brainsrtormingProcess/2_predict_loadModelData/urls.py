from django.urls import path
from . import views

app_name = 'predict'

urlpatterns = [
    path('', views.start, name='start'),
    path('changeCols', views.changeCols, name='changeCols'),
    path('simpleAccuracy', views.simpleAccuracy, name='simpleAccuracy'),
    # path('loadNewData', views.loadNewData, name='loadNewData'),

]