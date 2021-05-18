import os
from django.shortcuts import render
from django.http import JsonResponse
from django.urls import reverse, reverse_lazy
from django.conf import settings
from django.http import HttpResponseRedirect

import pandas as pd
# from .models import BaseData, CrossValScores, CrossValModelNames

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import plotly.express as px
from plotly.offline import plot
from plotly.graph_objs import Scatter
import plotly.graph_objs as go

import matplotlib.pyplot as plt


from .functions import Funcs as fu
from .functions import MajorityVoteClassifier as MVC
from .samples import Samples as sa


def welcome(request):
    return render(request, 'welcome.html')

def start(request):
    if request.method=='POST':
        URL=request.POST.get('url')
        baseDataObj, pdFrame=fu.baseData(URL)
        infoNames,infoList=fu.dataInfo(pdFrame)
        samples, parameters=fu.dataTable(pdFrame)
        context={ "samples": samples, "parameters": parameters, 'url':URL, 'infoNames':infoNames,'infoList':infoList}
    else:
        context={}
    return render(request, 'start.html', context)



def changeCols(request):
    if request.method=='POST':
        URL=request.POST.get('url')
        pdFrame=fu.adaptParameters(fu,request.POST,URL)
        samples, parameters=fu.dataTable(pdFrame)
        fig = px.scatter_matrix(pdFrame, dimensions=pdFrame.columns.drop(['classification']), color="classification",  width=800, height=800)
        plt_div = plot(fig, output_type='div')
        # fu.makeCrosval(fu,pdFrame)
        context= {'plot_div': plt_div, "samples": samples,  "parameters": parameters, 'url':URL}
    else:
        context={}
    return render(request, 'start.html', context)



def simpleAccuracy(request):
    if request.method=='POST':
        URL=request.POST.get('url')

        baseDataObj, pdFrame=fu.baseData(URL)
        # pcaPlot, pcaData,principalDf =fu.makePCA(fu,pdFrame)
        # LDAPlot, LDAData,LDADf=fu.makeLDA(fu,pdFrame)

        scoresPCACV, modelNames= fu.modelScoringCrossVal(fu, URL,pdFrame,PCA(n_components=2) )
        scoresLDACV, modelNames= fu.modelScoringCrossVal(fu,URL, pdFrame,LDA(n_components=2) )

        optimizedScores, opimizedmodelNames, bestSingle=fu.optimizeHyperParameters(fu, pdFrame,URL)

        mvc_scores=fu.majorityVote(fu, pdFrame, bestSingle, opimizedmodelNames,URL)
        mvc_score=[round(mvc_scores[i][0]*100,2) for i in range(len(mvc_scores))]
        mvc_dev=[round(mvc_scores[i][1]*100,2) for i in range(len(mvc_scores))]
        mvc_name=[mvc_scores[i][2] for i in range(len(mvc_scores))]

        context={ 'url':URL, 'modelNames': modelNames, 'scoresPCACV':scoresPCACV, 'scoresLDACV':scoresLDACV, 'optimizedScores': optimizedScores, 'opimizedmodelNames': opimizedmodelNames, 'mvc_score': mvc_score, 'mvc_dev': mvc_dev, 'mvc_name': mvc_name}#   'pcaPlot':pcaPlot,'LDAPlot':LDAPlot,
    else:
        context={}
    return render(request, 'dimRed.html', context)




