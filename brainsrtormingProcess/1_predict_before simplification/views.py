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

def loadData(request):
    if request.method=='POST':
        # getmodel
        URL_Model=request.POST.get('URL_Model')
        baseDataObj, pdFrame_Model=fu.baseData(URL_Model)
        infoNames_Model,infoList_Model=fu.dataInfo(pdFrame_Model)
        samples_Model, parameters_Model=fu.dataTable(pdFrame_Model)
        #get Test
        URL_Test=request.POST.get('URL_Test')
        pdFrame_Test=sa.baseData(URL_Test)
        infoNames_Test,infoList_Test=sa.dataInfo(pdFrame_Test)
        samples_Test, parameters_Test=sa.dataTable(pdFrame_Test)

        context={"samples_Test": samples_Test, "parameters_Test": parameters_Test, 'URL_Test':URL_Test, 'infoNames_Test':infoNames_Test,'infoList_Test':infoList_Test, "samples_Model": samples_Model, "parameters_Model": parameters_Model, 'URL_Model':URL_Model, 'infoNames_Model':infoNames_Model,'infoList_Model':infoList_Model}
    else:
        context={}
    return render(request, 'start.html', context)


def simpleAccuracy(request):
    if request.method=='POST':
        URL_Model=request.POST.get('URL_Model')
        URL_Test=request.POST.get('URL_Test')
        baseDataObj, pdFrame=fu.baseData(URL_Model)
        baseDataObj_Test, X_data=fu.baseData(URL_Test)
        pcaPlot, pcaData,principalDf =fu.projectIn2D(fu,pdFrame,PCA(n_components=2))
        LDAPlot, LDAData,LDADf=fu.projectIn2D(fu,pdFrame,LDA(n_components=2))
        

        scoresPCACV, modelNames, predicted_PCA= fu.modelScoringCrossVal(fu, URL_Model,pdFrame,PCA(n_components=2),X_data )
        scoresLDACV, modelNames, predicted_LDA= fu.modelScoringCrossVal(fu,URL_Model, pdFrame,LDA(n_components=2) ,X_data)

        optimizedScores, opimizedmodelNames, bestSingle,bestParameters=fu.optimizeHyperParameters(fu, pdFrame,URL_Model)
        PDA_optimizedScores, PDA_opimizedmodelNames, PDA_bestSingle,PDA_bestParameters=fu.optimizeHyperParameters(fu, principalDf,URL_Model)
        LDA_optimizedScores, LDA_opimizedmodelNames, LDA_bestSingle ,LDA_bestParameters =fu.optimizeHyperParameters(fu, LDADf,URL_Model)

        mvc_scores=fu.majorityVote(fu, pdFrame, bestParameters, opimizedmodelNames,URL_Model)
        mvc_score=[round(mvc_scores[i][0]*100,2) for i in range(len(mvc_scores))]
        mvc_dev=[round(mvc_scores[i][1]*100,2) for i in range(len(mvc_scores))]
        mvc_name=[mvc_scores[i][2] for i in range(len(mvc_scores))]

        context={ 'URL_Model':URL_Model, 'pcaPlot':pcaPlot,'LDAPlot':LDAPlot,'modelNames': modelNames, 'scoresPCACV':scoresPCACV, 'scoresLDACV':scoresLDACV, 'optimizedScores': optimizedScores, 'opimizedmodelNames': opimizedmodelNames,'PDA_optimizedScores':PDA_optimizedScores, 'PDA_opimizedmodelNames':PDA_opimizedmodelNames, 'PDA_bestSingle':PDA_bestSingle,'LDA_optimizedScores':LDA_optimizedScores, 'LDA_opimizedmodelNames':LDA_opimizedmodelNames, 'LDA_bestSingle':LDA_bestSingle, 'mvc_score': mvc_score, 'mvc_dev': mvc_dev, 'mvc_name': mvc_name}   
    else:
        context={}
    return render(request, 'dimRed.html', context)




