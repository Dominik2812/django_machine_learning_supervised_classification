from django.shortcuts import render
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from .functions import Funcs as fu



def welcome(request):
    return render(request, 'welcome.html')

def loadData(request):
    if request.method=='POST':
        # get and show modelData
        URL_Model=request.POST.get('URL_Model')
        pdFrame_Model=fu.baseData(URL_Model)
        samples_Model, parameters_Model=fu.dataTable(pdFrame_Model)
        #get and show TestData
        URL_Test=request.POST.get('URL_Test')
        pdFrame_Test=fu.baseData(URL_Test)
        samples_Test, parameters_Test=fu.dataTable(pdFrame_Test)

        context={"samples_Test": samples_Test, "parameters_Test": parameters_Test, 'URL_Test':URL_Test, "samples_Model": samples_Model, "parameters_Model": parameters_Model, 'URL_Model':URL_Model}
    else:
        context={}
    return render(request, 'start.html', context)


def simpleAccuracy(request):
    if request.method=='POST':
        #load the two datasets
        URL_Model=request.POST.get('URL_Model')
        URL_Test=request.POST.get('URL_Test')
        pdFrame=fu.baseData(URL_Model)
        X_data=fu.baseData(URL_Test)

        #make get the scores of all models and the predicted classification of the testdata
        scoresPCACV, modelNames, pred_framesPCA= fu.modelScoringCrossVal(fu, URL_Model,pdFrame,PCA(n_components=2),"PCA", X_data )
        scoresLDACV, modelNames,  pred_framesLDA= fu.modelScoringCrossVal(fu,URL_Model, pdFrame,LDA(n_components=2) ,"LDA",X_data)

        #create 2D plots of the predicted test data classification for all models
        LDA_PLots=[]
        PCA_PLots=[]
        for i, _ in enumerate(modelNames):

            LDA_Plot=fu.projectIn2D(fu,pred_framesLDA[i],LDA(n_components=2),"LDA",URL_Test, modelNames[i])
            LDA_PLots.append(LDA_Plot)
            PCA_Plot=fu.projectIn2D(fu,pred_framesPCA[i],PCA(n_components=2),"PCA",URL_Test,modelNames[i])
            PCA_PLots.append(PCA_Plot)

        PCA_zipped=zip(PCA_PLots, modelNames,scoresPCACV )
        LDA_zipped=zip(LDA_PLots, modelNames,scoresLDACV )
        context={ 'URL_Model':URL_Model,'modelNames': modelNames,'LDA_zipped': LDA_zipped,'PCA_zipped': PCA_zipped}

    else:
        context={}
    return render(request, 'dimRed.html', context)




