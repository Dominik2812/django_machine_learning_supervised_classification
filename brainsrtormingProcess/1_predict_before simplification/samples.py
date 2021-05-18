import pandas as pd
import numpy as np

from .models import BaseData, CrossVal,OptimizedHyperParameters,MajorityVote

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import plotly.express as px
from plotly.offline import plot
from plotly.graph_objs import Scatter
import plotly.graph_objs as go

import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline

class Samples():

    def baseData(URL):
        baseData = pd.read_csv(URL) 
        pdFrame=baseData
        return pdFrame
    
    def dataTable(pdFrame):
        parameters=[]
        for col in pdFrame.columns:
            parameters.append(col)
        pdFrameDict=pdFrame.to_dict()

        samples = []
        for i in range(len(pdFrame.head())):
            sample=[i]
            for key in pdFrameDict:
                print(key)
            for key in pdFrameDict:

                sample.append(pdFrameDict[key][i])
            samples.append(sample)
        return samples, parameters
        
    def dataInfo(pdFrame):
        infoNames=['columns','Collumns that contain missing values','dimensions','samples/Column','Statistics']
        infoList=[pdFrame.columns.values,pdFrame.isnull().sum(),pdFrame.shape,pdFrame.count().head,pdFrame.describe()]
        return infoNames,infoList

    def adaptParameters(self,postrequest,URL):
        new_parameters=[]
        for element in postrequest:
            if postrequest[element]!= URL and element != 'csrfmiddlewaretoken':
                new_parameters.append(postrequest[element])
        baseDataObj,pdFrame=self.baseData(URL)
        pdFrame.columns= new_parameters
        baseDataObj.save()
        pdFrame=baseDataObj.data

        return pdFrame


    def makePCA(self,pdFrame):
        X,y,Xstd = self.split_standardize(pdFrame)
        # PCA
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(Xstd)
        principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
        y = pd.DataFrame(data=y, columns=['classification'])
        
        principalDf = pd.concat([principalDf, pdFrame[['classification']]], axis = 1)
        pcaData=[[principalDf['principal component 1'][i],principalDf['principal component 2'][i],principalDf['classification'][i]] for i in range(len(principalDf))]
        pcaDataFrame = pd.DataFrame(pcaData, columns =['C1','C2','Classi']) 

        # Potting
        fig = px.scatter(pcaDataFrame,  x="C1", y="C2", color="Classi") 
        pcaPlot = plot(fig, output_type='div')
        
        return pcaPlot, pcaData, principalDf   

    def makeLDA(self,pdFrame):
        X,y,Xstd = self.split_standardize(pdFrame)
        
        # LDA, only works if there aare at least 3 classes
        try:
            lda = LDA(n_components=2)
            LDAcomponents = lda.fit_transform(Xstd, y)
            LDADf = pd.DataFrame(data = LDAcomponents, columns = ['LDA component 1', 'LDA component 2'])
            
            LDADf = pd.concat([LDADf, pdFrame[['classification']]], axis = 1)
            LDAData=[[LDADf['LDA component 1'][i],LDADf['LDA component 2'][i],LDADf['classification'][i]] for i in range(len(LDADf))]
            LDADataFrame = pd.DataFrame(LDAData, columns =['C1','C2','Classi']) 

            # Potting
            fig = px.scatter(LDADataFrame,  x="C1", y="C2", color="Classi") 
            LDAPlot = plot(fig, output_type='div')
        except:
            LDAPlot, LDAData ,LDADf  = [],[],pd.DataFrame()
        return LDAPlot, LDAData ,LDADf       


    def split_standardize(pdFrame):
        features =[col for col in pdFrame.columns]
        features.remove('classification')
        # Standardize Data
        X = pdFrame.loc[:, features].values
        y = pdFrame['classification']

        Xstd= StandardScaler().fit_transform(X)
        return X,y,Xstd



    def makeCrosval(self,URL, pdFrame,model,dimred):
        X,y,Xstd = self.split_standardize(pdFrame)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=1)
        pipe = make_pipeline(StandardScaler(),dimred, model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        scores = cross_val_score(estimator=pipe,X=X_train,y=y_train,cv=10,n_jobs=1)
        singleScore, crossValScore, meanScore = pipe.score(X_test, y_test),scores,(round(np.mean(scores*100),2), round(np.std(scores*100),0))
        return y_pred,singleScore, crossValScore, meanScore


    def modelScoringCrossVal(self, URL,pdFrame, dimred):
        try:
            CV=CrossVal.objects.get(url=URL, dimRed=dimred)
            scores,modelNames=CV.cv_scores, CV.cv_names
        except:
            tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4,random_state=1)
            forest = RandomForestClassifier(criterion='entropy',max_depth=4, n_estimators=100, random_state=1,n_jobs=1)
            knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
            svm = SVC(kernel='linear', C=1.0, random_state=1)
            svmKernel = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0) 
            lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
            ppn = Perceptron(eta0=0.1, random_state=1) 

            models=[tree_model,forest,knn,svm,svmKernel,lr,ppn]
            modelNames=['Decision Tree', 'RandomForrest',  'KNN', 'SVM', 'SVM-Kernel','LogisticRegression','Perceptron']

            table=BaseData.objects.get(url=URL)
            CV=CrossVal()
            CV.cv_names, CV.url,  CV.dimRed,CV.baseData= modelNames, URL, dimred, table
            

            scores=list()
            for i,model in enumerate(models):
                y_pred,singleScore, crossValScore, meanScore=self.makeCrosval(self,URL, pdFrame,model,dimred)
                scores.append(meanScore)
            CV.cv_scores=scores
            CV.save()

        return scores,modelNames

    def optimizeHyperParameters(self, pdFrame,URL):
        try:
            OHP=OptimizedHyperParameters.objects.get(url=URL)
            optimizedScores,modelNames, bestSingle=OHP.optimizedScores,OHP.modelNames, OHP.bestSingle
        except:
            X,y,Xstd = self.split_standardize(pdFrame)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=1)

            #####################################

            knn = KNeighborsClassifier( p=2)
            # pipe_knn = knn #make_pipeline(StandardScaler(), knn)
            param_range_knn_neigh = [2,5,10]
            param_grid_knn_neigh = {'n_neighbors': param_range_knn_neigh, 'metric':['euclidean','manhattan','minkowski']} #'chebyshev',
            #####################################

            tree=DecisionTreeClassifier(random_state=0)
            # pipe_tree = tree
            param_grid_tree_depth = {'max_depth': [1, 2, 3, 4, None]}
            #####################################

            lr = LogisticRegression(multi_class='ovr', random_state=1)
            param_range_log_C = [ 0.001,  0.1, 1.0, 10.0,  1000.0]
            param_grid_log_C = {'C': param_range_log_C, 
                        'solver': ['lbfgs']}
            #####################################

            forest = RandomForestClassifier(criterion='entropy', 
            random_state=1,
            n_jobs=1)
            pipe_forrest=forest

            param_grid_Forest = {'max_depth': [3, 5, 7, 10, None], 'n_estimators': [10,100]}
            #####################################

            svm = SVC(random_state=1)
            pipe_svc = make_pipeline(StandardScaler(),svm)
            param_range_svm_C = [ 0.001, 0.1, 1.0, 10.0,  1000.0]
            param_grid_SVC = [{'svc__C': param_range_svm_C, 'svc__kernel': ['linear']},{'svc__C': param_range_svm_C, 'svc__gamma': param_range_svm_C, 'svc__kernel': ['rbf']}]

            models=[tree,pipe_svc,knn,lr,forest]
            clafs=[tree,pipe_svc,knn,lr,forest]

            param_grid=[param_grid_tree_depth,param_grid_SVC, param_grid_knn_neigh, param_grid_log_C,param_grid_Forest]
            modelNames= [ 'Tree', 'SVM','KNN', 'LogRg','RandomForrest']

            optimizedScores=list()
            bestSingle=dict()
            for i, model in enumerate(models):
                gs = GridSearchCV(estimator=model,param_grid=param_grid[i],scoring='accuracy',refit=True,cv=2)
                scores = cross_val_score(gs, X_train, y_train,scoring='accuracy', cv=5)
                try:
                    bestSingelScore=gs.fit(X_train, y_train).best_score_
                    bestParameters=gs.fit(X_train, y_train).best_params_
                except:
                    print('has none')
                meanScore=(round(np.mean(scores)*100,2), round(np.std(scores)*100,0))
                optimizedScores.append(meanScore)
                bestSingle[modelNames[i]]=[bestParameters,bestSingelScore]
            table=BaseData.objects.get(url=URL)
            OHP=OptimizedHyperParameters()
            OHP.optimizedScores,OHP.modelNames, OHP.bestSingle=optimizedScores,modelNames, bestSingle
            OHP.baseData=table
            OHP.url=URL
            OHP.save()
            
        return optimizedScores,modelNames, bestSingle

    def majorityVote(self, pdFrame, bestSingle, modelNames,URL):
        try:
            MV=MajorityVote.objects.get(url=URL)
            mvc_scores=MV.mvc_scores
        except:
            X,y,Xstd = self.split_standardize(pdFrame)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=1)
            #####################################
            #################################
            max_depth_DT=bestSingle['Tree'][0]['max_depth']
            C_svm,kernel_svm=bestSingle['SVM'][0]['svc__C'],bestSingle['SVM'][0]['svc__kernel']
            metric_knn, n_neighbors_knn=bestSingle['KNN'][0]['metric'], bestSingle['KNN'][0]['n_neighbors']
            C_lr, solver_lr=bestSingle['LogRg'][0]['C'], bestSingle['LogRg'][0]['solver']
            max_depth_RF, n_estimators_RF=bestSingle['RandomForrest'][0]['max_depth'], bestSingle['RandomForrest'][0]['n_estimators']
            

            #####################################
            #################################
            tree=DecisionTreeClassifier(random_state=0, max_depth=max_depth_DT)
            svm = SVC(C=C_svm,kernel=kernel_svm, random_state=1)
            knn = KNeighborsClassifier( p=2, metric=metric_knn, n_neighbors=n_neighbors_knn)
            lr = LogisticRegression(multi_class='ovr', random_state=1,C=C_lr, solver=solver_lr)
            forest = RandomForestClassifier(criterion='entropy', max_depth=max_depth_RF,n_estimators=n_estimators_RF, random_state=1, n_jobs=1)

            #####################################
            #################################
            pipe_tree = tree
            pipe_svm = Pipeline([['sc', StandardScaler()],['clf', svm]])
            pipe_knn = Pipeline([['sc', StandardScaler()],['clf', knn]])
            pipe_lr = Pipeline([['sc', StandardScaler()],['clf', lr]])
            pipe_forest = forest


            mv_clf = MajorityVoteClassifier(classifiers=[pipe_tree, pipe_svm , pipe_knn, pipe_lr,pipe_forest])

            modelNames.append('Majority voting')
            all_clf = [pipe_tree, pipe_svm , pipe_knn, pipe_lr,pipe_forest, mv_clf]
            mvc_scores=list()
            for clf, label in zip(all_clf, modelNames):
                scores = cross_val_score(estimator=clf, X=X_train, y=y_train,cv=10,scoring='accuracy')
                mvc_scores.append((scores.mean(), scores.std(), label))
            table=BaseData.objects.get(url=URL)
            MV=MajorityVote()
            MV.mvc_scores = mvc_scores
            MV.baseData=table
            MV.url=URL
            MV.save()
        return mvc_scores
            
        





from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator


class MajorityVoteClassifier(BaseEstimator, 
                             ClassifierMixin):
    """ A majority vote ensemble classifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
      Different classifiers for the ensemble

    vote : str, {'classlabel', 'probability'} (default='classlabel')
      If 'classlabel' the prediction is based on the argmax of
        class labels. Else if 'probability', the argmax of
        the sum of probabilities is used to predict the class label
        (recommended for calibrated classifiers).

    weights : array-like, shape = [n_classifiers], optional (default=None)
      If a list of `int` or `float` values are provided, the classifiers
      are weighted by importance; Uses uniform weights if `weights=None`.

    """
    def __init__(self, classifiers, vote='classlabel', weights=None):

        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            Matrix of training examples.

        y : array-like, shape = [n_examples]
            Vector of target class labels.

        Returns
        -------
        self : object

        """
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'"
                             "; got (vote=%r)"
                             % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d classifiers'
                             % (len(self.weights), len(self.classifiers)))

        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            Matrix of training examples.

        Returns
        ----------
        maj_vote : array-like, shape = [n_examples]
            Predicted class labels.
            
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' vote

            #  Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_features is the number of features.

        Returns
        ----------
        avg_proba : array-like, shape = [n_examples, n_classes]
            Weighted average probability for each class per example.

        """
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
            return out