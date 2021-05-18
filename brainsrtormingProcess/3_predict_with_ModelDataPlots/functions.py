import pandas as pd
import numpy as np
from .models import BaseData, CrossVal, ProjectionIn2D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import plotly.express as px
from plotly.offline import plot
from sklearn.pipeline import make_pipeline

class Funcs():
    # loads csv file, converts it to a BaseData Object a
    def baseData(URL):
        baseData = pd.read_csv(URL) 
        try:
            baseDataObj=BaseData.objects.get(url=URL)
        except:
            baseDataObj=BaseData()
            baseDataObj.data=baseData
            baseDataObj.url=URL
            baseDataObj.save()
        pdFrame=baseDataObj.data
        return pdFrame
    # show Table on star 
    def dataTable(pdFrame):
        parameters=[]
        for col in pdFrame.columns:
            parameters.append(col)
        pdFrameDict=pdFrame.to_dict()

        samples = []
        for i in range(len(pdFrame.head())):
            sample=[i]
            for key in pdFrameDict:
                sample.append(pdFrameDict[key][i])
            samples.append(sample)
        return samples,  parameters

    def projectIn2D(self,pdFrame, dimred,projection, URL, model='None'):
        X,y = self.split(pdFrame)
        X_std=StandardScaler().fit_transform(X)
        try:
            PI2D=ProjectionIn2D.objects.get(url=URL, dimRed=projection, model=model)
            Plot=PI2D.plot
        except:
            # LDA, only works if there are at least 3 classes
            try:
                components = dimred.fit_transform(X_std, y)
                df = pd.DataFrame(data = components, columns = ['C1', 'C2'])
                df = pd.concat([df, pdFrame[['classification']]], axis = 1)
                data=[[df['C1'][i],df['C2'][i],df['classification'][i]] for i in range(len(df))]

                # Plotting
                df_plot = pd.DataFrame(data, columns =['C1','C2','Classi']) 
                hover=dict()
                for col in pdFrame.columns:
                    hover[col]=pdFrame[col]
                fig = px.scatter(df_plot,  x="C1", y="C2", color="Classi",hover_data=hover) 
                fig.update_layout(showlegend=False)
                Plot = plot(fig, output_type='div')

            except:
                Plot, data,df  = [],[],pd.DataFrame()
            table=BaseData.objects.get(url=URL)
            PI2D=ProjectionIn2D()
            PI2D.url, PI2D.plot,  PI2D.model,PI2D.dimRed ,PI2D.baseData = URL,Plot,model,projection, table
            PI2D.save()
        return Plot

    def split(pdFrame):
        features =[col for col in pdFrame.columns]
        features.remove('classification')
        X = pdFrame.loc[:, features].values
        y = pdFrame['classification']

        return X,y

    def makeCrosval(self,URL, pdFrame,model,dimred,X_data):

        X,y = self.split(pdFrame)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=1)

        pipe = make_pipeline(StandardScaler(),dimred, model)
        pipe.fit(X_train, y_train)
        # X_data_std=StandardScaler().fit_transform(X_data)

        y_pred_data=pipe.predict(X_data)
        scores = cross_val_score(estimator=pipe,X=X_train,y=y_train,cv=2,n_jobs=1)
        meanScore = (round(np.mean(scores*100),2), round(np.std(scores*100),0))

        return y_pred_data,meanScore


    def modelScoringCrossVal(self, URL,pdFrame, dimred,projection, X_data):

        features =[col for col in pdFrame.columns]
        features.remove('classification')
        X_data= X_data.loc[:, features]

        try:
            CV=CrossVal.objects.get(url=URL, dimRed=projection)
            scores,modelNames, pred_frames=CV.cv_scores, CV.cv_names,CV.pred_frames
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
            CV.cv_names, CV.url,  CV.dimRed,CV.baseData= modelNames, URL, projection, table
            

            scores=list()
            pred_frames=list()
            for i,model in enumerate(models):
                y_pred_data, meanScore=self.makeCrosval(self,URL, pdFrame,model,dimred,X_data)
                scores.append(meanScore)
                class_frame=pd.DataFrame(y_pred_data, columns=['classification'])
                # X_data_frame=
                predicted_Frame=pd.concat([X_data,class_frame], axis=1)
                pred_frames.append(predicted_Frame)

            CV.cv_scores=scores
            CV.pred_frames=pred_frames
            CV.save()

        return scores,modelNames, pred_frames

#