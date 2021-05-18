# Django-Simple-Data-Classifier with Ploty express

## Motivation
A rapid simple method to classify your multidimensional data that stems from your experiments, measurements or studies with only two clicks. The code works with function-based views. Tip: use this program in combination with my related repository **django_machinelearning_clusters**.

## How to use it
### Loading the data sets
You need two sets of data; a "model" and a "test" dataset. The latter is the data to be classified and thus contains no classification yet, only the numeric values. In contrast, the model dataset needs to contain classified data (e.g. results from previous experiments ... ) as it will be used to train the machine learning models. To feed the app with both datasets paste their URLs (being it local or remote) into the corresponding fields and press load Data. 
###### Note: Both datasets need to be in the csv format and in the model data set the head of the column that contains the classification data should be explicitely named "classification".
![loadData](snapshots/loadData.png?raw=true "loadData")
### Classify
After loading the data the heads of the tables are shown, for a final check for irregularities. If everything is ok push 'Start Classification'. 
###### Note: Depending on the size of your datasets the actual classification process might take several minutes. For the relatively small 'iris.csv' that I provided for you to get started, it takes 29 seconds. Any further analysis on the same model data will be faster, as the trained model is stored as an object in the database. 
The results are then displayed in the manner as can be seen in the pic below.
![afterAnalysis](snapshots/afterAnalysis.png?raw=true "afterAnalysis")

### What do the results mean?
The results are split into two columns; PCA (Principal component Analysis) and LDA (Linear Discriminant Analysis). Both, PCA and LDA, project your multidimensional data into two dimensions and thereby enable you to get a visual impression. Data points close to each other are in both cases likely to be the same class. The spatial separation of classes however is better achieved by LDA as its 2D projection is based upon the categories, whereas PCA projection is based upon variance. 

#### "Test" data results and crossvalidation
All the graphs below show the "test" dataset, each obtained by a different training method (Decision tree, Random Forrest, K Next Neighbors, Support Vector Machine, (with and without Kernel trick), Logistic regression and Perceptron). Each of these methods was trained by a 10-fold cross-validation on the "model" data beforehand. The scoring below each plot shows how many right predictions on average during the cross-validation were made. The higher this scoring, the more trust you can put into the predicted classification of your "test" data. 

![retrieveOriginalData](snapshots/retrieveOriginalData.png?raw=true "retrieveOriginalData")
#### To which samples belong the points in the graphs
To uncover which sample in your original test data set is represented by the points in the 2D projection hover over the data points in the graphs. 

#### Get a feeling for the meaning of the results with createTestData.py
If you are not familiar with machine learning algorithms, it is important to get at least a rough idea of how to interpret the results. One way to achieve this is to get "artificial experience" with different test datasets. For that use createTestData.py, which generates artificial test datasets on the base of your model data. Random samples from the model dataset are picked and a random number in the order of magnitude of the original value is added. How big this deviation is can be controlled by the deviation factor (range: 0-1, recommended <0.1). The size of the artificial dataset is tuned by the fraction factor (range: 0-1). Play around with different test datasets. 

## Details of the code in this project and lessons to be learned
### Crucial logics
The code contains only three views. The "welcome" provides the form to load the CSV files. 

The second view "loadData" loads the CSV files and converts them into pd.DataFrames by using the function base data(URL). Those data frames are then passed to the "dataTable" function that converts the frame into parameters (column names) and the corresponding sample. Samples and parameters are then sent to the Datatable templates where the first 5 samples of the datasets are displayed. 

The third view "simpleAccuracy(request)" carries out the cross-validation by the "modelScoringCrossVal" method. This method provides the scoring of the cross-validation and the names of the classification methods that are then displayed inside the "dimred.html". It however also provides the predicted classification of the test dataset, which is then used in "projectIn2D(self,pdFrame, dimred, projection, URL, model='None')" to plot the data. 

###### Note: the URLs of the model data and the test data are also passed from view to view employing hidden input forms such as the following
![hiddenURL](snapshots/hiddenURL.png?raw=true "hiddenURL")


### functions.py 
to avoid overloading of the views.py module I added functions.py that takes care of the machine learning logic. In the following, methods refer to functions in this module. 
### How to get Plotly graphs from the views.py to a template 

The scatterplots are created in the following manner in the "projectIn2D" method.  The hover data has to be a dictionary.
![createPlot](snapshots/createPlot.png?raw=true "createPlot")
Over the views.py the plot is then passed to a template with the following structure. 
![receivePlot](snapshots/receivePlot.png?raw=true "receivePlot")



### models.py
If you simply want to recall the results of your classification that you have already carried out, the app doesn't need to go through the entire process of recalculation again.  For that, any analysis can is stored as an object in the database. There are three types of objects "BaseData" that stores the read data as a Pandas frame, "CrossVal" that stores the names of the methods(DecisionTree, ....), cross-validation scores and the predicted classification and nonetheless "ProjectionIn2D" that stores the plots.  

![objects](snapshots/models.png?raw=true "objects")

The latter two are in a OneToMany-Relationship with the first, by a ForeignKey attribute. At the beginning of any calculation the function will first try to get the results from the database. If the object is not found, data is calculated and stored in a corresponding object. 

![checkAndCalc](snapshots/checkAndCalc.png?raw=true "checkAndCalc")

###### Note: The objects in the database are mainly identified by the URL of the CSV. Any modification in the original CSV file (such as adding or corrected data) will thus only be recognized if the URL is changed as well. 
Pandas Frames and plots are stored as PickleFireldobjects. 

#### deficits of the code
Due to time shortage, some deficits still have to be corrected: 
the colors of the classes might differ from graph to graph. 
the style is written inline and not in a separate CSS file

#### yet to be done
Data preparation is a crucial precondition to classify data. In the current version of the app, data preparation still has to be done beforehand.  
Depending on the model dataset the classification methods might need optimized hyperparameters. An interface to tune these hyperparameters is in work. 
Also, the visualization can be optimized by color-coding the graph regions according to the expected class.
