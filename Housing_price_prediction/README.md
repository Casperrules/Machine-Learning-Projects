# Predicting house prices using Regression based model 
### Description:
Uploads consist of python codes that are used to load(pandas) datasets fro training and testing the trained model. The prediction is done using Supervised learning.
We load the data
Clean the data, to be able to use it to train the model
Choose an appropriate classifier(in this case sklearn.linear_model.Ridge)
Fit the data to the model. i.e. train the model with the training data
Run the classifier with the new testing data , to generate predictions.

Also the result of the test data prediction is stored in a csv file along with the Id of the house for which the prediction is done

Pridicting the prices of house is best done using a regression model as the output is not a fixed number of classes but instead is continious which can vbe best described using regression.
What the model does is given the input values, the system tries to learn parameters for a curve that best fits the data. This is done in the training phase. Ones the equation of the curve with minimum cost is obtained,
the system is ready to substitute the values of given features into the quation to generate the cost(label).

### Contents of the folder:
The folder contains :
#### train.csv:
This file contains well labeled data that is used to train the classifier
#### test.csv:
This file contains unlabelled data for which we need to predict the costs
#### mySubmit.csv:
Contains of the valued predicted for the values of features in test.csv against the Id of the house, also taken from the test.csv. This file is uploaded to the kaggle house price prediction challenge.
#### data_description.txt:
This file contains the details explaining the meaning of various columns in the dataset
#### House_pricing.ipynb:
Consists of jupiter notebook codes that implement the prediction program.
#### House_price.py:
Contains the same codes as in the notebook.(will not necessarily execute as the notebook. The code has been simply copied and pasted)
