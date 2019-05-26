# The hand-written digit recognition
### Imports:
##### Pandas:
Pandas is a powerful open source tool for data manipulation and analysis suing easy to use data structures.
We use pandas in this project only to read the predefined datasets for training and testing the classifier

##### Opencv:
Python image processing library provides many easy methods for image handeling and manipulation. We however , use the library to view the image 
of the digit for which the prediction is being made. Just for fun

##### Sklearn:
Provides many easy to use tools for creationn and uitlization of machine learning models. We use two classifiers to predict the input digit image
We use random forest(which gave better results) and decesion tree classifiers.
##### CSV:
We write out our predictions into a csv file named mySubmit.csv . For simplifying the task we import csv and use the writer to write data row by row
into the mySubmit.csv

###### The part of code that enable to see the image of the digit has been commented out. Although it can be used by minor changes. Also if opencv is not installed, matplotlib can be used for the task.

### Content of the folder:
The folders consists of 
#### digit_recognizer.py:
This python file consists of the codes to manipulate dataset and make predictions for the testing data, or a new input data.
For using it for an input data though, changes have to be done to take the input from the user. Or to read the input image, processs it to fit the appropriate size and then change the dimensions to a 1 dimentional array to be fed to the program.

#### mySubmit.csv:
This file consists of the predicted values recieved when the testing data was passed from the program.
The file consists of two entries: 1. The ImageId-basically the image number in the test dataset. 2. Label-The predicted class(i.e predicted digit). This file is uploded to kaggle for evaluation.
#### test_train_data.zip:
This zip file contains two csv files
###### train.csv:
This file contains prelabeled images. Data from this file is used to train the classifier. Consists of 42000 rows (i.e. 42000 28x28 grescaled images of digits)
###### test.csv:
This file contains data for 28000 images of digits. The data from this file is used to test the predictions of the classifier.

### Performance:
The submitted file gives a score of 94.071%
