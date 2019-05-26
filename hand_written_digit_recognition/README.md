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

