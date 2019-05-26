"""
hand written digit recognition
using dataset provided by kaggle
decesion tree is used to classify the image into one of the classes of the digit
images can be viewed and loaded using opencv

Author- Adarsh Dubey
"""
import numpy as np
import csv
import cv2
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dataset=pd.read_csv(r'C:\Users\ALOK DUBEY\Desktop\digit_recognition\train.csv').as_matrix()
#read dataset as matrix for easier slicing
labels=dataset[:,0]#first column is label
features=dataset[:,1:]#rest of the columns is pixel values hence features
#instantiate the clasifier
clf=DecisionTreeClassifier()
rfclf=RandomForestClassifier()

#fit the data to the classifier

clf.fit(features,labels)#not needed

rfclf.fit(features,labels)

#classifier is ready

#load the test dataset
testData=pd.read_csv(r'C:\Users\ALOK DUBEY\Desktop\digit_recognition\test.csv')

#predictions

predict=rfclf.predict(testData)
#see the digit
#testData[0].shape=(28,28)
#cv2.imshow('digit',testData)
#new predictions
#print(predict)
#write data in csv submit file

with open(r'C:\Users\ALOK DUBEY\Desktop\digit_recognition\mySubmit.csv','w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['ImageId','Label'])
    for index,label in enumerate(predict):
        writer.writerow([index+1,label])
#random forest generates better results
