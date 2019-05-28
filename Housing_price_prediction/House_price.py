import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
data=pd.read_csv('train.csv')
data.drop(columns=['Id'],inplace=True)
##############################################################
#cleaning data for regression
#remove NA and put -1 instead
data=data.fillna(-1)
#MSZoning col
data['MSZoning']=data['MSZoning'].replace('RL',0)
data['MSZoning']=data['MSZoning'].replace('RM',1)
data['MSZoning']=data['MSZoning'].replace('FV',2)
data['MSZoning']=data['MSZoning'].replace('A',3)
data['MSZoning']=data['MSZoning'].replace('C',4)
data['MSZoning']=data['MSZoning'].replace('I',5)
data['MSZoning']=data['MSZoning'].replace('RH',6)
data['MSZoning']=data['MSZoning'].replace('RP',7)
################################################################
#Street col
data['Street']=data['Street'].replace('Grvl',0)
data['Street']=data['Street'].replace('Pave',1)
###############################################################
#Alley col
data['Alley']=data['Alley'].replace(-1,0)#NA here means no alley available
data['Alley']=data['Alley'].replace('Grvl',1)
data['Alley']=data['Alley'].replace('Pave',2)
################################################################
#LotShape col
data=data.replace({'LotShape' : { 'Reg' : 0, 'IR1' : 1, 'IR2' : 2,'IR3':3 }})
data=data.replace({'LandContour':{'Lvl':0,'Bnk':1,'HLS':2,'Low':3}})
data=data.replace('Brk Cmn','BrkCmn')
#################################################################
#To be honest alll the above can be done at once likke so:

data=data.replace({'Utilities' :{'AllPub':0,'NoSewr':1,'NoSeWa':2,'ELO':3},
                   'LotConfig':{'Inside':0,'Corner':1,'CulDSac':2,'FR2':3,'FR3':4},
                   'LandSlope':{'Gtl':0,'Mod':1,'Sev':2},
                   'Neighborhood':{'Blmngtn':0,'Blueste':1,'BrDale':2,'BrkSide':3,'ClearCr':4,'CollgCr':5,'Crawfor':6,'Edwards':7,'Gilbert':8,'IDOTRR':9,'MeadowV':10,'Mitchel':11,'NAmes':12,'NoRidge':13,'NPkVill':14,'NridgHt':15,'NWAmes':16,'OldTown':17,'SWISU':18,'Sawyer':19,'SawyerW':20,'Somerst':21,'StoneBr':22,'Timber':23,'Veenker':24},
                   'Condition1':{'Artery':0,'Feedr':1,'Norm':2,'RRNn':3,'RRAn':4,'PosN':5,'PosA':6,'RRNe':7,'RRAe':8},
                   'Condition2':{'Artery':0,'Feedr':1,'Norm':2,'RRNn':3,'RRAn':4,'PosN':5,'PosA':6,'RRNe':7,'RRAe':8},
                   'BldgType':{'1Fam':0,'2fmCon':1,'Duplex':2,'TwnhsE':3,'Twnhs':4},
                   'HouseStyle':{'1Story':0,'1.5Fin':1,'1.5Unf':2,'2Story':3,'2.5Fin':4,'2.5Unf':5,'SFoyer':6,'SLvl':7},
                   'RoofStyle':{'Flat':0,'Gable':1,'Gambrel':2,'Hip':3,'Mansard':4,'Shed':5},
                   'RoofMatl':{'ClyTile':0,'CompShg':1,'Membran':2,'Metal':3,'Roll':4,'Tar&Grv':5,'WdShake':6,'WdShngl':7},
                   'Exterior1st':{'AsbShng':0,'AsphShn':1,'BrkComm':2,'BrkFace':3,'CBlock':4,'CemntBd':5,'HdBoard':6,'ImStucc':7,
                                 'MetalSd':8,'Other':9,'Plywood':10,'PreCast':11,'Stone':12,'Stucco':13,'VinylSd':14,'Wd Sdng':15,
                                 'WdShing':16,'Wd Shng':16,'WdShng':18},
                   'Exterior2nd':{'AsbShng':0,'AsphShn':1,'BrkComm':2,'BrkFace':3,'CBlock':4,'CmentBd':5,'HdBoard':6,'ImStucc':7,
                                 'MetalSd':8,'Other':9,'Plywood':10,'PreCast':11,'Stone':12,'Stucco':13,'VinylSd':14,'Wd Sdng':15,
                                 'WdShing':16,'Wd Shng':17,'WdShng':18},
                   'MasVnrType':{'Brk Cmn':-1,'BrkCmn':0,'BrkFace':1,'CBlock':2,'None':3,'Stone':4},
                   'ExterQual':{'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0},
                   'ExterCond':{'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0},
                   'Foundation':{'BrkTil':0,'CBlock':1,'PConc':2,'Slab':3,'Stone':4,'Wood':5},
                   'BsmtQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,-1:0},
                   'BsmtCond':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,-1:0},
                   'BsmtExposure':{'No':0,'Mn':1,'Av':2,'Gd':3},
                   'BsmtFinType1':{'GLQ':5,'ALQ':4,'BLQ':3,'Rec':2,'LwQ':1,'Unf':0},
                   'BsmtFinType2':{'GLQ':5,'ALQ':4,'BLQ':3,'Rec':2,'LwQ':1,'Unf':0},
                   'Heating':{'Floor':0,'GasA':1,'GasW':2,'Grav':3,'OthW':4,'Wall':5},
                   'HeatingQC':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},
                   'CentralAir':{'N':0,'Y':1},
                   'Electrical': {'SBrkr':0,'FuseA':1,'FuseF':2,'FuseP':3,'Mix':4},
                   'KitchenQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},
                   'Functional':{'Typ':0,'Min1':1,'Min2':2,'Mod':3,'Maj1':4,'Maj2':5,'Sev':6,'Sal':7},
                   'FireplaceQu':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},
                   'GarageType':{'2Types':0,'Attchd':1,'Basment':2,'BuiltIn':3,'CarPort':4,'Detchd':5},
                   'GarageFinish':{'Fin':0,'RFn':1,'Unf':2},
                   'GarageQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},
                   'GarageCond':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},
                   'PavedDrive':{'Y':1,'N':2,'P':3},
                   'PoolQC':{'Ex':5,'Gd':4,'TA':3,'Fa':2},
                   'Fence':{'GdPrv':0,'MnPrv':1,'GdWo':2,'MnWw':3},
                   'MiscFeature':{'Elev':0,'Gar2':1,'Othr':3,'Shed':4,'TenC':5},
                   'SaleType':{'WD':0,'CWD':1,'VWD':2,'New':3,'COD':4,'Con':5,'ConLw':6,'ConLI':7,'ConLD':8,'Oth':9},
                   'SaleCondition':{'Normal':0,'Abnorml':1,'AdjLand':2,'Alloca':3,'Family':4,'Partial':5}
                  })
data=data.replace('C (all)',.1)
data=data.replace('BrkCmn',0)
#################################################################################################################################################################

#the machine starts to learn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
label=data['SalePrice']
features=data
features.drop(columns=['SalePrice'],inplace=True)

X_train,X_test,y_train,y_test=train_test_split(features,label,test_size=0.30)
#########################################################################################################################
from sklearn.linear_model import Ridge
clf1=Ridge(alpha=1.0)
clf1.fit(X_train,y_train)
#############################################################################################################################
from sklearn.model_selection import cross_val_score
max(cross_val_score(clf1, X_test, y_test))
#gives accurecy of 84%
##############################################################################################################################
#loading test data
df_test=pd.read_csv(r'C:\Users\ALOK DUBEY\Desktop\housing_prices\test.csv')
#df_test.head()
ids=list(df_test['Id'])
df_test.drop(columns=['Id'],inplace=True)
####################################################################################################################################
#beautification of test data
#remove NA and put -1 instead
df_test=df_test.fillna(-1)
df_test['MSZoning']=df_test['MSZoning'].replace('RL',0)
df_test['MSZoning']=df_test['MSZoning'].replace('RM',1)
df_test['MSZoning']=df_test['MSZoning'].replace('FV',2)
df_test['MSZoning']=df_test['MSZoning'].replace('A',3)
df_test['MSZoning']=df_test['MSZoning'].replace('C',4)
df_test['MSZoning']=df_test['MSZoning'].replace('I',5)
df_test['MSZoning']=df_test['MSZoning'].replace('RH',6)
df_test['MSZoning']=df_test['MSZoning'].replace('RP',7)
df_test['Street']=df_test['Street'].replace('Grvl',0)
df_test['Street']=df_test['Street'].replace('Pave',1)
df_test['Alley']=df_test['Alley'].replace(-1,0)
df_test['Alley']=df_test['Alley'].replace('Grvl',1)
df_test['Alley']=df_test['Alley'].replace('Pave',2)
df_test=df_test.replace({'LotShape' : { 'Reg' : 0, 'IR1' : 1, 'IR2' : 2,'IR3':3 }})
df_test=df_test.replace({'LandContour':{'Lvl':0,'Bnk':1,'HLS':2,'Low':3}})
df_test=df_test.replace('Brk Cmn','BrkCmn')
df_test=df_test.replace({'Utilities' :{'AllPub':0,'NoSewr':1,'NoSeWa':2,'ELO':3},
                   'LotConfig':{'Inside':0,'Corner':1,'CulDSac':2,'FR2':3,'FR3':4},
                   'LandSlope':{'Gtl':0,'Mod':1,'Sev':2},
                   'Neighborhood':{'Blmngtn':0,'Blueste':1,'BrDale':2,'BrkSide':3,'ClearCr':4,'CollgCr':5,'Crawfor':6,'Edwards':7,'Gilbert':8,'IDOTRR':9,'MeadowV':10,'Mitchel':11,'NAmes':12,'NoRidge':13,'NPkVill':14,'NridgHt':15,'NWAmes':16,'OldTown':17,'SWISU':18,'Sawyer':19,'SawyerW':20,'Somerst':21,'StoneBr':22,'Timber':23,'Veenker':24},
                   'Condition1':{'Artery':0,'Feedr':1,'Norm':2,'RRNn':3,'RRAn':4,'PosN':5,'PosA':6,'RRNe':7,'RRAe':8},
                   'Condition2':{'Artery':0,'Feedr':1,'Norm':2,'RRNn':3,'RRAn':4,'PosN':5,'PosA':6,'RRNe':7,'RRAe':8},
                   'BldgType':{'1Fam':0,'2fmCon':1,'Duplex':2,'TwnhsE':3,'Twnhs':4},
                   'HouseStyle':{'1Story':0,'1.5Fin':1,'1.5Unf':2,'2Story':3,'2.5Fin':4,'2.5Unf':5,'SFoyer':6,'SLvl':7},
                   'RoofStyle':{'Flat':0,'Gable':1,'Gambrel':2,'Hip':3,'Mansard':4,'Shed':5},
                   'RoofMatl':{'ClyTile':0,'CompShg':1,'Membran':2,'Metal':3,'Roll':4,'Tar&Grv':5,'WdShake':6,'WdShngl':7},
                   'Exterior1st':{'AsbShng':0,'AsphShn':1,'BrkComm':2,'BrkFace':3,'CBlock':4,'CemntBd':5,'HdBoard':6,'ImStucc':7,
                                 'MetalSd':8,'Other':9,'Plywood':10,'PreCast':11,'Stone':12,'Stucco':13,'VinylSd':14,'Wd Sdng':15,
                                 'WdShing':16,'Wd Shng':16,'WdShng':18},
                   'Exterior2nd':{'AsbShng':0,'AsphShn':1,'BrkComm':2,'BrkFace':3,'CBlock':4,'CmentBd':5,'HdBoard':6,'ImStucc':7,
                                 'MetalSd':8,'Other':9,'Plywood':10,'PreCast':11,'Stone':12,'Stucco':13,'VinylSd':14,'Wd Sdng':15,
                                 'WdShing':16,'Wd Shng':17,'WdShng':18},
                   'MasVnrType':{'Brk Cmn':-1,'BrkCmn':0,'BrkFace':1,'CBlock':2,'None':3,'Stone':4},
                   'ExterQual':{'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0},
                   'ExterCond':{'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0},
                   'Foundation':{'BrkTil':0,'CBlock':1,'PConc':2,'Slab':3,'Stone':4,'Wood':5},
                   'BsmtQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,-1:0},
                   'BsmtCond':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,-1:0},
                   'BsmtExposure':{'No':0,'Mn':1,'Av':2,'Gd':3},
                   'BsmtFinType1':{'GLQ':5,'ALQ':4,'BLQ':3,'Rec':2,'LwQ':1,'Unf':0},
                   'BsmtFinType2':{'GLQ':5,'ALQ':4,'BLQ':3,'Rec':2,'LwQ':1,'Unf':0},
                   'Heating':{'Floor':0,'GasA':1,'GasW':2,'Grav':3,'OthW':4,'Wall':5},
                   'HeatingQC':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},
                   'CentralAir':{'N':0,'Y':1},
                   'Electrical': {'SBrkr':0,'FuseA':1,'FuseF':2,'FuseP':3,'Mix':4},
                   'KitchenQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},
                   'Functional':{'Typ':0,'Min1':1,'Min2':2,'Mod':3,'Maj1':4,'Maj2':5,'Sev':6,'Sal':7},
                   'FireplaceQu':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},
                   'GarageType':{'2Types':0,'Attchd':1,'Basment':2,'BuiltIn':3,'CarPort':4,'Detchd':5},
                   'GarageFinish':{'Fin':0,'RFn':1,'Unf':2},
                   'GarageQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},
                   'GarageCond':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},
                   'PavedDrive':{'Y':1,'N':2,'P':3},
                   'PoolQC':{'Ex':5,'Gd':4,'TA':3,'Fa':2},
                   'Fence':{'GdPrv':0,'MnPrv':1,'GdWo':2,'MnWw':3},
                   'MiscFeature':{'Elev':0,'Gar2':1,'Othr':3,'Shed':4,'TenC':5},
                   'SaleType':{'WD':0,'CWD':1,'VWD':2,'New':3,'COD':4,'Con':5,'ConLw':6,'ConLI':7,'ConLD':8,'Oth':9},
                   'SaleCondition':{'Normal':0,'Abnorml':1,'AdjLand':2,'Alloca':3,'Family':4,'Partial':5}
                  })
df_test=df_test.replace('C (all)',.1)
df_test=df_test.replace('BrkCmn',0)
##################################################################################################################################################################
predictions=list(clf.predict(df_test))
for i,v in enumerate(predictions):
    predictions[i]="%.1f"%v
################################################################################################################################################
#just writing the data to mySubmit.csv
import csv
with open(r'C:\Users\ALOK DUBEY\Desktop\housing_prices\mySubmit.csv','w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['Id','SalePrice'])
    for Id,price in zip(ids,predictions):
        writer.writerow([Id,price])
####################################################################----The End----#################################################################################
