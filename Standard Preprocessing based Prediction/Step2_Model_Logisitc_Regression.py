"""
Created on Sun Feb 11 21:44:55 2018

@author: Gomathypriya Dhanapal
@University: National University of Singapore
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("train_derived2.csv")
test = pd.read_csv("test_derived2.csv")

swap_columns = ['OA_MAIS', 'GV_CURBWGT', 'GV_DVLAT', 'GV_DVLONG','GV_LANES', 'GV_OTVEHWGT', 'GV_SPLIMIT', 
 'GV_MODELYR_2000', 'GV_MODELYR_2001','GV_MODELYR_2002', 'GV_MODELYR_2003', 'GV_MODELYR_2004',
 'GV_MODELYR_2005', 'GV_MODELYR_2006', 'GV_MODELYR_2007','GV_MODELYR_2008', 'GV_MODELYR_2009', 
 'GV_MODELYR_2010','GV_MODELYR_2011', 'GV_MODELYR_2012', 
 'GV_WGTCDTR_Passenger Car','GV_WGTCDTR_Truck (<=10000 lbs.)', 'GV_WGTCDTR_Truck (<=6000 lbs.)',
 'Energy_bag','Airbag_Seatbelt', 'VE_GAD_2c', 'Driver_side', 'Age_3c',
 'VE_PDOF_TR','GV_ENERGY','OA_HEIGHT',
 'OA_SEX_Female','OA_SEX_Male', 'OA_SEX_missing','OA_AGE', 'OA_WEIGHT',
 'GV_FOOTPRINT','VE_ORIGAVTW', 'VE_WHEELBAS','OA_BAGDEPLY_Deployed', 'OA_BAGDEPLY_Not Deployed', 
 'OA_MANUSE_0.0','OA_MANUSE_1.0', 'OA_MANUSE_missing',
 'VE_GAD1_Front', 'VE_GAD1_Left','VE_GAD1_Rear','VE_GAD1_Right', 'VE_GAD1_missing']

variables = ['OA_MAIS', 'GV_CURBWGT', 'GV_DVLAT', 'GV_DVLONG','GV_LANES', 'GV_OTVEHWGT', 'GV_SPLIMIT', 
 'GV_MODELYR_2000', 'GV_MODELYR_2001','GV_MODELYR_2002', 'GV_MODELYR_2003', 'GV_MODELYR_2004',
 'GV_MODELYR_2005', 'GV_MODELYR_2006', 'GV_MODELYR_2007','GV_MODELYR_2008', 'GV_MODELYR_2009', 
 'GV_MODELYR_2010','GV_MODELYR_2011', 'GV_MODELYR_2012', 
 'GV_WGTCDTR_Passenger Car','GV_WGTCDTR_Truck (<=10000 lbs.)', 'GV_WGTCDTR_Truck (<=6000 lbs.)',
 'Energy_bag','Airbag_Seatbelt', 'VE_GAD_2c', 'Driver_side', 'Age_3c']

train1 = train[variables]
test1 = test[variables]

categorical_columns = ['Energy_bag','Airbag_Seatbelt', 'VE_GAD_2c', 'Driver_side', 'Age_3c']
     
#one_hot_encoding
train1 = pd.get_dummies(train1,columns=categorical_columns)
test1 = pd.get_dummies(test1,columns=categorical_columns)

data_final_vars=train1.columns.values.tolist()
target=[data_final_vars[0]]
inputs=[i for i in data_final_vars if i not in target]

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 18)
rfe = rfe.fit(train1[inputs], train1[target] )
print(rfe.support_)
print(rfe.ranking_)

#the following variables were obtained from rfe.suppport when run in python notebook
select_cols = ['GV_CURBWGT','GV_LANES','GV_OTVEHWGT',
 'GV_MODELYR_2006','GV_MODELYR_2008','GV_MODELYR_2009',
 'GV_MODELYR_2011','GV_WGTCDTR_Truck (<=10000 lbs.)',
 'GV_WGTCDTR_Truck (<=6000 lbs.)','Energy_bag_0','Energy_bag_1','Energy_bag_3',
 'Airbag_Seatbelt_1',
 'Airbag_Seatbelt_2',
 'Airbag_Seatbelt_3',
 'Driver_side_0',
 'Age_3c_0',
 'Age_3c_1']

X=train1[select_cols]
y=train1[target]
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())


#select cols with P <0.05
select_cols1 = ['GV_LANES','GV_OTVEHWGT','GV_MODELYR_2008',
 'GV_WGTCDTR_Truck (<=6000 lbs.)','Energy_bag_0','Energy_bag_1','Energy_bag_3',
 'Airbag_Seatbelt_1',
 'Airbag_Seatbelt_2',
 'Airbag_Seatbelt_3',
 'Driver_side_0',
 'Age_3c_0',
 'Age_3c_1']

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
X=train1[select_cols1]
logreg.fit(X, np.ravel(y))

X_t=test1[select_cols1]
y_t=test1[target]
y_pred = logreg.predict(X_t)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_t, y_t)))

from sklearn.metrics import classification_report
print(classification_report(y_t, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

logit_roc_auc = roc_auc_score(y_t, logreg.predict(X_t))
fpr, tpr, thresholds = roc_curve(y_t, logreg.predict_proba(X_t)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-Logistic Regression')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
#plt.show()

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X, np.ravel(y), cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

df = pd.DataFrame({'Actual': y_t['OA_MAIS'], 'Pred':y_pred})
print (pd.crosstab(df.Actual >0 , df.Pred > 0))
