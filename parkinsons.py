import pandas as pd

#Importing Dataset
a=pd.read_csv('/home/srees/Downloads/parkinsons.csv')

#The dataset consists of 24 attributes and 195 observations in which the target or labels are 'status'
a.head()
a.shape

a.dtypes
#For Visualization
import seaborn as sns
sns.catplot(x='status',kind='count',data=a)
for i in a:
    if i != 'status' and i != 'name':
        sns.catplot(x='status',y=i,kind='box',data=a)

"""
From the boxplot shown above it is very evident that if a patient has a lower rate of 'HNR','MDVP:Flo(Hz)','MDVP:Fhi(Hz)','MDVP:Fo(Hz)' ,then he/she is affected by parkinsons disease.
Dropping the attribute 'name' as it provides no useful insight.
"""
b=a.drop(['name'],axis=1)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,mean_squared_error

"""
The dataset is split into features(the independent variables) and labels(the dependent or target variable):
"""
features=a.drop(['status','name'],axis=1)
labels=a['status']

"""
Let's normalize the data using the minmax scaler to bring the feature variables within the range -1 to 1:
"""

scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels

"""
Splitting of dataset:

Here the split for training data is 80% and testing is 20% and a random state of 5 is given so aa to pick values at random at a count of 5.
"""

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=5)

#Importing various classification algorithm to find which algorithm suits the best for the dataset
#cross validation:
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRFClassifier,XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier,RandomForestClassifier

"""
Cross validation:

The cross validation score of various classification algorithms are used to check the average accuracy of the algorithms:
"""

print('log reg',lr,lr.mean())
print('xgbd',xgbc,xgbc.mean())
print('xgb',xgb,xgb.mean())
print('svm',svm,svm.mean())
#print('nb',nb,nb.mean)
print('dtc',dtc,dtc.mean())
print('adb',adb,adb.mean())
print('bbc',bbc,bbc.mean())
print('etc',etc,etc.mean())
print('gbc',gbc,gbc.mean())
print('rfc',rfc,rfc.mean())

"""
All the cross valildation scores of various algorithms are shown above.
 Here the xgboost and the extra tree classifier algorithms has a high rate of accuracy of 89% and 91% respectively,
hence we will use both these algorithms to fit two models and find the best model out of them.
"""

#XGboost
model=XGBClassifier()
model.fit(x_train,y_train)
y_predtr=model.predict(x_train)
print(accuracy_score(y_train,y_predtr)*100)
y_pred=model.predict(x_test)
    print(accuracy_score(y_test, y_pred)*100)

"""

After the model is fit the accuracy of the training data is 100% and the testing data is 92%. 
92% of the time this model would correctly predict the status of the patient.
"""
#Extra trees classifier
"""

Here the extratrees classifier algorithm is used to fit a model
(no parameter tuning is done)
"""
model=ExtraTreesClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)
y_predtr=model.predict(x_train)
print(accuracy_score(y_train,y_predtr)*100)

"""
Comparing the xgboost algorithm , 
the extra trees classifier has a better accuracy. 
In this model the rate of accuracy for the test data is 97% hence this model is the best suited for this dataset.
"""

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
y_pred=pd.DataFrame(y_pred)
print("The predicted results of the test data are shown data:")
print(y_pred)

"""
Therefore, the extra trees classifier algorithm is used in this model to predict the status of the patient. 
This model could be very helpful in early diagnosis of the disease.
"















