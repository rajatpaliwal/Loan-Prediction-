
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
from time import time
from IPython.display import display

get_ipython().magic(u'matplotlib inline')
data = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
data_1= pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")
data = data.drop('Loan_ID',axis=1)





# In[6]:


n_records= len(data)
n_records_test= len(data_1)
print("Total number of records: {}".format(n_records))
print("Total number of records: {}".format(n_records_test))
display(data.head(n=10))


data['Loan_Status_Label'] = data['Loan_Status'].map(lambda x: 1 if x =='Y' else 0)


# In[7]:


Loan_Status_raw= data['Loan_Status_Label']
features_raw = data.drop('Loan_Status_Label', axis = 1)
train_raw = features_raw.drop('Loan_Status', axis=1)
display (train_raw.head(n=5))


# In[11]:


Test_Loan_ID = data_1['Loan_ID']
data_1= data_1.drop('Loan_ID',axis=1)
print(data_1.isnull().sum())
print(data.isnull().sum())


# In[12]:


from sklearn.preprocessing import LabelEncoder
import pandas as pd
features_final = pd.get_dummies(train_raw)
features_final_test= pd.get_dummies(data_1)


le = LabelEncoder()
le.fit(Loan_Status_raw)
Loan = le.transform(Loan_Status_raw)
encoded = list(features_final.columns)
encoded_test = list(features_final_test.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))
print("{} total features after one-hot encoding.".format(len(encoded_test)))
print Loan


# In[17]:


TP = np.sum(Loan_Status_raw)
TN = 0
FN = 0
n_Loan_given = Loan_Status_raw.value_counts()[1]
n_Loan_not_given = Loan_Status_raw.value_counts()[0]
Loan_given_percent = (float(n_Loan_given)*100/float(n_records))
accuracy = (Loan_given_percent/100)
recall = 1.0
precision = accuracy
fscore = ((precision*recall)*(1+0.5*0.5))/(((0.5*0.5)*precision)+recall)
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


# In[30]:


from sklearn.preprocessing import Imputer
import numpy
values = features_final_test.values
val = features_final.values
imputer = Imputer()
transformed_values = imputer.fit_transform(values)
training = imputer.fit_transform(val)
print(numpy.isnan(training).sum())
print training


# In[31]:


X_train= training
y_train = Loan
X_test = transformed_values
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))




# In[82]:


from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
pca = PCA()
X_transformed = pca.fit_transform(X_train)
logreg = LogisticRegression()
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
  
grid_obj = GridSearchCV(logreg, parameters)
grid_fit = grid_obj.fit(X_transformed,y_train)
best_clf = grid_fit.best_estimator_
X_test_transformed = pca.transform(X_test)

best_predictions = best_clf.predict(X_test_transformed)
popularity= np.array(best_predictions)[np.newaxis]
mid = pd.DataFrame(popularity.T)
mid = mid.replace(1, 'Y')
mid = mid.replace(0,'N')
mid.to_csv("prediction.csv",index=False, header=None)







# In[75]:


from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier().fit(clk_train,y_train)
importances = model.feature_importances_


# In[71]:


from sklearn.base import clone
X_train_reduced = clk_train[clk_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = clk_test[clk_test.columns.values[(np.argsort(importances)[::-1])[:5]]]
clf = (clone(clf_C)).fit(X_train_reduced, y_train)
reduced_predictions = clf.predict(X_test_reduced)
popularity= np.array(reduced_predictions)[np.newaxis]
mid = pd.DataFrame(popularity.T)
mid = mid.replace(1, 'Y')
mid = mid.replace(0,'N')
mid.to_csv("prediction.csv",index=False, header=None)

