# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 21:19:47 2019

@author: dashi
"""
##############################################################################
##Random Forest Classifier
##############################################################################

#Import scikit-learn dataset library
# from sklearn import datasets
import pandas as pd

#Load dataset
smallData = pd.read_csv("smalldataset_Final_Python_2.csv", sep =";")
#Incase first column is 'Unnamed'
#smallData.drop(smallData.columns[[0]], axis=1, inplace=True)



# Import train_test_split function
from sklearn.model_selection import train_test_split

X=smallData[['WordCountFactor', 'Impact_SocialMedia', 'Format_Urgency', 
             'Impact_Document_Content', 'Response_Urgency_NGO_Type']]  # Features
y=smallData['PriorityIndex_Lookup']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Perform Prediction
#Insert [[WordCountFactor (Value between 0-100), 
#         Impact_SocialMedia (Value between 0-1000), 
#         Format_Urgency (Value between 0-100), 
#         Impact_Document_Content (Value between 0-100), 
#         Response_Urgency_NGO_Type (value between 0-100)]]
clf.predict([[50, 50, 40, 20]])

#Result is value between 0-10 which is Priority Index Lookup 10 being Higher Priority

#Importance Scores from the Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

import pandas as pd
feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
feature_imp



import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


##############################################################################
##Part 2- Remove lowest Feature Value and Re-Do Random Forest Classifier
##############################################################################


# Import train_test_split function
#from sklearn.cross_validation import train_test_split
# Split dataset into features and labels
X_mod=smallData[['WordCountFactor', 'Impact_SocialMedia', 'Impact_Document_Content', 'Response_Urgency_NGO_Type']]  # Removed lowest feature
y_mod=smallData['PriorityIndex_Lookup']                                       
# Split dataset into training set and test set
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_mod, y_mod, test_size=0.70, random_state=5) # 70% training and 30% test


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train_2,y_train_2)

# prediction on test set
y_pred_2=clf.predict(X_test_2)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
##############################################################################
