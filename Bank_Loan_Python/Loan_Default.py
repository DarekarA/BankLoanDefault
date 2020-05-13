#!/usr/bin/env python
# coding: utf-8

# <center>
#     <h1>Bank Loan Default Case</h1>
# </center>
# 
# **Background:** 
# The loan default dataset has 8 variables and 850 records, each record being loan default status for each customer. Each Applicant was rated as “Defaulted” or “Not-Defaulted”. New applicants for loan application can also be evaluated on these 8 predictor variables and classified as a default or non-default based on predictor variables.  
# 

# # 1. Set Environment and load packages

# In[4]:


import os


# In[5]:


os.chdir("C:/Users/Abhishek/Desktop/Bank_Loan_Edwisor/Bank_Loan_Python")


# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate


# # 2. Load dataset and Data Pre Processing

# ## Data Pre-Processing 

# ### Load Dataset

# In[7]:


bankloans = pd.read_csv("bank-loan.csv")


# In[8]:


bankloans.head()


# In[9]:


bankloans.columns


# In[10]:


#number of observations and features
bankloans.shape


# In[11]:


#data types in the dataframe
bankloans.info()


# # 3.  Checking for missing values

# In[12]:


#check for any column has missing values
bankloans.isnull().any()


# In[13]:


#check for number of missing values
bankloans.isnull().sum()


# # 4.1 Separate the numeric and categorical variable names & 
# # 4.2 splitting the data set into two sets - existing customers and new customers
#        
#        

# In[14]:


#Segregating the numeric and categorical variable names

numeric_var_names = [key for key in dict(bankloans.dtypes) if dict(bankloans.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]
catgorical_var_names = [key for key in dict(bankloans.dtypes) if dict(bankloans.dtypes)[key] in ['object']]


# In[15]:


numeric_var_names


# In[16]:


#splitting the data set into two sets - existing customers and new customers

bankloans_existing = bankloans.loc[bankloans.default.isnull() == 0] #isnulll is false
bankloans_new = bankloans.loc[bankloans.default.isnull() == 1] #isnull is true


# In[17]:


bankloans_existing.shape


# In[18]:


bankloans_existing.describe(percentiles=[.25,0.5,0.75,0.90,0.95])


# # 5. Checking for Outliers

# In[19]:


sns.boxplot(y = "age",data=bankloans_existing)
plt.title("Box-Plot of age")
plt.show()


# In[20]:


sns.boxplot(y = "employ",data=bankloans_existing)
plt.title("Box-Plot of employee tenure")
plt.show()


# In[21]:


sns.boxplot(y = "income",data=bankloans_existing)
plt.title("Box-Plot of employee income")
plt.show()


# In[22]:


sns.boxplot(y = "debtinc",data=bankloans_existing)
plt.title("Box-Plot of employee debt to income ratio")
plt.show()


# In[23]:


sns.boxplot(y = "creddebt",data=bankloans_existing)
plt.title("Box-Plot of Credit to debit ratio")
plt.show()


# In[24]:


income_minlimit = bankloans_existing["income"].quantile(0.75) + 1.5 * (bankloans_existing["income"].quantile(0.75) - bankloans_existing["income"].quantile(0.25))
income_minlimit


# #clip_upper() is used to trim values at specified input threshold. We use this function to trim all the values above the threshold of the input value to the specified input value.
# #quantile() function return values at the given quantile over requested axis, a numpy.percentile.
# 

# #Trim as in it deletes the outlier and replaces it with the upper limit value

# In[25]:


def outlier_capping(x):
    """A funtion to remove and replace the outliers for numerical columns"""
    x = x.clip_upper(x.quantile(0.95))
    
    return(x)


# # 5.1 outlier treatment

# In[26]:


#outlier treatment
bankloans_existing = bankloans_existing.apply(lambda x: outlier_capping(x))


# In[27]:


bankloans_existing.isnull().any()


#     # 6 Correlation

# In[28]:


##Correlation Matrix
bankloans_existing.corr()


# In[29]:


#Visualize the correlation using seaborn heatmap

sns.heatmap(bankloans_existing.corr(),annot=True,fmt="0.2f",cmap="coolwarm")
plt.show()


# In[30]:


bankloans_existing.shape


# In[31]:


bankloans_new.shape


# In[32]:


#Indicator variable unique types

bankloans_existing['default'].value_counts()


# In[33]:


bankloans_existing['default'].value_counts().plot.bar()
plt.xlabel("default")
plt.ylabel("count")
plt.title("Distribution of default")
plt.show()


# In[34]:


#percentage of unique types in indicator variable

round(bankloans_existing['default'].value_counts()/bankloans_existing.shape[0] * 100,3)


# In[ ]:





# ## Data Exploratory Analysis
# - Bivariate Analysis - Numeric(TTest)/ Categorical(Chisquare)
# - Bivariate Analysis - Visualization
# - Variable Reduction - Multicollinearity

# In[35]:


## performing the independent t test on numerical variables

tstats_df = pd.DataFrame()

for eachvariable in numeric_var_names:
    tstats = stats.ttest_ind(bankloans_existing.loc[bankloans_existing["default"] == 1,eachvariable],bankloans_existing.loc[bankloans_existing["default"] == 0, eachvariable],equal_var=False)
    temp = pd.DataFrame([eachvariable, tstats[0], tstats[1]]).T
    temp.columns = ['Variable Name', 'T-Statistic', 'P-Value']
    tstats_df = pd.concat([tstats_df, temp], axis=0, ignore_index=True)
    
tstats_df =  tstats_df.sort_values(by = "P-Value").reset_index(drop = True)


# # P <0.05 therefore we reject null hypo.
# Null hypo = 
# <br>
# Alternate hype = Bothe means from different distributions(which is that the population means are not equal)
# <br>
# P<0.05 ARE for :
# Age=true,Address(P: 0.00000207201)=true,creddebt(P: 0.000000390256)=true,etc

# In[36]:


tstats_df


# ### Bi-Variate Analysis

# In[37]:


def BivariateAnalysisPlot(segment_by):
    """A funtion to analyze the impact of features on the target variable"""
    
    fig, ax = plt.subplots(ncols=1,figsize = (10,8))
    
    #boxplot
    sns.boxplot(x = 'default', y = segment_by, data=bankloans_existing)
    plt.title("Box plot of "+segment_by)
    
    
    plt.show()
    


# In[38]:


BivariateAnalysisPlot("age")
#For default 0 the average age is 35, default 1 is 32


# In[39]:


BivariateAnalysisPlot("ed")
#Both Same


# In[40]:


BivariateAnalysisPlot("employ")
##For default 0 the avrage eploy is 8, default 1 is 3


# In[41]:


BivariateAnalysisPlot("address")
#For default 0 its 8, for default 1 its 5


# In[42]:


BivariateAnalysisPlot("income")
#For default 0 its 35, for default 1 its 25


# In[43]:


BivariateAnalysisPlot("debtinc")
#For default 0 its 8, for default 1 its 13


# In[44]:


BivariateAnalysisPlot("creddebt")
#For default 0 its 0.5, for default 1 its 1.5


# In[45]:


BivariateAnalysisPlot("othdebt")


#     ### Multi Collinearity Check

# In[ ]:





# In[46]:


from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[47]:


features = "+".join(bankloans_existing.columns.difference(["default"]))


# In[48]:


features


# In[49]:


#perform vif

a, b = dmatrices(formula_like= 'default ~ ' + features,data=bankloans_existing,return_type="dataframe")
vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(b.values, i) for i in range(b.shape[1])]
vif["Features"] = b.columns


# In[50]:


vif


# ### Observations
# ----
# <big>
# - There are 850 observations and 9 features in the data set
# - All 9 features are numerical in nature
# - There are no missing values in the data set
# - Out of 850 customers data, 700 are existing customers and 150 are new customers
# - In the 700 existing customers, 517 customers are tagged as non defaulters and remaining 183 are tagged as defaulters
# - The data is highly imbalanced
# - From VIF check, found out that the correlation between the variables is within the acceptable limits

# In[ ]:





# ## Model Building and Model Diagnostics
# 
#    - Logistic Regression
#    - Decision Tree classifier
# ---
# **Model Diagnostics**
# 
# - Train and Test split
# - Significance of each Variable
# - Gini and ROC / Concordance analysis
# - Classification Table Analysis - Accuracy

# # Model 1 : Logistic Regression

# In[51]:


#fearturecolumns has all columns except 'default'
featurecolumns = bankloans_existing.columns.difference(['default'])
featurecolumns


# In[52]:


#Train and test split

train_X,test_X,train_y,test_y = train_test_split(bankloans_existing[featurecolumns],
                                                 bankloans_existing['default'], stratify = bankloans_existing['default'], test_size = 0.2, random_state = 123)


# In[53]:


train_X.shape


# In[54]:


test_X.shape


# In[55]:


train_y.shape


# In[56]:


test_y.shape


# In[57]:


train_y.unique()


# In[58]:


round(train_y.value_counts()/train_y.shape[0] * 100,3)


# In[59]:


train_y.value_counts()


# # logreg.fit() to fit the model

# In[60]:


## Model Building

logreg = LogisticRegression()
logreg.fit(train_X,train_y)


# In[61]:


#Features and their coefficients

coefficient_df =  pd.DataFrame({'Features' : pd.Series(featurecolumns),
                        "Coefficients" : pd.Series(logreg.coef_[0])})
coefficient_df


# In[62]:


logreg.intercept_


# In[ ]:





# ### Model Performance 
# - Test data set

# #### Metrics
# 
# - Recall: Ratio of the total number of correctly classified positive examples divide to the total number of positive examples. High Recall indicates the class is correctly recognized
# - Precision: To get the value of precision we divide the total number of correctly classified positive examples by the total number of predicted positive examples. High Precision indicates an example labeled as positive is indeed positive

# # logreg.predict() to predict the Data
# #see differenve between predicted value and Actual value

# In[63]:


#Predicting the test cases
bankloans_test_pred_log = pd.DataFrame({'actual':test_y, 'predicted': logreg.predict(test_X)}) #here we predict
bankloans_test_pred_log = bankloans_test_pred_log.reset_index()
bankloans_test_pred_log.head()


# In[64]:


#creating a confusion matrix

cm_logreg = metrics.confusion_matrix(bankloans_test_pred_log.actual,
                                    bankloans_test_pred_log.predicted,labels = [1,0])
cm_logreg


# In[65]:


#Visualize the confusion matrix
sns.heatmap(cm_logreg,annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels = ["Default", "Not Default"] , yticklabels = ["Default", "Not Default"])
plt.title("Confusion Matrix for Test data")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


# In[66]:


#find precision score

prec_score = metrics.precision_score(bankloans_test_pred_log.actual, bankloans_test_pred_log.predicted)
print("Precision score :", round(prec_score,3))


# In[67]:


recall_score = metrics.recall_score(bankloans_test_pred_log.actual, bankloans_test_pred_log.predicted)
print("recall_score:", round(recall_score , 3))


# In[68]:


#find the overall accuracy of model

acc_score = metrics.accuracy_score(bankloans_test_pred_log.actual,bankloans_test_pred_log.predicted)
print("Accuracy of model :", round(acc_score,3))


# In[69]:


bankloans_test_pred_log.actual.value_counts()


# In[70]:


print(metrics.classification_report(bankloans_test_pred_log.actual, bankloans_test_pred_log.predicted))


# ##  Inference
# -----
# 
# <big>
# Overall test accuracy is 80%. But it is not a good measure. There are lot of cases which are default and the model has predicted them as not default. The objective of the model is to identify the customers who will default, so that the bank can intervene and act.This might be the case as the default model assumes people with more than 0.5 probability will not default. 
# </big>
# 

# # This might be the case as the default model assumes people with more than 0.5 probability will not default. 
# <br>
# 
# # So we now find the optimum cutoff value

# ### Find the optimum cutoff value

# In[71]:


#probabilty of prediction
#logreg.predict_proba() Function used
# here 0.789495… is the probability that the output will be 0 and 0.210505… is the probability of output being 1

predict_prob_df = pd.DataFrame(logreg.predict_proba(test_X))
predict_prob_df.head()


# In[72]:


bankloans_test_pred_log.head()


# In[73]:


predict_prob_df.head(1)


# In[74]:


bankloans_test_pred_log = pd.concat([bankloans_test_pred_log, predict_prob_df], axis = 1)
bankloans_test_pred_log.columns = ['index', 'actual', 'predicted', 'default_0','default_1']

bankloans_test_pred_log.head()


# In[75]:


#find the auc score
#AUC score for the case is 0.86. A score for a perfect classifier would be 1. Most often you get something in between.

auc_score = metrics.roc_auc_score(bankloans_test_pred_log.actual, bankloans_test_pred_log.default_1)
round(auc_score,2)


# In[76]:


#Draw a roc curve

fpr, tpr, thresholds = metrics.roc_curve(bankloans_test_pred_log.actual, bankloans_test_pred_log.default_1, 
                                         drop_intermediate= False)


plt.plot(fpr, tpr , label = 'ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Curve')
plt.legend(loc="lower right")
plt.show()


# <big>
# - Cutoff would be optimum where specificity(1-FPR) and sensitivity(TPR) would be maximum for the given cutoff

# In[77]:


##TPR - Sensitivity
##1-FPR - Specificity

i = np.arange(len(tpr))

roc_like_df = pd.DataFrame({'falsepositiverate' : pd.Series(fpr, index=i),'sensitivity' : pd.Series(tpr, index = i), 
              'specificity' : pd.Series(1-fpr, index = i),'cutoff' : pd.Series(thresholds, index = i)})
roc_like_df['total'] = roc_like_df['sensitivity'] + roc_like_df['specificity']


# In[78]:


#Here we see the highest cutoff is 0.224326
roc_like_df[roc_like_df['total']==roc_like_df['total'].max()]


# In[79]:


plt.subplots(figsize=(10,6))
plt.scatter(roc_like_df['cutoff'], roc_like_df['sensitivity'], marker='*', label='Sensitivity')
plt.scatter(roc_like_df['cutoff'], roc_like_df['specificity'], marker='*', label='Specificity')
plt.scatter(roc_like_df['cutoff'], roc_like_df['falsepositiverate'], marker='*', label='FPR')
plt.title('For each cutoff, pair of sensitivity and FPR is plotted for ROC')
plt.legend()

plt.show()

#My Assumption : x is cutoff , y is sens,spec,falsep etc. we choose 0.224 (sens & spec intercept there i.e it has highest value)


# In[80]:


#Predicting with new cut-off probability
bankloans_test_pred_log['new_labels'] = bankloans_test_pred_log['default_1'].map( lambda x: 1 if x >= 0.224326 else 0 )

bankloans_test_pred_log.head()


# In[81]:


#creating a confusion matrix

cm_logreg = metrics.confusion_matrix(bankloans_test_pred_log.actual,
                                    bankloans_test_pred_log.new_labels,labels = [1,0])
cm_logreg


# In[82]:


sns.heatmap(cm_logreg,annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels = ["Default", "Not Default"] , yticklabels = ["Default", "Not Default"])
plt.title("Confusion Matrix for Test data")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


# In[83]:


#classification report 

print(metrics.classification_report(bankloans_test_pred_log.actual,bankloans_test_pred_log.new_labels))


# In[84]:


#ability of the classifier to find all the positive samples

recall_score = metrics.recall_score(bankloans_test_pred_log.actual, bankloans_test_pred_log.new_labels)
print("recall_score:", round(recall_score , 3))


# In[85]:


#find the overall accuracy of model

acc_score = metrics.accuracy_score(bankloans_test_pred_log.actual,bankloans_test_pred_log.new_labels)
print("Accuracy of model :", round(acc_score,3))


# #### Inference
# -----
# 
# <big>
# Even though the overall accuracy of the model is reduced from 80% to 75% by taking optimum cutoff as 0.224, Model performance i.e recall score (ability of the model to find all the positive samples - find all the default customers) has increased from 54% to 89%. The drawback of changing the cutoff value can be seen in drastic drop of precision score (ability of model not to label non default customers as default customers) from 67% to 52%. 
# 
# </big>
# 
# - We have a choice to make depending on the value we place on the true positives and our tolerance for false postivies, in practical the cutoff values depends on the business decision values.

# # Model 2 : Random Forest

# In[86]:


from sklearn.ensemble import RandomForestClassifier


# In[87]:


#Fit the model
RF = RandomForestClassifier(n_estimators = 400).fit(train_X,train_y)


# In[88]:


#Predicting the test cases
bankloans_test_pred_RFlog = pd.DataFrame({'actual':test_y, 'predicted': RF.predict(test_X)}) #here we predict
bankloans_test_pred_RFlog = bankloans_test_pred_RFlog.reset_index()
bankloans_test_pred_RFlog.head()


# In[89]:


#creating a confusion matrix

cm_RF = metrics.confusion_matrix(bankloans_test_pred_RFlog.actual,
                                    bankloans_test_pred_RFlog.predicted,labels = [1,0])
cm_RF


# In[90]:


#Visualize the confusion matrix
sns.heatmap(cm_RF,annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels = ["Default", "Not Default"] , yticklabels = ["Default", "Not Default"])
plt.title("Confusion Matrix for Test data")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


# In[91]:


#find precision score

prec_score = metrics.precision_score(bankloans_test_pred_RFlog.actual, bankloans_test_pred_RFlog.predicted)
print("Precision score :", round(prec_score,3))


# In[92]:


#intuitively the ability of the classifier to find all the positive samples

recall_score = metrics.recall_score(bankloans_test_pred_RFlog.actual, bankloans_test_pred_RFlog.predicted)
print("recall_score:", round(recall_score , 3))


# In[93]:


#find the overall accuracy of model

acc_score = metrics.accuracy_score(bankloans_test_pred_RFlog.actual,bankloans_test_pred_RFlog.predicted)
print("Accuracy of model :", round(acc_score,3))


# In[94]:


#probabilty of prediction
#logreg.predict_proba() Function used
# here 0.789495… is the probability that the output will be 0 and 0.210505… is the probability of output being 1

predict_prob_RF = pd.DataFrame(RF.predict_proba(test_X))
predict_prob_RF.head()


# In[95]:


bankloans_test_pred_RFlog.actual.value_counts()


# In[96]:


predict_prob_RF.head()


# In[97]:


predict_prob_df.head(1)


# In[98]:


bankloans_test_pred_RFlog = pd.concat([bankloans_test_pred_RFlog, predict_prob_RF], axis = 1)
bankloans_test_pred_RFlog.columns = ['index', 'actual', 'predicted', 'default_0','default_1']

bankloans_test_pred_RFlog.head()


# In[99]:


#Predicting with new cut-off probability
bankloans_test_pred_RFlog['new_labels'] = bankloans_test_pred_RFlog['default_1'].map( lambda x: 1 if x >= 0.224326 else 0 )

bankloans_test_pred_RFlog.head()


# In[100]:


#Again Test the model


# In[101]:


#creating a confusion matrix

cm_RF = metrics.confusion_matrix(bankloans_test_pred_RFlog.actual,
                                    bankloans_test_pred_RFlog.new_labels,labels = [1,0])
cm_RF


# In[102]:


#Visualize the confusion matrix
sns.heatmap(cm_RF,annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels = ["Default", "Not Default"] , yticklabels = ["Default", "Not Default"])
plt.title("Confusion Matrix for Test data")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


# In[103]:


#find precision score

prec_score = metrics.precision_score(bankloans_test_pred_RFlog.actual, bankloans_test_pred_RFlog.new_labels)
print("Precision score :", round(prec_score,3))


# In[104]:


#find the overall accuracy of model

acc_score = metrics.accuracy_score(bankloans_test_pred_RFlog.actual,bankloans_test_pred_RFlog.new_labels)
print("Accuracy of model :", round(acc_score,3))


# In[105]:


#intuitively the ability of the classifier to find all the positive samples

recall_score = metrics.recall_score(bankloans_test_pred_RFlog.actual, bankloans_test_pred_RFlog.new_labels)
print("recall_score:", round(recall_score , 3))


# In[106]:


bankloans_test_pred_RFlog.actual.value_counts()


# In[107]:


#classification report 
print("Classification Report for Random Forest")
print(metrics.classification_report(bankloans_test_pred_RFlog.actual,bankloans_test_pred_RFlog.new_labels))


# # Model 3 :  Decision Tree Classifier

# In[108]:


from sklearn.tree import DecisionTreeClassifier


# In[109]:


#Fit the model
DT = DecisionTreeClassifier().fit(train_X,train_y)


# In[110]:


#Predicting the test cases
bankloans_test_pred_DTlog = pd.DataFrame({'actual':test_y, 'predicted': DT.predict(test_X)}) #here we predict
bankloans_test_pred_DTlog = bankloans_test_pred_DTlog.reset_index()
bankloans_test_pred_DTlog.head()


# In[111]:


#creating a confusion matrix

cm_DT = metrics.confusion_matrix(bankloans_test_pred_DTlog.actual,
                                    bankloans_test_pred_DTlog.predicted,labels = [1,0])
cm_DT


# In[112]:


#Visualize the confusion matrix
sns.heatmap(cm_DT,annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels = ["Default", "Not Default"] , yticklabels = ["Default", "Not Default"])
plt.title("Confusion Matrix for Test data")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


# In[114]:


#find precision score

prec_score = metrics.precision_score(bankloans_test_pred_RFlog.actual, bankloans_test_pred_RFlog.predicted)
print("Precision score :", round(prec_score,3))


# In[115]:


#intuitively the ability of the classifier to find all the positive samples

recall_score = metrics.recall_score(bankloans_test_pred_RFlog.actual, bankloans_test_pred_RFlog.predicted)
print("recall_score:", round(recall_score , 3))


# In[116]:


#intuitively the ability of the classifier to find all the positive samples

recall_score = metrics.recall_score(bankloans_test_pred_RFlog.actual, bankloans_test_pred_RFlog.predicted)
print("recall_score:", round(recall_score , 3))


# In[117]:


#find the overall accuracy of model

acc_score = metrics.accuracy_score(bankloans_test_pred_RFlog.actual,bankloans_test_pred_RFlog.predicted)
print("Accuracy of model :", round(acc_score,3))


# In[118]:


#probabilty of prediction
#logreg.predict_proba() Function used
# here 0.789495… is the probability that the output will be 0 and 0.210505… is the probability of output being 1

predict_prob_DT = pd.DataFrame(RF.predict_proba(test_X))
predict_prob_DT.head()


# In[119]:


bankloans_test_pred_DTlog = pd.concat([bankloans_test_pred_DTlog, predict_prob_RF], axis = 1)
bankloans_test_pred_DTlog.columns = ['index', 'actual', 'predicted', 'default_0','default_1']

bankloans_test_pred_DTlog.head()


# In[120]:


#Predicting with new cut-off probability
bankloans_test_pred_DTlog['new_labels'] = bankloans_test_pred_DTlog['default_1'].map( lambda x: 1 if x >= 0.224326 else 0 )

bankloans_test_pred_DTlog.head()


# In[121]:


#Visualize the confusion matrix
sns.heatmap(cm_DT,annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels = ["Default", "Not Default"] , yticklabels = ["Default", "Not Default"])
plt.title("Confusion Matrix for Test data")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


# In[122]:


#find precision score

prec_score = metrics.precision_score(bankloans_test_pred_DTlog.actual, bankloans_test_pred_DTlog.new_labels)
print("Precision score :", round(prec_score,3))


# In[124]:


#find the overall accuracy of model

acc_score = metrics.accuracy_score(bankloans_test_pred_DTlog.actual,bankloans_test_pred_DTlog.new_labels)
print("Accuracy of model :", round(acc_score,3))


# In[123]:


#intuitively the ability of the classifier to find all the positive samples

recall_score = metrics.recall_score(bankloans_test_pred_DTlog.actual, bankloans_test_pred_DTlog.new_labels)
print("recall_score:", round(recall_score , 3))


# In[125]:


#classification report 
print("Classification Report for Random Forest")
print(metrics.classification_report(bankloans_test_pred_DTlog.actual,bankloans_test_pred_DTlog.new_labels))


# In[ ]:





# Model Selection and Business Insights
# 
# - Based on the F1-score (harmonic mean of precision and recall), logistic model with f1 score (for positive labels - default customers) of 0.66 is giving better results than decision tree model with f1 score of 0.44 And Better than Random Forest 0.63. -So we will use the logistic regression model to predict the credit worthiness of the customers 
# # We will Predict the credit risk for remainimg 150 customers using the logistic model with cutoff as 0.224 

# In[153]:


#probability for new customers
print("probability for new customers")
new_cust_prob = pd.DataFrame(logreg.predict_proba(bankloans_new[featurecolumns]))
new_cust_prob.columns = ["prob_default_0", "prob_default_1"]
new_cust_prob.index = bankloans_new.index


# In[126]:


new_cust_prob.head()


# In[127]:


bankloans_new_predicted = pd.concat([bankloans_new,new_cust_prob],axis=1)
bankloans_new_predicted.head()


# In[128]:


#using the cutoff value we will predict the default

bankloans_new_predicted['predicted_default'] = bankloans_new_predicted["prob_default_1"].apply(lambda x: 1 if x > 0.224 else 0)


# In[129]:


bankloans_new_predicted.head()


# In[130]:


#Model Prediction For the New customers (150)

bankloans_new_predicted.predicted_default.value_counts()


# In[131]:


#Model Prediction

bankloans_new_predicted.predicted_default.value_counts().plot.bar()

plt.ylabel("Count")
plt.xlabel("Default")
plt.show()


# In[132]:


bankloans_new_predicted.to_csv('Predictions For New.csv')


# Model Selection and Business Insights
# 
# - Based on the F1-score (harmonic mean of precision and recall), logistic model with f1 score (for positive labels - default customers) of 0.66 is giving better results than decision tree model with f1 score of 0.44 And Better than Random Forest 0.63. -So we will use the logistic regression model to predict the credit worthiness of the customers 
# -We will Predict the credit risk for remainimg 150 customers using the logistic model with cutoff as 0.224 

# ## Finally, let's save the winning model.
# - We need to save your prediction models to file, and then restore them in order to reuse your previous work to: test your model on new data, compare multiple models, or anything else.
# - Pickle is the standard way of serializing objects in Python.Pickle operation to serialize your machine learning algorithms and save the serialized format to a file.
# - Later you can load this file to deserialize your model and use it to make new predictions.

# In[133]:


import pickle


# Let's save the winning <code style="color:steelblue">Logistic Model</code> object into a pickle file.

# In[134]:


import joblib


# In[135]:


#Save the Model
joblib.dump(RF,"C:/Users/Abhishek/Desktop/Bank_Loan_Edwisor/Bank_Loan_Python/Logreg_Model.pkl")


# In[154]:


#Load the model from the file 
Logreg_Model = joblib.load('Logreg_Model.pkl')
print("Done")


# In[137]:


#Loaded model then can be used some time later...
 


# In[ ]:





# In[ ]:





# In[ ]:




