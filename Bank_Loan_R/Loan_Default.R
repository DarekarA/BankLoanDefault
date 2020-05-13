rm(list=ls())

setwd("C:/Users/Abhishek/Desktop/Bank_Loan_Edwisor/Bank_Loan_R")
getwd()

#install.packages("doSNOW")
#install.packages("rpart.plot")

#1. Set Environment and package load

x = c("ggplot2", "corrgram", "DMwR", "usdm", "caret", "randomForest", "e1071",
      "DataCombine", "doSNOW", "inTrees", "rpart.plot", "rpart",'MASS','xgboost','stats','C50')

#load Packages
lapply(x, require, character.only = TRUE)

rm(x)


#2. Load dataset and Data Pre Processing

bankloans=read.csv("bank-loan.csv")

str(bankloans)
summary(bankloans)

#3. Checking for missing values
missing_val=data.frame(apply(bankloans,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val) #new column "Columns" has values as all rows in the index
names(missing_val)[1]="Missing_Percentage" #Rename 1st row name
missing_val$Missing_Percentage = (missing_val$Missing_Percentage/nrow(bankloans)) * 100 #Convert data in missing percentage columns to actual percentages.
missing_val = missing_val[order(-missing_val$Missing_Percentage),] #Order percentage in descending
row.names(missing_val)=NULL
missing_val = missing_val[,c(2,1)] #interchange missing percentage and column position
# Only "default"	Column has missing values


#4.1 Separate the numeric and categorical variable names 
numeric_index = sapply(bankloans,is.numeric)
numeric_data = bankloans[,numeric_index]
numeric_data

cnames = colnames(numeric_data)
cnames

#4.2 splitting the data set into two sets - existing customers and new customers

#bankloans_existing = bankloans.loc[bankloans.default.isnull() == 0] #isnulll is false

bankloans_existing = bankloans[which(bankloans$default>=0),]
bankloans_new = bankloans[-which(bankloans$default>=0),]

unique(bankloans$default)


#5.Checking for Outliers

pl1 = ggplot(bankloans_existing,aes(y = age))
pl1 +stat_boxplot(geom = "errorbar", width=0.5) + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

pl1 = ggplot(bankloans_existing,aes(y = employ))
pl1 +stat_boxplot(geom = "errorbar", width=0.5) + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

pl1 = ggplot(bankloans_existing,aes(y = income))
pl1 +stat_boxplot(geom = "errorbar", width=0.5) + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

pl1 = ggplot(bankloans_existing,aes(y = debtinc))
pl1 +stat_boxplot(geom = "errorbar", width=0.5) + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

pl1 = ggplot(bankloans_existing,aes(y = creddebt))
pl1 +stat_boxplot(geom = "errorbar", width=0.5) + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

#5.1 Outlier Treatment

#bankloans_existing$default.quantile(0.95)
#quantile(bankloans_existing$age,probs=c(0.75))
#quantile(bankloans_existing$age)

for(i in cnames){
vals = bankloans_existing[,i] %in% boxplot.stats(bankloans_existing[,i])$out #PUTS ALL OUTLIERS IN VALS VARIABLE
bankloans_existing[which(vals),i] = NA  #SET ALL OUTLIERS(Present in VALs) TO NA AND THEN IMPUTE
}

#lets check the NA's
data.frame(apply(bankloans_existing,2,function(x){sum(is.na(x))}))

#Imputing with KNN
bankloans_existing = knnImputation(bankloans_existing,k=3)

# lets check the missing values
data.frame(apply(bankloans_existing,2,function(x){sum(is.na(x))}))

#All the missing values have been imputed

#6 Correlation


library(corrplot)

bankloans_existing.cor = cor(bankloans_existing)
#corrplot(bankloans_existing.cor)
palette = colorRampPalette(c("green", "white", "red")) (20)
heatmap(x = bankloans_existing.cor, col = palette, symm = TRUE)


#Indicator variable unique types
table(bankloans_existing$default)
#We can see we have 517 0's and 183 1's


 #Multi Collinearity Check
  library(usdm)
  vif(bankloans_existing[,-1])
 
  vifcor(bankloans_existing[,-1], th = 0.9)

  #Backup
  #vm=bankloans_existing
  #bankloans_existing=vm
#Split the data into Train and test
  
  #### Trial 
  #bankloans_existing$default <- as.factor((bankloans_existing$default))
  #unique(bankloans_existing$default)
  ####
  
  set.seed(1000) #Random.seed is an integer vector, containing the random number generator (RNG) state for random number generation in R. It can be saved and restored, but should not be altered by the user.
  library(caret)
  library(Rcpp)
  tr.idx = createDataPartition(bankloans_existing$default,p=0.80,list = FALSE) # 80% in trainin and 20% in Validation Datasets
  train_bankloans_existing = bankloans_existing[tr.idx,] #new variable train_data which will have 80% of data  560
  test_bankloans_existing = bankloans_existing[-tr.idx,] #new variable test_data which will have 20% of data   140
  table(train_bankloans_existing$default) #total 560

#**********************************************************************************************************************************************************  
### Logistic Regression    
  #library(rpart)
  ##rpart for regression
  
  ## Model Building
  fit = glm(default ~ ., data = train_bankloans_existing,family = "binomial")  

  #Features and their coefficients
  mylogit <- glm(default ~ address + age + creddebt + debtinc + ed +employ + income + othdebt, data = train_bankloans_existing) 
  summary(mylogit)
  
  #Predict for new test cases
  #bankloans_test_pred_log = predict(fit, test_bankloans_existing[,-9])
  bankloans_test_pred_log = predict(fit, newdata = test_bankloans_existing[,-9],type="response")
  #bankloans_test_pred_log has the probability for each cell to be 0 or 1, Now convert it into 1 and 0.
 
   #Convert prob into 1 and 0
  bankloans_test_pred_log= ifelse(bankloans_test_pred_log>0.224,1,0) #if >0.5 then 1 else 0(i.e less thn 0.5)
  
  bankloans_test_pred_log
  range(bankloans_test_pred_log)
  unique(bankloans_test_pred_log)
  length(bankloans_test_pred_log)
########### To Predict and create confusion matrix :      #########################  
  #ConfusionMatrix
  cm=table(test_bankloans_existing$default,bankloans_test_pred_log)# 
  cm
  
  
  #Confusion matrix variables
  tp=cm[2,2];tn=cm[1,1];fp=cm[1,2];fn=cm[2,1];total = tp + tn + fp +fn
  
  #find precision score : tp/(tp+fp)
  ps = tp/(tp+fp) # Rejected this : cm[1,1]/sum(cm[1,1:2]) #0.91
  print("Precision for Logistic regression :")
  ps
  
  # Recall: tp/(tp + fn):
  recall =tp/(tp + fn) #Rejected this : cm[1,1]/sum(cm[1:2,1]) 0.86
  print("Recall for Logistic regression :")
  recall
  
  # F-Score: 2 * precision * recall /(precision + recall):
  FScore=  2 * ps * recall /(ps + recall) #0.891
  print("Fscore for Logistic regression :")
  FScore
  
  #find the overall accuracy of model : (TP+TN)/total
  acc = (tp+tn)/total
  print("Accuracy for Logistic regression :")
  acc
#**********************************************************************************************************************************************************
 
  ### Random Forest  
  
  
  ## Model Building
  RF = randomForest(default ~ ., data = train_bankloans_existing,importance=TRUE,ntree=500)  
  
  #Features and their coefficients
  mylogitRF <- glm(default ~ address + age + creddebt + debtinc + ed +employ + income + othdebt, data = train_bankloans_existing) 
  summary(mylogitRF)
  
  #Predict for new test cases
  #bankloans_test_pred_log = predict(fit, test_bankloans_existing[,-9])
  bankloans_test_pred_logRF = predict(RF, newdata = test_bankloans_existing[,-9],type="response")
  #bankloans_test_pred_log has the probability for each cell to be 0 or 1, Now convert it into 1 and 0.
  bankloans_test_pred_logRF
  
  #Convert prob into 1 and 0
  bankloans_test_pred_logRF= ifelse(bankloans_test_pred_logRF>0.224,1,0) #if >0.5 then 1 else 0(i.e less thn 0.5)
  
  
  cmRF=table(test_bankloans_existing$default,bankloans_test_pred_logRF)# USE RF HERE MAYBE
  cmRF
  
  #Confusion matrix variables
  tpRF=cmRF[2,2];tnRF=cmRF[1,1];fpRF=cmRF[1,2];fnRF=cmRF[2,1];totalRF = tpRF + tnRF + fpRF +fnRF
  
  #precision score : tp/(tp+fp)
  psRF = tpRF/(tpRF+fpRF) 
  print("Precision for Random Forest :")
  psRF
  
  # Recall
  recallRF =tpRF/(tpRF + fnRF) 
  print("Recall for Random Forest :")
  recallRF
  
  # F-Score: 2 * precision * recall /(precision + recall):
  FScoreRF=  2 * psRF * recallRF /(psRF + recallRF) #
  print("Fscore for Random Forest :")
  FScoreRF
  
  #find the overall accuracy of model : (TP+TN)/total
  accRF = (tpRF+tnRF)/totalRF
  print("Accuracy for Random Forest :")
  accRF
  
  
#**********************************************************************************************************************************************************
  #Decision Tree
  
  DT = rpart(default ~ ., data = train_bankloans_existing, method = "class")  
  
  #Features and their coefficients
  mylogitDT <- glm(default ~ address + age + creddebt + debtinc + ed +employ + income + othdebt, data = train_bankloans_existing) 
  summary(mylogitDT)
  
  #Predict for new test cases
  bankloans_test_pred_logDT = predict(DT, newdata = test_bankloans_existing[,-9], type = "class")
  #bankloans_test_pred_log has the probability for each cell to be 0 or 1, Now convert it into 1 and 0.
  bankloans_test_pred_logDT
  #Convert prob into 1 and 0
  #bankloans_test_pred_logDT= ifelse(bankloans_test_pred_logDT>0.224,1,0) #if >0.5 then 1 else 0(i.e less thn 0.5)
  bankloans_test_pred_logDT
  
  cmDT=table(test_bankloans_existing$default,bankloans_test_pred_logDT)# 
  cmDT
  
  #Confusion matrix variables
  tpDT=cmDT[2,2];tnDT=cmDT[1,1];fpDT=cmDT[1,2];fnDT=cmDT[2,1];totalDT = tpDT + tnDT + fpDT +fnDT
  
  #precision score : tp/(tp+fp)
  psDT = tpDT/(tpDT+fpDT) 
  psDT
  
  # Recall
  recallDT = tpDT/(tpDT + fnDT) 
  recallDT
  
  # F-Score: 2 * precision * recall /(precision + recall):
  FScoreDT=  2 * psDT * recallDT /(psDT + recallDT) #
  FScoreDT
  
  #find the overall accuracy of model : (TP+TN)/total
  accDT = (tpDT+tnDT)/totalDT
  accDT
  
#**********************************************************************************************************************************************************
  
  #MODEL Evaluation
  # Logistic regression 1.precision Score :0.36   2.Recall :0.64    3. F-Score :0.46     4.Accuracy : 0.70
  
  #Random Forest        1.precision Score : 0.39  2.Recall : 0.78   3. F-Score :0.52    4.Accuracy : 0.71
   
  #Decision Tree        1.precision Score : 0.37  2.Recall :0.39    3. F-Score :0.38   4.Accuracy :0.75
  
  #Model Selection and Business Insights
        #Based on the F1-score (harmonic mean of precision and recall), Random Forest model with f1 score (for positive labels - default customers) of :0.52  is giving better results than Logistic Regression model with f1 score of 0.46 And Better than Decision Tree 0.38. -So we will use the Ransom Forest model to predict the credit worthiness of the customers 
         #We will Predict the credit risk for remainimg 150 customers using the logistic model with cutoff as 0.224 
 
#**********************************************************************************************************************************************************
  bankloans_new_matrix2 = as.matrix(sapply(bankloans_new,as.numeric))
  #bankloans_new$Predicted_default = bankloans_test_pred_logRF
  
  # Saving the trained model
  saveRDS(RF, "./final_prediction_using_RF.rds")
  
  # loading the saved model
  super_model <- readRDS("./final_prediction_using_RF.rds")
  print(super_model)
  
  # Lets now predict on test dataset
  predict_default = predict(super_model,bankloans_new_matrix2)
  
  #
  bankloans_new$predicted_default = predict_default
  bankloans_new$predicted_default= ifelse(bankloans_new$predicted_default>0.224,1,0)
  bankloans_new$predicted_default
  
  # Now lets write(save) the predicted fare_amount in disk as .csv format 
  write.csv(bankloans_new,"Predicted_default.csv",row.names = FALSE)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
   