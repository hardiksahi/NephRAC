import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import warnings
from scipy import interp
import matplotlib.pyplot as plt
import math


warnings.filterwarnings("ignore")


healthData = pd.read_csv('/Users/hardiksahi/Desktop/Final Health Data/FinalDataML.csv')

print(healthData.shape)

# Keeping only the desired variables.
# Assigning NuLL in Kidneydisease a category of 12
obj1List = ['age', 'sex', 'totalDuration', 'Kidneydisease', 'diabetes', 'Heartdisease', 'Kidneytumor','Mentalhealth','vasc_accdnt','vasc_disease','Combinelung','Albumin','Hemoglobin', 'Creatinine','Urea','Potassium','phosphorus']
subsetDataObj1 = healthData[obj1List]

#print(subsetDataObj1.shape)
#print(subsetDataObj1.columns)
#print(subsetDataObj1.isnull().sum())
#subsetDataObj1.fillna(12.0)
#df.replace(-999, np.nan)


#print(subsetDataObj1.isnull().sum())
#print(subsetDataObj1['Kidneydisease'])

male_subset = subsetDataObj1[subsetDataObj1['sex'] == 'm']
female_subset = subsetDataObj1[subsetDataObj1['sex'] == 'f']

# COmpute means of 'Albumin','Hemoglobin', 'Creatinine','Urea','Potassium','phosphorus'

mean_m_albumin  = math.floor(np.mean(male_subset['Albumin'].dropna()))
print("mean_m_albumin" , mean_m_albumin)

mean_f_albumin  = math.floor(np.mean(female_subset['Albumin'].dropna()))
print("mean_f_albumin" , mean_f_albumin)

mean_m_hb  = math.floor(np.mean(male_subset['Hemoglobin'].dropna()))
print("mean_m_hb" , mean_m_hb)

mean_f_hb  = math.floor(np.mean(female_subset['Hemoglobin'].dropna()))
print("mean_f_hb" , mean_f_hb)

mean_m_Creatinine  = math.floor(np.mean(male_subset['Creatinine'].dropna()))
print("mean_m_Creatinine" , mean_m_Creatinine)

mean_f_Creatinine  = math.floor(np.mean(female_subset['Creatinine'].dropna()))
print("mean_f_Creatinine" , mean_f_Creatinine)

mean_m_Urea  = round(np.mean(male_subset['Urea'].dropna()),2)
print("mean_m_Urea" , mean_m_Urea)

mean_f_Urea  = round(np.mean(female_subset['Urea'].dropna()),2)
print("mean_f_Urea" , mean_f_Urea)

mean_m_Potassium  = round(np.mean(male_subset['Potassium'].dropna()),2)
print("mean_m_Potassium" , mean_m_Potassium)

mean_f_Potassium  = round(np.mean(female_subset['Potassium'].dropna()),2)
print("mean_f_Potassium" , mean_f_Potassium)

mean_m_phosphorus  = round(np.mean(male_subset['phosphorus'].dropna()),2)
print("mean_m_phosphorus" , mean_m_phosphorus)

mean_f_phosphorus  = round(np.mean(female_subset['phosphorus'].dropna()),2)
print("mean_f_phosphorus" , mean_f_phosphorus)

subsetDataObj1.loc[(subsetDataObj1['sex'] == 'm') & pd.isnull(subsetDataObj1['Albumin']),'Albumin'] = mean_m_albumin
subsetDataObj1.loc[(subsetDataObj1['sex'] == 'f') & pd.isnull(subsetDataObj1['Albumin']),'Albumin'] = mean_f_albumin

subsetDataObj1.loc[(subsetDataObj1['sex'] == 'm') & pd.isnull(subsetDataObj1['Hemoglobin']),'Hemoglobin'] = mean_m_hb
subsetDataObj1.loc[(subsetDataObj1['sex'] == 'f') & pd.isnull(subsetDataObj1['Hemoglobin']),'Hemoglobin'] = mean_f_hb

subsetDataObj1.loc[(subsetDataObj1['sex'] == 'm') & pd.isnull(subsetDataObj1['Creatinine']),'Creatinine'] = mean_m_Creatinine
subsetDataObj1.loc[(subsetDataObj1['sex'] == 'f') & pd.isnull(subsetDataObj1['Creatinine']),'Creatinine'] = mean_f_Creatinine

subsetDataObj1.loc[(subsetDataObj1['sex'] == 'm') & pd.isnull(subsetDataObj1['Urea']),'Urea'] = mean_m_Urea
subsetDataObj1.loc[(subsetDataObj1['sex'] == 'f') & pd.isnull(subsetDataObj1['Urea']),'Urea'] = mean_f_Urea

subsetDataObj1.loc[(subsetDataObj1['sex'] == 'm') & pd.isnull(subsetDataObj1['Potassium']),'Potassium'] = mean_m_Potassium
subsetDataObj1.loc[(subsetDataObj1['sex'] == 'f') & pd.isnull(subsetDataObj1['Potassium']),'Potassium'] = mean_f_Potassium

subsetDataObj1.loc[(subsetDataObj1['sex'] == 'm') & pd.isnull(subsetDataObj1['phosphorus']),'phosphorus'] = mean_m_phosphorus
subsetDataObj1.loc[(subsetDataObj1['sex'] == 'f') & pd.isnull(subsetDataObj1['phosphorus']),'phosphorus'] = mean_f_phosphorus



#subsetDataObj1.to_csv('/Users/hardiksahi/Desktop/Final Health Data/withImputation.csv')

#print("After imp", subsetDataObj1.isnull().sum())

#subsetDataObj1 = 

# Concatenating all the columns together
#subsetDataObj1 = pd.concat([subsetDataObj1, pd.DataFrame(imputedDataSet, columns = columnsToImpute)], axis=1)
#print("After imputaation",subsetDataObj1.isnull().sum())
#print("Size of final subset", subsetDataObj1.shape)


listToConvertCat = ['sex','Kidneydisease', 'diabetes', 'Heartdisease', 'Kidneytumor','Mentalhealth','vasc_accdnt','vasc_disease','Combinelung']
for col in listToConvertCat:
    subsetDataObj1[col] = subsetDataObj1[col].astype('category')



print("Before encoding.." , subsetDataObj1.shape)
# Convert all categorical to one hot encoding

#pd.get_dummies(subsetDataObj1, columns = listToConvertCat)

X = pd.get_dummies(subsetDataObj1)

print("After encoding.." , X.shape)

#print(X.columns)

Y = healthData['Withdrawal']
#withdrawalV = healthData['Withdrawal']

#Y = withdrawalV.reshape(withdrawalV.shape[0],1)

print("shape of train X", X.shape)
print("shape of train Y", Y.shape)


from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(penalty="l1", C = 1)

#Training the model
fitLog = logisticRegr.fit(X, Y)

for i in range(X.shape[1]):
    print(X.columns[i], ":", fitLog.coef_[0][i])
    
       

#Cross validation to calcuate how good the model is?
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

sm = SMOTE(random_state=12, ratio = 1)
kf = KFold(n_splits=10)
sumV = 0
aucLog = 0
sensLog = 0
specLog = 0

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train_index, test_index in kf.split(X):
    #print("Training index", train_index)
    #print("Testing index", test_index)
    X_train, Y_train = X.as_matrix()[train_index], Y[train_index]
    X_test, Y_test = X.as_matrix()[test_index], Y[test_index]
    
    X_train_smote, Y_train_smote = sm.fit_sample(X_train,Y_train)
    
    logisticRegr.fit(X_train_smote, Y_train_smote)
    y_pred = logisticRegr.predict(X_test)
    y_pred_prob = logisticRegr.predict_proba(X_test)
    sumV+=accuracy_score(Y_test,y_pred)
    
    fpr, tpr, threshold = roc_curve(Y_test, y_pred_prob[:,1], pos_label = 1)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    
    
    vv = auc(fpr, tpr)
    
    aucs.append(vv)
    
    sens = recall_score(Y_test, y_pred, pos_label=1, average="binary")
    spec = recall_score(Y_test, y_pred, pos_label=0, average="binary")
    #print(vv)
    aucLog += vv
    sensLog+=sens
    specLog+=spec
    
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, vv))
    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)    

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Logistic Regression')
plt.legend(loc="lower right")
plt.show()

averageAcc = sumV/10
averageAuc = aucLog/10
averageSensLog = sensLog/10
averageSpecLog = specLog/10

print("Accuracy of the model_smote [LR] is" ,averageAcc*100)
print("AUC of the model_smote[LR] is" ,averageAuc)
print("Sensitivity of the model_smote[LR]", averageSensLog)
print("Specificity of the model_smote[LR]", averageSpecLog)









#RandomForest implementation...........................
from sklearn.ensemble import RandomForestClassifier
rFClassifier = RandomForestClassifier(n_estimators = 20, criterion = 'gini', max_features="sqrt", random_state=1, oob_score = True)
fitRF = rFClassifier.fit(X,Y)


sumVRF = 0
aucRF = 0
sensRF = 0
specRF = 0
for train_index, test_index in kf.split(X):
    X_train, Y_train = X.as_matrix()[train_index], Y[train_index]
    X_test, Y_test = X.as_matrix()[test_index], Y[test_index]
    
    X_train_smote, Y_train_smote = sm.fit_sample(X_train,Y_train)
    
    rFClassifier.fit(X_train_smote, Y_train_smote)
    y_pred = rFClassifier.predict(X_test)
    y_pred_prob = rFClassifier.predict_proba(X_test)
    sumVRF+=accuracy_score(Y_test,y_pred)
    fpr, tpr, threshold = roc_curve(Y_test, y_pred_prob[:,1], pos_label = 1)
    vv = auc(fpr, tpr)
    #print(vv)
    aucRF += vv
    sens = recall_score(Y_test, y_pred, pos_label=1, average="binary")
    spec = recall_score(Y_test, y_pred, pos_label=0, average="binary")
    sensRF+=sens
    specRF+=spec
    
    
    
    
averageAccRF = sumVRF/10
averageAucRF = aucRF/10
averageSensRF = sensRF/10
averageSpecRF = specRF/10

print("Accuracy of the model_smote[RF] is" ,averageAccRF*100)
print("AUC of the model_smote[RF] is" ,averageAucRF)
print("Sensitivity of the model_smote[RF]", averageSensRF)
print("Specificity of the model_smote[RF]", averageSpecRF)

#Accuracy of the model[RF] is 87.5608828006
#AUC of the model[RF] is 0.595964104428



## SVM...................
from sklearn.svm import SVC
sVClassifier = SVC(C = 1, kernel='rbf', random_state = 1)

fitSVC =  sVClassifier.fit(X,Y)

sumVSVC = 0
aucSVC = 0
sensSVC = 0
specSVC = 0

for train_index, test_index in kf.split(X):
    X_train, Y_train = X.as_matrix()[train_index], Y[train_index]
    X_test, Y_test = X.as_matrix()[test_index], Y[test_index]
    
    X_train_smote, Y_train_smote = sm.fit_sample(X_train,Y_train)
    
    sVClassifier.fit(X_train_smote, Y_train_smote)
    y_pred = sVClassifier.predict(X_test)
    #y_pred_prob = rFClassifier.predict_proba(X_test)
    sumVSVC+=accuracy_score(Y_test,y_pred)
    fpr, tpr, threshold = roc_curve(Y_test, sVClassifier.decision_function(X_test), pos_label = 1)
    vv = auc(fpr, tpr)
    #print(vv)
    aucSVC += vv
    sens = recall_score(Y_test, y_pred, pos_label=1, average="binary")
    spec = recall_score(Y_test, y_pred, pos_label=0, average="binary")
    sensSVC+=sens
    specSVC+=spec
    
    
    
averageAccSVC = sumVSVC/10
averageAucSVC = aucSVC/10
averageSensSVC = sensSVC/10
averageSpecSVC = specSVC/10

print("Accuracy of the model_smote[SVC] is" ,averageAccSVC*100)
print("AUC of the model_smote[SVC] is" ,averageAucSVC)
print("Sensitivity of the model_smote[SVC]", averageSensSVC)
print("Specificity of the model_smote[SVC]", averageSpecSVC)


## AdaBoost..........
from sklearn.ensemble import AdaBoostClassifier
adaBoostClassifier = AdaBoostClassifier(n_estimators=40, learning_rate=1.25, random_state = 1)

fitAda = adaBoostClassifier.fit(X,Y)

#for i in range(X.shape[1]):
#    print(X.columns[i], ":", fitAda.feature_importances_[i])


sumVAda = 0
aucAda = 0
sensAda = 0
specAda = 0

for train_index, test_index in kf.split(X):
    X_train, Y_train = X.as_matrix()[train_index], Y[train_index]
    X_test, Y_test = X.as_matrix()[test_index], Y[test_index]
    
    X_train_smote, Y_train_smote = sm.fit_sample(X_train,Y_train)
    
    adaBoostClassifier.fit(X_train_smote, Y_train_smote)
    y_pred = adaBoostClassifier.predict(X_test)
    y_pred_prob = adaBoostClassifier.predict_proba(X_test)
    sumVAda+=accuracy_score(Y_test,y_pred)
    fpr, tpr, threshold = roc_curve(Y_test, y_pred_prob[:,1], pos_label = 1)
    vv = auc(fpr, tpr)
    #print(vv)
    aucAda += vv
    sens = recall_score(Y_test, y_pred, pos_label=1, average="binary")
    spec = recall_score(Y_test, y_pred, pos_label=0, average="binary")
    sensAda+=sens
    specAda+=spec
    
    
    
    
averageAccAda = sumVAda/10
averageAucAda = aucAda/10
averageSensAda = sensAda/10
averageSpecAda = specAda/10

print("Accuracy of the model_smote[Ada] is" ,averageAccAda*100)
print("AUC of the model_smote[Ada] is" ,averageAucAda)
print("Sensitivity of the model_smote[Ada]", averageSensAda)
print("Specificity of the model_smote[Ada]", averageSpecAda)







