# NephRAC
This is a project to determine whether or not a patient will withdraw from dialysis. The dataset that was used was takedn from a Dialysis centre in Waterloo, Canada. So, it cannot be shared because of confidential information.

The dataset was highly unbalanced. Hence I used SMOTE technique to upsample the minority class as can be seen in the code.
Various classification techniques that are used are:

1. Random Forest
2. AdaBoost
3. Logistic Regression
4. Support Vector Machine.

Logistic Regression with L1 penalty gave the best sensitivity of 63% and AUROC of 0.67.
