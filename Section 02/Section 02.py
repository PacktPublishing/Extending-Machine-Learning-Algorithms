
# Adaboost Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
dtree = DecisionTreeClassifier(criterion='gini',max_depth=1)

adabst_fit = AdaBoostClassifier(base_estimator= dtree,
        n_estimators=5000,learning_rate=0.05,random_state=42)

adabst_fit.fit(x_train, y_train)

print ("\nAdaBoost - Train Confusion Matrix\n\n",pd.crosstab(y_train,adabst_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nAdaBoost  - Train accuracy",round(accuracy_score(y_train,adabst_fit.predict(x_train)),3))
print ("\nAdaBoost  - Train Classification Report\n",classification_report(y_train,adabst_fit.predict(x_train)))

print ("\n\nAdaBoost  - Test Confusion Matrix\n\n",pd.crosstab(y_test,adabst_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nAdaBoost  - Test accuracy",round(accuracy_score(y_test,adabst_fit.predict(x_test)),3))
print ("\nAdaBoost - Test Classification Report\n",classification_report(y_test,adabst_fit.predict(x_test)))

# Gradientboost Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbc_fit = GradientBoostingClassifier(loss='deviance',learning_rate=0.05,n_estimators=5000,
                                     min_samples_split=2,min_samples_leaf=1,max_depth=1,random_state=42 )
gbc_fit.fit(x_train,y_train)

print ("\nGradient Boost - Train Confusion Matrix\n\n",pd.crosstab(y_train,gbc_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nGradient Boost - Train accuracy",round(accuracy_score(y_train,gbc_fit.predict(x_train)),3))
print ("\nGradient Boost  - Train Classification Report\n",classification_report(y_train,gbc_fit.predict(x_train)))

print ("\n\nGradient Boost - Test Confusion Matrix\n\n",pd.crosstab(y_test,gbc_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nGradient Boost - Test accuracy",round(accuracy_score(y_test,gbc_fit.predict(x_test)),3))
print ("\nGradient Boost - Test Classification Report\n",classification_report(y_test,gbc_fit.predict(x_test)))

  
# Xgboost Classifier
import xgboost as xgb

xgb_fit = xgb.XGBClassifier(max_depth=2, n_estimators=5000, learning_rate=0.05)
xgb_fit.fit(x_train, y_train)

print ("\nXGBoost - Train Confusion Matrix\n\n",pd.crosstab(y_train,xgb_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nXGBoost - Train accuracy",round(accuracy_score(y_train,xgb_fit.predict(x_train)),3))
print ("\nXGBoost  - Train Classification Report\n",classification_report(y_train,xgb_fit.predict(x_train)))

print ("\n\nXGBoost - Test Confusion Matrix\n\n",pd.crosstab(y_test,xgb_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nXGBoost - Test accuracy",round(accuracy_score(y_test,xgb_fit.predict(x_test)),3))
print ("\nXGBoost - Test Classification Report\n",classification_report(y_test,xgb_fit.predict(x_test)))


#Ensemble of Ensembles - by fitting various classifiers
clwght = {0:0.3,1:0.7}

# Classifier 1
from sklearn.linear_model import LogisticRegression
clf1_logreg_fit = LogisticRegression(fit_intercept=True,class_weight=clwght)
clf1_logreg_fit.fit(x_train,y_train)

print ("\nLogistic Regression for Ensemble - Train Confusion Matrix\n\n",pd.crosstab(y_train,clf1_logreg_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nLogistic Regression for Ensemble - Train accuracy",round(accuracy_score(y_train,clf1_logreg_fit.predict(x_train)),3))
print ("\nLogistic Regression for Ensemble - Train Classification Report\n",classification_report(y_train,clf1_logreg_fit.predict(x_train)))

print ("\n\nLogistic Regression for Ensemble - Test Confusion Matrix\n\n",pd.crosstab(y_test,clf1_logreg_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nLogistic Regression for Ensemble - Test accuracy",round(accuracy_score(y_test,clf1_logreg_fit.predict(x_test)),3))
print ("\nLogistic Regression for Ensemble - Test Classification Report\n",classification_report(y_test,clf1_logreg_fit.predict(x_test)))


# Classifier 2
from sklearn.tree import DecisionTreeClassifier
clf2_dt_fit = DecisionTreeClassifier(criterion="gini",max_depth=5,min_samples_split=2,
                                     min_samples_leaf=1,random_state=42,class_weight=clwght)
clf2_dt_fit.fit(x_train,y_train)

print ("\nDecision Tree for Ensemble - Train Confusion Matrix\n\n",pd.crosstab(y_train,clf2_dt_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nDecision Tree for Ensemble - Train accuracy",round(accuracy_score(y_train,clf2_dt_fit.predict(x_train)),3))
print ("\nDecision Tree for Ensemble - Train Classification Report\n",classification_report(y_train,clf2_dt_fit.predict(x_train)))

print ("\n\nDecision Tree for Ensemble - Test Confusion Matrix\n\n",pd.crosstab(y_test,clf2_dt_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nDecision Tree for Ensemble - Test accuracy",round(accuracy_score(y_test,clf2_dt_fit.predict(x_test)),3))
print ("\nDecision Tree for Ensemble - Test Classification Report\n",classification_report(y_test,clf2_dt_fit.predict(x_test)))


# Classifier 3
from sklearn.ensemble import RandomForestClassifier
clf3_rf_fit = RandomForestClassifier(n_estimators=10000,criterion="gini",max_depth=6,
                                min_samples_split=2,min_samples_leaf=1,class_weight = clwght)
clf3_rf_fit.fit(x_train,y_train)       

print ("\nRandom Forest for Ensemble - Train Confusion Matrix\n\n",pd.crosstab(y_train,clf3_rf_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nRandom Forest for Ensemble - Train accuracy",round(accuracy_score(y_train,clf3_rf_fit.predict(x_train)),3))
print ("\nRandom Forest for Ensemble - Train Classification Report\n",classification_report(y_train,clf3_rf_fit.predict(x_train)))

print ("\n\nRandom Forest for Ensemble - Test Confusion Matrix\n\n",pd.crosstab(y_test,clf3_rf_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nRandom Forest for Ensemble - Test accuracy",round(accuracy_score(y_test,clf3_rf_fit.predict(x_test)),3))
print ("\nRandom Forest for Ensemble - Test Classification Report\n",classification_report(y_test,clf3_rf_fit.predict(x_test)))


# Classifier 4
from sklearn.ensemble import AdaBoostClassifier
clf4_dtree = DecisionTreeClassifier(criterion='gini',max_depth=1,class_weight = clwght)
clf4_adabst_fit = AdaBoostClassifier(base_estimator= clf4_dtree,
        n_estimators=5000,learning_rate=0.05,random_state=42)

clf4_adabst_fit.fit(x_train, y_train)

print ("\nAdaBoost for Ensemble  - Train Confusion Matrix\n\n",pd.crosstab(y_train,clf4_adabst_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nAdaBoost for Ensemble   - Train accuracy",round(accuracy_score(y_train,clf4_adabst_fit.predict(x_train)),3))
print ("\nAdaBoost for Ensemble   - Train Classification Report\n",classification_report(y_train,clf4_adabst_fit.predict(x_train)))

print ("\n\nAdaBoost for Ensemble   - Test Confusion Matrix\n\n",pd.crosstab(y_test,clf4_adabst_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nAdaBoost for Ensemble   - Test accuracy",round(accuracy_score(y_test,clf4_adabst_fit.predict(x_test)),3))
print ("\nAdaBoost for Ensemble  - Test Classification Report\n",classification_report(y_test,clf4_adabst_fit.predict(x_test)))


ensemble = pd.DataFrame()

ensemble["log_output_one"] = pd.DataFrame(clf1_logreg_fit.predict_proba(x_train))[1]
ensemble["dtr_output_one"] = pd.DataFrame(clf2_dt_fit.predict_proba(x_train))[1]
ensemble["rf_output_one"] = pd.DataFrame(clf3_rf_fit.predict_proba(x_train))[1]
ensemble["adb_output_one"] = pd.DataFrame(clf4_adabst_fit.predict_proba(x_train))[1]

ensemble = pd.concat([ensemble,pd.DataFrame(y_train).reset_index(drop = True )],axis=1)

# Fitting meta-classifier
meta_logit_fit =  LogisticRegression(fit_intercept=False)
meta_logit_fit.fit(ensemble[['log_output_one','dtr_output_one','rf_output_one','adb_output_one']],ensemble['Attrition_ind'])

coefs =  meta_logit_fit.coef_
print ("Co-efficients for LR, DT, RF & AB are:",coefs)

ensemble_test = pd.DataFrame()
ensemble_test["log_output_one"] = pd.DataFrame(clf1_logreg_fit.predict_proba(x_test))[1]
ensemble_test["dtr_output_one"] = pd.DataFrame(clf2_dt_fit.predict_proba(x_test))[1]
ensemble_test["rf_output_one"] = pd.DataFrame(clf3_rf_fit.predict_proba(x_test))[1]
ensemble_test["adb_output_one"] = pd.DataFrame(clf4_adabst_fit.predict_proba(x_test))[1]

ensemble_test["all_one"] = meta_logit_fit.predict(ensemble_test[['log_output_one','dtr_output_one','rf_output_one','adb_output_one']])

ensemble_test = pd.concat([ensemble_test,pd.DataFrame(y_test).reset_index(drop = True )],axis=1)

print ("\n\nEnsemble of Models - Test Confusion Matrix\n\n",pd.crosstab(ensemble_test['Attrition_ind'],ensemble_test['all_one'],rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nEnsemble of Models - Test accuracy",round(accuracy_score(ensemble_test['Attrition_ind'],ensemble_test['all_one']),3))
print ("\nEnsemble of Models - Test Classification Report\n",classification_report(ensemble_test['Attrition_ind'],ensemble_test['all_one']))




# Ensemble of Ensembles - by applying bagging on simple classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

clwght = {0:0.3,1:0.7}

eoe_dtree = DecisionTreeClassifier(criterion='gini',max_depth=1,class_weight = clwght)
eoe_adabst_fit = AdaBoostClassifier(base_estimator= eoe_dtree,
        n_estimators=500,learning_rate=0.05,random_state=42)
eoe_adabst_fit.fit(x_train, y_train)

print ("\nAdaBoost - Train Confusion Matrix\n\n",pd.crosstab(y_train,eoe_adabst_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nAdaBoost - Train accuracy",round(accuracy_score(y_train,eoe_adabst_fit.predict(x_train)),3))
print ("\nAdaBoost  - Train Classification Report\n",classification_report(y_train,eoe_adabst_fit.predict(x_train)))

print ("\n\nAdaBoost - Test Confusion Matrix\n\n",pd.crosstab(y_test,eoe_adabst_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nAdaBoost - Test accuracy",round(accuracy_score(y_test,eoe_adabst_fit.predict(x_test)),3))
print ("\nAdaBoost - Test Classification Report\n",classification_report(y_test,eoe_adabst_fit.predict(x_test)))


bag_fit = BaggingClassifier(base_estimator= eoe_adabst_fit,n_estimators=50,
                            max_samples=1.0,max_features=1.0,
                            bootstrap=True,
                            bootstrap_features=False,
                            n_jobs=-1,
                            random_state=42)

bag_fit.fit(x_train, y_train)

print ("\nEnsemble of AdaBoost - Train Confusion Matrix\n\n",pd.crosstab(y_train,bag_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nEnsemble of AdaBoost - Train accuracy",round(accuracy_score(y_train,bag_fit.predict(x_train)),3))
print ("\nEnsemble of AdaBoost  - Train Classification Report\n",classification_report(y_train,bag_fit.predict(x_train)))

print ("\n\nEnsemble of AdaBoost - Test Confusion Matrix\n\n",pd.crosstab(y_test,bag_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nEnsemble of AdaBoost - Test accuracy",round(accuracy_score(y_test,bag_fit.predict(x_test)),3))
print ("\nEnsemble of AdaBoost - Test Classification Report\n",classification_report(y_test,bag_fit.predict(x_test)))








