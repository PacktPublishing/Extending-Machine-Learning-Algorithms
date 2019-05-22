letter_data = read.csv("letterdata.csv")

set.seed(123)
numrow = nrow(letter_data)
trnind = sample(1:numrow,size = as.integer(0.7*numrow))
train_data = letter_data[trnind,]
test_data = letter_data[-trnind,]


library(e1071)

accrcy <- function(matrx){ 
  return( sum(diag(matrx)/sum(matrx)))}

precsn <- function(matrx){
  return(diag(matrx) / rowSums(matrx))
}

recll <- function(matrx){
  return(diag(matrx) / colSums(matrx))
}




# SVM - Linear Kernel
svm_fit = svm(letter~.,data = train_data,kernel="linear",cost=1.0,scale = TRUE)

tr_y_pred = predict(svm_fit, train_data)
ts_y_pred = predict(svm_fit,test_data)

tr_y_act = train_data$letter;ts_y_act = test_data$letter

tr_tble = table(tr_y_act,tr_y_pred)
print(paste("Train Confusion Matrix"))
print(tr_tble)
tr_acc = accrcy(tr_tble)
print(paste("SVM Linear Kernel Train accuracy:",round(tr_acc,4)))

tr_prec = precsn(tr_tble)
print(paste("SVM Linear Kernel Train Precision:"))
print(tr_prec)

tr_rcl = recll(tr_tble)
print(paste("SVM Linear Kernel Train Recall:"))
print(tr_rcl)

ts_tble = table(ts_y_act,ts_y_pred)
print(paste("Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_tble)
print(paste("SVM Linear Kernel Test accuracy:",round(ts_acc,4)))

ts_prec = precsn(ts_tble)
print(paste("SVM Linear Kernel Test Precision:"))
print(ts_prec)

ts_rcl = recll(ts_tble)
print(paste("SVM Linear Kernel Test Recall:"))
print(ts_rcl)


# SVM - Polynomial Kernel
svm_poly_fit = svm(letter~.,data = train_data,kernel="poly",cost=1.0,degree = 2  ,scale = TRUE)

tr_y_pred = predict(svm_poly_fit, train_data)
ts_y_pred = predict(svm_poly_fit,test_data)

tr_y_act = train_data$letter;ts_y_act = test_data$letter


tr_tble = table(tr_y_act,tr_y_pred)
print(paste("Train Confusion Matrix"))
print(tr_tble)
tr_acc = accrcy(tr_tble)
print(paste("SVM Polynomial Kernel Train accuracy:",round(tr_acc,4)))

tr_prec = precsn(tr_tble)
print(paste("SVM Polynomial Kernel Train Precision:"))
print(tr_prec)

tr_rcl = recll(tr_tble)
print(paste("SVM Polynomial Kernel Train Recall:"))
print(tr_rcl)

ts_tble = table(ts_y_act,ts_y_pred)
print(paste("Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_tble)
print(paste("SVM Polynomial Kernel Test accuracy:",round(ts_acc,4)))

ts_prec = precsn(ts_tble)
print(paste("SVM Polynomial Kernel Test Precision:"))
print(ts_prec)

ts_rcl = recll(ts_tble)
print(paste("SVM Polynomial Kernel Test Recall:"))
print(ts_rcl)



# SVM - RBF Kernel
svm_rbf_fit = svm(letter~.,data = train_data,kernel="radial",cost=1.0,gamma = 0.2  ,scale = TRUE)

tr_y_pred = predict(svm_rbf_fit, train_data)
ts_y_pred = predict(svm_rbf_fit,test_data)

tr_y_act = train_data$letter;ts_y_act = test_data$letter

tr_tble = table(tr_y_act,tr_y_pred)
print(paste("Train Confusion Matrix"))
print(tr_tble)
tr_acc = accrcy(tr_tble)
print(paste("SVM RBF Kernel Train accuracy:",round(tr_acc,4)))

tr_prec = precsn(tr_tble)
print(paste("SVM RBF Kernel Train Precision:"))
print(tr_prec)

tr_rcl = recll(tr_tble)
print(paste("SVM RBF Kernel Train Recall:"))
print(tr_rcl)

ts_tble = table(ts_y_act,ts_y_pred)
print(paste("Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_tble)
print(paste("SVM RBF Kernel Test accuracy:",round(ts_acc,4)))

ts_prec = precsn(ts_tble)
print(paste("SVM RBF Kernel Test Precision:"))
print(ts_prec)

ts_rcl = recll(ts_tble)
print(paste("SVM RBF Kernel Test Recall:"))
print(ts_rcl)



# Grid search - RBF Kernel
library(e1071)
svm_rbf_grid = tune(svm,letter~.,data = train_data,kernel="radial",scale=TRUE,ranges = list(
  cost = c(0.1,0.3,1,3,10,30),
  gamma = c(0.001,0.01,0.1,0.3,1)
  
),
tunecontrol = tune.control(cross = 5)
)

print(paste("Best parameter from Grid Search"))
print(summary(svm_rbf_grid))

best_model = svm_rbf_grid$best.model

tr_y_pred = predict(best_model,data = train_data,type = "response")
ts_y_pred = predict(best_model,newdata = test_data,type = "response")

tr_y_act = train_data$letter;ts_y_act = test_data$letter


tr_tble = table(tr_y_act,tr_y_pred)
print(paste("Train Confusion Matrix"))
print(tr_tble)
tr_acc = accrcy(tr_tble)
print(paste("SVM RBF Kernel Train accuracy:",round(tr_acc,4)))

tr_prec = precsn(tr_tble)
print(paste("SVM RBF Kernel Train Precision:"))
print(tr_prec)

tr_rcl = recll(tr_tble)
print(paste("SVM RBF Kernel Train Recall:"))
print(tr_rcl)

ts_tble = table(ts_y_act,ts_y_pred)
print(paste("Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_tble)
print(paste("SVM RBF Kernel Test accuracy:",round(ts_acc,4)))

ts_prec = precsn(ts_tble)
print(paste("SVM RBF Kernel Test Precision:"))
print(ts_prec)

ts_rcl = recll(ts_tble)
print(paste("SVM RBF Kernel Test Recall:"))
print(ts_rcl)