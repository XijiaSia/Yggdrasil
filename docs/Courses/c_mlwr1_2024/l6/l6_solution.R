
rm(list = ls())

#-----------------------------------------------------#
#------ Task 1: KNN method and Cross validation ------#
#-----------------------------------------------------#

library(class)

# Import data
dat_tr = read.table("BreastCancerTrain.txt", header = T, sep = ",")
dat_te = read.table("BreastCancerTest.txt", header = T, sep = ",")

dat_tr = dat_tr[,c(1,2,3,6)]
dat_te = dat_te[,c(1,2,3,6)]

# Task 1.1
K = seq(1,300,2)
num_k = length(K)

res_tr = res_te = numeric(num_k)

for(i in 1:num_k){
  y_tr_pre = knn(dat_tr[,-1], dat_tr[,-1], dat_tr[,1], K[i])
  res_tr[i] = mean(y_tr_pre == dat_tr[,1])
  y_te_pre = knn(dat_tr[,-1], dat_te[,-1], dat_tr[,1], K[i])
  res_te[i] =mean(y_te_pre == dat_te[,1])  
}

plot(1:num_k, res_tr, type = "b", col = "red", pch = 20, cex = 0.5,
     ylab = "Accuracy", xlab = "k", ylim = range(c(res_tr, res_te)))
points(1:num_k, res_te, type = "b", col = "blue", pch = 20, cex = 0.5)

# Task 1.2

K = seq(1,300,2)
num_k = length(K)

cv_acc = numeric(num_k) # creat a vector 'cv_acc' to save all the cross validation results for all candidate models
for(i in 1:num_k){
  # loop over all candidate models
  y_pre = knn.cv(train = dat_tr[,-1], cl = dat_tr[,1], k = K[i]) # get the prediction by leave one out cross validation
  cv_acc[i] = mean(dat_tr[,1] == y_pre) # calculate the cv accuracy for the ith candidate model
}
plot(1:num_k, cv_acc, type="b", pch = 20, cex = 0.5)
opt_k = which(cv_acc==max(cv_acc)) # find the optimal K
K[opt_k]

# Estimate the model performance with testing data
y_pre = knn(train = dat_tr[,-1], test = dat_te[,-1], cl = dat_tr[,1], k = K[opt_k])
mean(dat_te[,1] == y_pre)

# Task 1.3

K = seq(1,300,2)
num_k = length(K)

set.seed(8312)
cv_k_acc = numeric(num_k)
ID = matrix(sample(1:445), nrow = 5) # randomly split the sample into k folds
# each row of 'ID' contains all the id of observations in the corresponding fold

for(i in 1:num_k){
  temp_res = numeric(5)
  for(j in 1:5){
    y_pre = knn(train = dat_tr[-ID[j,],-1], # pick up the feature variables of observations not in the jth fold
                test = dat_tr[ID[j,],-1], # predict on the observations in the jth fold
                cl = dat_tr[-ID[j,],1], # target variable of observations not in the jth fold
                k = K[i])
    temp_res[j] = mean(y_pre == dat_tr[ID[j,],1]) # accuracy of the current model for the jth fold
  }
  cv_k_acc[i] = mean(temp_res) # overall cv accuracy for the jth fold
}

plot(1:num_k, cv_k_acc, type="b", pch = 20, cex = 0.5)
opt_k = which(cv_k_acc==max(cv_k_acc)) # find the optimal K
K[opt_k]

# Estimate the model performance with testing data
y_pre = knn(train = dat_tr[,-1], test = dat_te[,-1], cl = dat_tr[,1], k = K[opt_k])
mean(dat_te[,1] == y_pre)

#---------------------------------------#
#------ Task 2: Feature Selection ------#
#---------------------------------------#

#-------------#
# import data #
#-------------#

set.seed(8312)
n = 500
p = 20 
X = matrix(rnorm(n * p), nrow = n, ncol = p)
w = c(2, -3, 1, 2.3, 1.5, rep(0, p-5))
y = X %*% w + rnorm(n, mean = 0, sd = 1)
dat = data.frame(X, y = y)

#------------------------------------------#
#--- 2.1 Apply subset selection methods ---#
#------------------------------------------#

library(leaps)
m1 = regsubsets(y~., dat, nvmax = 10) # if you want to do forward/backward selection, then you need to add 'method' in the function
summary(m1) 

# solution 1
res = summary(m1)$which[8,-1]
temp_dat = dat[, res]
temp_dat$y = dat$y
m_temp = lm( y~., temp_dat )
m_temp$coefficients
# solution 2
coef(m1, id=8) # find all the variables and coefficients in the optimal model with 8 features 

#------------------------------------------------------------------------#
#--- 2.2: evaluate the optimal model with 8 features with testing set ---#
#------------------------------------------------------------------------#

set.seed(2024)
id = sample(1:dim(dat)[1], dim(dat)[1]*0.8)
dat_tr = dat[id,]
dat_te = dat[-id,]

m1 = regsubsets(y~., dat_tr, nvmax = 10)
x_test = model.matrix(y~., dat_te) # function for preparing prediction matrix for the testing set
coe = coef(m1, id = 8)
pred = x_test[, names(coe)]%*%coe # prediction. you also can implement this step by for loop
rmse = sqrt(mean((dat_te$y - pred)^2))
rmse

#-----------#
#--- 2.3 ---#
#-----------#

id_train = sample(id, length(id)*0.8)
dat_training = dat_tr[id_train, ] 
dat_validating = dat_tr[-id_train, ] # further split training set as training and validation set

# find the best model 
m1 = regsubsets(y~., dat_training, nvmax = 10) 
x_val = model.matrix(y~., dat_validating)

res = numeric(10)
for(i in 1:10){
  coe = coef(m1, id = i)
  pred = x_val[, names(coe)]%*%coe
  res[i] = sqrt(mean((dat_validating$y - pred)^2))
}
plot(res)
# we choose the model with 10 feature variables
# train the model with 10 feature variables and evaluate the model performance with the testing set.
which(res == min(res))

coef(m1, id = 10)
res = summary(m1)$which[8,-1]
temp_dat = dat_tr[, res]
temp_dat$y = dat_tr$y

final_m = lm(y~., data = temp_dat)
pred = predict(final_m, dat_te)
mean((dat_te$y - pred)^2) # it the estimate model performance

#-----------#
#--- 2.4 ---#
#-----------#

library(glmnet)
set.seed(2024)

model = cv.glmnet( x = as.matrix(dat_tr[,-21]), y = dat_tr[,21],
                   nfolds = 10, alpha = 1, type.measure = "mse")
plot(model)
model$lambda.min
model$lambda.1se
which(coef(model, s = "lambda.min")!=0)
which(coef(model, s = "lambda.1se")!=0)
