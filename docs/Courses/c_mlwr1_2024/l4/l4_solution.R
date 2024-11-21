# Import data
rm(list = ls()) # clean the workspace, remove all the objects in the r environment 

dat_tr = read.table("BreastCancerTrain.txt", header = T, sep = ",")
dat_te = read.table("BreastCancerTest.txt", header = T, sep = ",")

#--------------#
#--- Task 1 ---#
#--------------#

y = dat_tr$Diagnosis 
X = as.matrix(dat_tr[,c(2,3,6)]) # choose 'radius', 'texture', and 'smoothness'

# color = numeric(length(y)) # for data vis
# color[which(y == "M")] = 1 # for data vis
# pairs(X, col = color+1)

# estimate all means and covaraince matrix first. We use them for evaluate the likelihood value in line 26 and 27
mu_m = colMeans(X[which(y == "M"), ]) # estimate the means for M groups
S_m = cov(X[which(y == "M"), ]) # estimate the covariance matrix for M groups
mu_b = colMeans(X[which(y == "B"), ]) # estimate the means for B groups
S_b = cov(X[which(y == "B"), ]) # estimate the covariance matrix for B groups

library(mvtnorm) # need function 'dmvnorm' in this package 
# function for making decision based on the idea of GDA
classifier = function(x,mu1,S1,mu2,S2){
  ell1 = dmvnorm(x,mu1,S1) # the likelihood of x if assume it is from group 1
  ell2 = dmvnorm(x,mu2,S2) # the likelihood of x if assume it is from group 2
  res = ifelse(ell1 > ell2, "M", "B") # make decision
  return(res)
}

# test
y_te = dat_te$Diagnosis
X_te = as.matrix(dat_te[,c(2,3,6)])

y_pre = numeric(dim(dat_te)[1])
for(i in 1:dim(dat_te)[1]){ # loop over all observations in the testing set
  y_pre[i] = classifier(X_te[i, ], mu_m, S_m, mu_b, S_b) # apply our classifier
}
mean(y_pre == y_te) # accuracy

library(caret) # to calculate the confusion matrix and kappa statistic
confusionMatrix(as.factor(y_pre), as.factor(y_te), positive = "M")

#--------------#
#--- Task 2 ---#
#--------------#

p_m = mean(y == "M") # estimate the prior probability
classifier = function(x,mu1,S1,mu2,S2,p_m){
  # Here, we use function 'dmvnorm' in package 'mvtnorm'
  ell1 = dmvnorm(x,mu1,S1)*p_m # modify the likelihood by prior probability 
  ell2 = dmvnorm(x,mu2,S2)*(1-p_m)
  res = ifelse(ell1 > ell2, "M", "B")
  return(res)
}

for(i in 1:dim(dat_te)[1]){
  y_pre[i] = classifier(X_te[i, ], mu_m, S_m, mu_b, S_b, p_m)
}
mean(y_pre == y_te)
confusionMatrix(as.factor(y_pre), as.factor(y_te), positive = "M")


#--------------#
#--- Task 3 ---#
#--------------#

# 3.1
library(MASS)
# estimate the LDA model by function 'lda'
m = lda(Diagnosis~radius+texture+smoothness, data = dat_tr, prior = c(0.5,0.5))
# apply the output model 'm' on observations in the testing set.
res = predict(m, newdata = dat_te)

str(res)
y_pre = res$class
mean(y_pre == y_te)

# 3.2
# weights:
w = m$scaling
# bias:
w_0 = mean(m$means%*%m$scaling)

# 3.3
X_M = X[which(y == "M"), ]
N_XM = dim(X_M)[1]
X_M = X_M - matrix(rep(mu_m, N_XM), nrow = N_XM)

X_B = X[which(y == "B"), ]
N_XB = dim(X_B)[1]
X_B = X_B - matrix(rep(mu_b, N_XB), nrow = N_XB)

S_pooled = cov(rbind(X_M, X_B))

classifier = function(x,mu1,mu2,S,p_m = 0.5){
  # Here, we use function 'dmvnorm' in package 'mvtnorm'
  ell1 = dmvnorm(x,mu1,S)*p_m # modify the likelihood by prior probability 
  ell2 = dmvnorm(x,mu2,S)*(1-p_m)
  res = ifelse(ell1 > ell2, "M", "B")
  return(res)
}