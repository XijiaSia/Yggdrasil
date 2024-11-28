# solutions

#-------------------------#
#------ Import data ------#
#-------------------------#

library(MASS)
dat = Boston

n = dim(dat)[1]
y = dat$medv
x = dat$lstat

#----------------------#
#------ Task 1.1 ------#
#----------------------#

w_0_candidate = seq(34,35, 0.1)
w_1_candidate = seq(-1,0, 0.1)
# define a matrix 'res' to store all the mse
r0 = length(w_0_candidate)
r1 = length(w_1_candidate)
res = matrix(0, r0, r1)

# try all the combinations
for(i in 1:r0){
  for(j in 1:r1){
    residual = y - (w_0_candidate[i] + w_1_candidate[j]*x)
    res[i,j] = mean(residual^2)
  }
}
res # all MSEs

min(res)

(id = which(res == min(res), arr.ind = T))

# print the optimal solutions
w_0_candidate[id[1]]
w_1_candidate[id[2]]

# Alternative way by 'expand.grid' function
all_w_combinations = expand.grid(w_0_candidate, w_1_candidate)
head(all_w_combinations)

r = dim(all_w_combinations)[1]
res = numeric(r)
for(i in 1:r){
  residual = y - (all_w_combinations[i,1] + all_w_combinations[i,2]*x)
  res[i] = mean(residual^2)
}
id = which(res == min(res))
# print the optimal
all_w_combinations[id,]

#----------------------#
#------ Task 1.2 ------#
#----------------------#

n = dim(dat)[1]
y = dat$medv
x = dat$lstat
w1 = sum((x-mean(x))*(y-mean(y)))/sum((x-mean(x))^2)
# beta1 = cov(x,y)/var(x)
w0 = mean(y)-w1*mean(x)

print(paste("w1:", w1))
print(paste("w0:", w0))

#----------------------#
#------ Task 1.3 ------#
#----------------------#

m = lm(medv~lstat, data = dat)
m$coefficients


#----------------------#
#------ Task 2.1 ------#
#----------------------#

m1 = lm(medv~lstat+age, data = dat)
summary(m1)

y_pre = predict(m1, dat)
mean((dat$medv - y_pre)^2)

#----------------------#
#------ Task 2.2 ------#
#----------------------#

m1 = lm(medv~., data = dat)
summary(m1)

y_pre = predict(m1, dat)
mean((dat$medv - y_pre)^2)

#----------------------#
#------ Task 3.1 ------#
#----------------------#

set.seed(2023)
id = sample(1:n, round(0.8*n))
dat_tr = dat[id, ]
dat_te = dat[-id, ]

#---

m1 = lm(medv~lstat, data = dat_tr)
# training set
mse1_tr = mean( (dat_tr$medv - m1$fitted.value)^2 )
mse1_tr

# testing set
y_pre = predict(m1, newdata = dat_te)
mse1_te = mean((dat_te$medv - y_pre)^2)
mse1_te

#----------------------#
#------ Task 3.2 ------#
#----------------------#

m2 = lm(medv~poly(lstat,2), data = dat_tr)
#m2 = lm(dat_tr$medv~I(dat_tr$lstat) + I(dat_tr$lstat^2)) # it is an alternative way
# training set
mse2_tr = mean( (dat_tr$medv - m2$fitted.value)^2 )
mse2_tr

# testing set
y_pre = predict(m2, newdata = dat_te)
mse2_te = mean((dat_te$medv - y_pre)^2)
mse2_te

m7 = lm(medv~poly(lstat,7), data = dat_tr)
# training set
mse7_tr = mean((m7$residuals)^2)
mse7_tr

# testing set
y_pre = predict(m7, newdata = dat_te)
mse7_te = mean((dat_te$medv - y_pre)^2)
mse7_te

m20= lm(medv~poly(lstat,20), data = dat_tr)
# training set
mse20_tr = mean((m20$residuals)^2)
mse20_tr

# testing set
y_pre = predict(m20, newdata = dat_te)
mse20_te = mean((dat_te$medv - y_pre)^2)
mse20_te

