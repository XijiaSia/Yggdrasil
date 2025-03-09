library(keras)
rm(list = ls())

#--------------------------------#
#--- Step 1: prepare the data ---#
#--------------------------------#
dat = iris
dat = dat[which(dat$Species != "setosa"), ]
dat$Species = as.numeric(dat$Species)
pairs(dat[,-5], col = dat$Species)
x = as.matrix(dat[,-c(1,2,5)])
y = dat[,5]-2
#y = to_categorical(y)

#------------------------------#
#--- Step 2: Draw the model ---#
#------------------------------#
LogReg_Mod = keras_model_sequential(input_shape = 2, name = "LogReg") %>% 
  layer_dense(units = 1, activation = "sigmoid", name = "dense_layer")

LogReg_Mod
get_weights(LogReg_Mod)

#y_pre = predict(LogReg_Mod, x)

#---------------------------------#
#--- Step 3: Compile the model ---#
#---------------------------------#
LogReg_Mod %>% compile(loss = "binary_crossentropy", 
                       optimizer = optimizer_sgd(learning_rate = 0.1),
                       metrics = "accuracy")

#-------------------------------#
#--- Step 4: Train the model ---#
#-------------------------------#
LogReg_Mod %>% fit(x, y, epochs = 100, batch_size = 20)

#------------------#
#--- Prediction ---#
#------------------#
get_weights(LogReg_Mod)

# logistic regression
mod = glm(y~x, family = "binomial")
mod$coefficients
y_pre = predict(mod, data.frame(x))
y_pre = as.numeric(1/(1+exp(-y_pre)) > 0.5)
mean(y_pre == y)

