library(keras)

# Prepare data

x1 = seq(-1,1,0.01)
x2 = 0.1-dnorm(x1,0,0.4)
x3 = 0.9-dnorm(x1,0,0.4)
x = rbind(cbind(x1,x2),cbind(x1,x3))
y = c(rep(1,length(x2)),rep(0,length(x2)))

plot(x, col = y+1)

#----------------#
#--- task 4.1 ---#
#----------------#

# draw model

ann_mod = keras_model_sequential(input_shape = 2, name = "AnnMod") %>% 
  layer_dense(units = 2, activation = "sigmoid", name = "hidden") %>% 
  layer_dense(units = 1, activation = "sigmoid", name = "output")

summary(ann_mod)

ann_mod %>% compile(loss = "binary_crossentropy", 
                    optimizer = optimizer_sgd(learning_rate = 0.1),
                    metrics = "accuracy")

ann_mod %>% fit(x, y, epochs = 50, batch_size = 32)

#ann_mod$set_weights()
y_pre = predict(ann_mod, x)
y_pre = ifelse(y_pre>0.5, 1, 0)

mean(y_pre == y)

# this accuracy is just a descent result. In fact, there is an optimal model which is 
# not easy to get due to the complicated loss function.

#----------------#
#--- task 4.2 ---#
#----------------#

# here, suppose we had a good pre-training on the model 'ann_mod'
ann_weights_ini = get_weights(ann_mod)
# change the values 
ann_weights_ini[[1]] = matrix(c(5,18,-8,16), byrow = T, 2,2)
ann_weights_ini[[2]] = array(c(-3,8))
ann_weights_ini[[3]] = matrix(c(0,0),2,1)
ann_weights_ini[[4]] = array(0)
# set model parameters
ann_mod$set_weights(ann_weights_ini)
# we just set those values as the initial values for the model parameters

# Now we complie and train the model with the good initial guess of model parameters.
ann_mod %>% compile(loss = "binary_crossentropy", 
                    optimizer = optimizer_sgd(learning_rate = 0.1),
                    metrics = "accuracy")

ann_mod %>% fit(x, y, epochs = 10, batch_size = 32)

# as you can see, we eventually found the optimal model with 100% acc.

get_weights(ann_mod)
ann_weights_ini

#----------------#
#--- task 4.3 ---#
#----------------#

pretrained_weight = get_weights(ann_mod)
# change the values 
pretrained_weight[[3]] = matrix(c(1,1),2,1)
pretrained_weight[[4]] = array(1)
# set model parameters
ann_mod$set_weights(pretrained_weight)

# now, suppose have a good pre-trained model with model parameters, 'pretrained_weight'
# next, we want to do some fine tuning. 

# The first hidden layer will be fixed. It means we will not update the value of model parameters
# of this hidden layer in the training stage.
ann_mod$layers[[1]]$trainable = FALSE

# from the model summary you can see that only the 3 parameters in the output layer are trainable.
summary(ann_mod)

ann_mod %>% compile(loss = "binary_crossentropy", 
                    optimizer = optimizer_sgd(learning_rate = 0.1),
                    metrics = "accuracy")

ann_mod %>% fit(x, y, epochs = 10, batch_size = 32)

get_weights(ann_mod)
pretrained_weight

#----------------#
#--- task 4.4 ---#
#----------------#

# We also can use the pre-trained model as the feature extraction 
# Below, we can create a function to get the extracted features.
layer_output <- get_layer(ann_mod, 'hidden')$output
feature_extractor <- keras_model(inputs = ann_mod$input, outputs = layer_output)
z = predict(feature_extractor, x)

# If you plot the extracted feature z and original feature x, you will 
# understand what is so called end to end learning.
par(mfrow = c(1,2))
plot(x, col = y+1)
plot(z, col = y+1)
