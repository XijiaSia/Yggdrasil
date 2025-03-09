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

ann_mod$set_weights(ann_weights)
y_pre = predict(ann_mod, x)
y_pre = ifelse(y_pre>0.5, 1, 0)

#----------------#
#--- task 4.2 ---#
#----------------#

ann_weights_ini = get_weights(ann_mod)
# change the values 
ann_weights_ini[[1]] = matrix(c(5,18,-8,16), byrow = T, 2,2)
ann_weights_ini[[2]] = array(c(-3,8))
ann_weights_ini[[3]] = matrix(c(0,0),2,1)
ann_weights_ini[[4]] = array(0)
# set model parameters
ann_mod$set_weights(ann_weights_ini)

ann_mod %>% compile(loss = "binary_crossentropy", 
                    optimizer = optimizer_sgd(learning_rate = 0.1),
                    metrics = "accuracy")

ann_mod %>% fit(x, y, epochs = 10, batch_size = 32)

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

ann_mod$layers[[1]]$trainable = FALSE

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

layer_output <- get_layer(ann_mod, 'hidden')$output
feature_extractor <- keras_model(inputs = ann_mod$input, outputs = layer_output)
z = predict(feature_extractor, x)

par(mfrow = c(1,2))
plot(x, col = y+1)
plot(z, col = y+1)