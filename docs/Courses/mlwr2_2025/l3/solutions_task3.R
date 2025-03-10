rm(list = ls())
library(keras)

#--------------------------------#
#--- Step 1: prepare the data ---#
#--------------------------------#

mnist = dataset_mnist()
x_tr = mnist$train$x
g_tr = mnist$train$y
x_te = mnist$test$x
g_te = mnist$test$y

x_tr = array_reshape(x_tr, c(nrow(x_tr), 784))
x_te = array_reshape(x_te, c(nrow(x_te), 784))
y_tr = to_categorical(g_tr, 10)
y_te = to_categorical(g_te, 10)
x_tr = x_tr/255
x_te = x_te/255

#------------------------------#
#--- Step 2: Draw the model ---#
#------------------------------#

ANN_mod = keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = "softmax")

ANN_mod

#---------------------------------#
#--- Step 3: Compile the model ---#
#---------------------------------#

ANN_mod %>% compile(loss = "categorical_crossentropy", 
                    optimizer = optimizer_rmsprop(),
                    metrics = c("accuracy"))

#-------------------------------#
#--- Step 4: Train the model ---#
#-------------------------------#

Training_history = ANN_mod %>% 
  fit(x_tr, y_tr, repochs = 30, batch_size = 128, validation_split = 0.2)

Training_history
plot(Training_history)

Training_history$metrics

#------------------#
#--- Prediction ---#
#------------------#

# you can get the probabily for each group for each individual
predict_1 = predict(ANN_mod, x_te)
# or you can use the pipe line to get the categorical prediction directly
predict_2 = ANN_mod %>% predict(x_te) %>% k_argmax() %>% as.array()
mean(predict_2 == g_te)
table(predict_2, g_te)
