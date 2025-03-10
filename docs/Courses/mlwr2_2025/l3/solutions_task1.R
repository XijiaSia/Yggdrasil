library(keras)

#--------------------------------#
#--- Step 1: prepare the data ---#
#--------------------------------#
set.seed(2025)
N = 30 # sample size
x = runif(N, 0, 5) # regressor from uniform distribution
e = rnorm(N) # noise is from normal
y = 0.5 + 1.5*x + e
plot(x,y,pch=20,cex=2)

#------------------------------#
#--- Step 2: Draw the model ---#
#------------------------------#
Reg_Mod = keras_model_sequential(input_shape = 1) %>% 
  layer_dense(units = 1, activation = "linear")

get_weights(Reg_Mod)

y_pre = predict(Reg_Mod, x)
points(x, y_pre, col = "orange")
points(x, y_pre, type = "l", col = "orange")

summary(Reg_Mod)

# Alternatively, you can specify the input_shape in the first layer
# Reg_Mod = keras_model_sequential() %>% 
#   layer_dense(units = 1, activation = "linear", input_shape = 1)

#---------------------------------#
#--- Step 3: Compile the model ---#
#---------------------------------#
Reg_Mod %>% 
  compile(loss = "mean_squared_error",
          optimizer = optimizer_sgd(learning_rate = 0.05))

#-------------------------------#
#--- Step 4: Train the model ---#
#-------------------------------#
Reg_Mod %>% fit(x, y, epochs = 20, batch_size = N)
# check the weights estimation
get_weights(Reg_Mod)
# compare it with the outputs of 'lm'
lm(y~x)

#------------------#
#--- Prediction ---#
#------------------#
y_pre = predict(Reg_Mod, x)
points(x, y_pre, col = "blue")
points(x, y_pre, type = "l", col = "blue")
