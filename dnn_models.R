# /////////////////////////////////////////////////////////////////////////// 
# G Research Financial Forecasting Competition 
# /////////////////////////////////////////////////////////////////////////// 
# - Carson Goeke
# - 2/3/17

# Load Packages ------------------------------------------------------------- 
library(keras) # neural network
library(xgboost) # gradient boosting machines
library(randomForest) # random forest algorithm
library(bartMachine) # Bayesian Additive Regression Trees
library(dplyr) # data manipulation
library(readr) # reading data
library(DataCombine) # Creating lags
library(magrittr) # pipe operator
library(ggplot2) # visualization
library(DescTools) # Skew and Kurtosis
library(e1071) # svm

# Read in data -------------------------------------------------------------- 
train <- read_csv('~/Desktop/gresearch/train.csv')
test <- read_csv('~/Desktop/gresearch/test.csv')

# Data Munging -------------------------------------------------------------- 

# summary stats
summary(train) # x1: 55 NA's, x2: 5 NA's 
summary(test) # x1: 62 NA's, x2: 2 NA's 
# Also note that all the x variables are positive

# let's impute NA values with median so we can apply log transform
med_x1 <- train$x1[!is.na(train$x1)]  %>% median()
med_x2 <- train$x2[!is.na(train$x2)]  %>% median()
train[is.na(train$x1),'x1'] <- med_x1
test[is.na(test$x1),'x1'] <- med_x1
train[is.na(train$x2),'x2'] <- med_x2
test[is.na(test$x2),'x2'] <- med_x2

# log transform all x vars 
train_log_xs <- sapply(train[,grepl('x',colnames(train))], function(x){log(x + 0.0000001)}) %>% as.data.frame()
test_log_xs <- sapply(test[,grepl('x',colnames(test))], function(x){log(x + 0.0000001)}) %>% as.data.frame()
train_log_xs <- train_log_xs[,2:12] # don't need Index
test_log_xs <- test_log_xs[,2:12] # don't need Index
colnames(train_log_xs) <- c('l.x0', 'l.x1', 'l.x2', 'l.x3A', 'l.x3B', 'l.x3C', 'l.x3D', 'l.x3E',  'l.x4', 'l.x5', 'l.x6')
colnames(test_log_xs) <- colnames(train_log_xs)

# Square Transform
train_square_xs <- sapply(train[,grepl('x',colnames(train))], function(x){x^2}) %>% as.data.frame()
test_square_xs <- sapply(test[,grepl('x',colnames(test))], function(x){x^2}) %>% as.data.frame()
train_square_xs <- train_square_xs[,2:12] # don't need Index
test_square_xs <- test_square_xs[,2:12] # don't need Index
colnames(train_square_xs) <- c('sq.x0', 'sq.x1', 'sq.x2', 'sq.x3A', 'sq.x3B', 'sq.x3C', 'sq.x3D', 'sq.x3E',  'sq.x4', 'sq.x5', 'sq.x6')
colnames(test_square_xs) <- colnames(train_square_xs)

# Inverse Transform
train_inv_xs <- sapply(train[,grepl('x',colnames(train))], function(x){1/(x + 0.0000001)}) %>% as.data.frame()
test_inv_xs <- sapply(test[,grepl('x',colnames(test))], function(x){1/(x + 0.0000001)}) %>% as.data.frame()
train_inv_xs <- train_inv_xs[,2:12] # don't need Index
test_inv_xs <- test_inv_xs[,2:12] # don't need Index
colnames(train_inv_xs) <- c('inv.x0', 'inv.x1', 'inv.x2', 'inv.x3A', 'inv.x3B', 'inv.x3C', 'inv.x3D', 'inv.x3E',  'inv.x4', 'inv.x5', 'inv.x6')
colnames(test_inv_xs) <- colnames(train_inv_xs)

# Inverse Square Transform
train_inv2_xs <- sapply(train[,grepl('x',colnames(train))], function(x){1/(x + 0.0000001)^2}) %>% as.data.frame()
test_inv2_xs <- sapply(test[,grepl('x',colnames(test))], function(x){1/(x + 0.0000001)^2}) %>% as.data.frame()
train_inv2_xs <- train_inv2_xs[,2:12] # don't need Index
test_inv2_xs <- test_inv2_xs[,2:12] # don't need Index
colnames(train_inv2_xs) <- c('inv2.x0', 'inv2.x1', 'inv2.x2', 'inv2.x3A', 'inv2.x3B', 'inv2.x3C', 'inv2.x3D', 'inv2.x3E',  'inv2.x4', 'inv2.x5', 'inv2.x6')
colnames(test_inv2_xs) <- colnames(train_inv2_xs)

# Cosine Transform
train_cos_xs <- sapply(train[,grepl('x',colnames(train))], function(x){cos(x*2*pi)}) %>% as.data.frame()
test_cos_xs <- sapply(test[,grepl('x',colnames(test))], function(x){cos(x*2*pi)}) %>% as.data.frame()
train_cos_xs <- train_cos_xs[,2:12] # don't need Index
test_cos_xs <- test_cos_xs[,2:12] # don't need Index
colnames(train_cos_xs) <- c('cos.x0', 'cos.x1', 'cos.x2', 'cos.x3A', 'cos.x3B', 'cos.x3C', 'cos.x3D', 'cos.x3E',  'cos.x4', 'cos.x5', 'cos.x6')
colnames(test_cos_xs) <- colnames(train_cos_xs)

# Sine Transform
train_sin_xs <- sapply(train[,grepl('x',colnames(train))], function(x){sin(x*2*pi)}) %>% as.data.frame()
test_sin_xs <- sapply(test[,grepl('x',colnames(test))], function(x){sin(x*2*pi)}) %>% as.data.frame()
train_sin_xs <- train_sin_xs[,2:12] # don't need Index
test_sin_xs <- test_sin_xs[,2:12] # don't need Index
colnames(train_sin_xs) <- c('sin.x0', 'sin.x1', 'sin.x2', 'sin.x3A', 'sin.x3B', 'sin.x3C', 'sin.x3D', 'sin.x3E',  'sin.x4', 'sin.x5', 'sin.x6')
colnames(test_sin_xs) <- colnames(train_sin_xs)


# Interaction Terms # could write this with a nested loop but fuck it
interactions <- function(df) {
  
  # x0
  df$x0.x1 <- df$x0 * df$x1
  df$x0.x2 <- df$x0 * df$x2
  df$x0.x3A <- df$x0 * df$x3A
  df$x0.x3B <- df$x0 * df$x3B
  df$x0.x3B <- df$x0 * df$x3B
  df$x0.x3C <- df$x0 * df$x3C
  df$x0.x3D <- df$x0 * df$x3D
  df$x0.x3E <- df$x0 * df$x3E
  df$x0.x4 <- df$x0 * df$x4
  df$x0.x5 <- df$x0 * df$x5
  df$x0.x6 <- df$x0 * df$x6
  
  # x1
  df$x1.x2 <- df$x1 * df$x2
  df$x1.x3A <- df$x1 * df$x3A
  df$x1.x3B <- df$x1 * df$x3B
  df$x1.x3B <- df$x1 * df$x3B
  df$x1.x3C <- df$x1 * df$x3C
  df$x1.x3D <- df$x1 * df$x3D
  df$x1.x3E <- df$x1 * df$x3E
  df$x1.x4 <- df$x1 * df$x4
  df$x1.x5 <- df$x1 * df$x5
  df$x1.x6 <- df$x1 * df$x6
  
  # x2
  df$x2.x3A <- df$x2 * df$x3A
  df$x2.x3B <- df$x2 * df$x3B
  df$x2.x3B <- df$x2 * df$x3B
  df$x2.x3C <- df$x2 * df$x3C
  df$x2.x3D <- df$x2 * df$x3D
  df$x2.x3E <- df$x2 * df$x3E
  df$x2.x4 <- df$x2 * df$x4
  df$x2.x5 <- df$x2 * df$x5
  df$x2.x6 <- df$x2 * df$x6
  
  # x3A
  df$x3A.x3B <- df$x3A * df$x3B
  df$x3A.x3B <- df$x3A * df$x3B
  df$x3A.x3C <- df$x3A * df$x3C
  df$x3A.x3D <- df$x3A * df$x3D
  df$x3A.x3E <- df$x3A * df$x3E
  df$x3A.x4 <- df$x3A * df$x4
  df$x3A.x5 <- df$x3A * df$x5
  df$x3A.x6 <- df$x3A * df$x6
  
  # x3B
  df$x3B.x3C <- df$x3B * df$x3C
  df$x3B.x3D <- df$x3B * df$x3D
  df$x3B.x3E <- df$x3B * df$x3E
  df$x3B.x4 <- df$x3B * df$x4
  df$x3B.x5 <- df$x3B * df$x5
  df$x3B.x6 <- df$x3B * df$x6
  
  # x3C
  df$x3C.x3D <- df$x3C * df$x3D
  df$x3C.x3E <- df$x3C * df$x3E
  df$x3C.x4 <- df$x3C * df$x4
  df$x3C.x5 <- df$x3C * df$x5
  df$x3C.x6 <- df$x3C * df$x6
  
  # x3D
  df$x3D.x3E <- df$x3D * df$x3E
  df$x3D.x4 <- df$x3D * df$x4
  df$x3D.x5 <- df$x3D * df$x5
  df$x3D.x6 <- df$x3D * df$x6
  
  # x3E
  df$x3E.x4 <- df$x3E * df$x4
  df$x3E.x5 <- df$x3E * df$x5
  df$x3E.x6 <- df$x3E * df$x6
  
  # x4
  df$x4.x5 <- df$x4 * df$x5
  df$x4.x6 <- df$x4 * df$x6
  
  # x5
  df$x5.x6 <- df$x5 * df$x6
  
  # return df
  return(df)
}
train %<>% interactions()
test %<>% interactions()

# Merge with other transforms
train <- cbind(train, train_log_xs, train_square_xs, train_inv_xs, train_inv2_xs, train_cos_xs)
test <- cbind(test, test_log_xs, test_square_xs, test_inv_xs, test_inv2_xs, test_cos_xs)
rm(train_log_xs, test_log_xs, train_square_xs, test_square_xs, train_inv_xs, test_inv_xs, train_inv2_xs, test_in2v_xs)

# get day of week by taking day mod 7
train$day_of_week <- train$Day %% 7
test$day_of_week <- test$Day %% 7

# get approximate monthly periodicity by taking day mod 30
train$day_of_month <- train$Day %% 30
test$day_of_month <- test$Day %% 30

# get stats of y for each stock
y_stats <- train %>% group_by(Stock) %>% summarize(mean_y = mean(y),
                                                   median_y = median(y),
                                                   sd_y = sd(y),
                                                   skew_y = Skew(y),
                                                   kurt_y = Kurt(y)) %>% as.data.frame()

y_stats[is.na(y_stats),] <- 0

y_stats$low_y <- (y_stats$median_y < quantile(y_stats$median_y)[2]) %>% as.numeric()
y_stats$high_y <- (y_stats$median_y > quantile(y_stats$median_y)[4]) %>% as.numeric()
y_stats$high_sd <- (y_stats$sd_y > quantile(y_stats$sd_y)[4]) %>% as.numeric()
y_stats$left_skew <- (y_stats$skew_y < quantile(y_stats$skew_y)[2]) %>% as.numeric()
y_stats$right_skew <- (y_stats$skew_y > quantile(y_stats$skew_y)[4]) %>% as.numeric()
y_stats$high_kurt <- (y_stats$kurt_y > quantile(y_stats$kurt_y)[4]) %>% as.numeric()

# replace na values
y_stats[is.na(y_stats)] <- 0

# merge with train and test
train <- merge(train, y_stats, by = 'Stock', all.x = TRUE, sort = FALSE)
test <- merge(test, y_stats, by = 'Stock', all.x = TRUE, sort = FALSE)

# convert day of week and Market to dummy vars
train$day_of_week %<>% as.factor()
test$day_of_week %<>% as.factor()
train$Market %<>% as.factor()
test$Market %<>% as.factor()

# Check for any important Stocks specifically
train$Stock %<>% as.factor()
test$Stock %<>% as.factor()
levels(test$Stock) <- levels(train$Stock)

# was previously tibble which doesnt work with model.matrix
train %<>% as.data.frame() 
test %<>% as.data.frame() 

# convert vars to dummies
train_days <- model.matrix(~ train[,'day_of_week'] - 1, train) %>% as.data.frame()
colnames(train_days) <- c('d.0', 'd.1', 'd.2', 'd.5', 'd.6')
test_days <- model.matrix(~ test[,'day_of_week'] - 1, test) %>% as.data.frame()
colnames(test_days) <- c('d.0', 'd.1', 'd.2', 'd.5', 'd.6')

train_market <- model.matrix(~ train[,'Market'] - 1, train) %>% as.data.frame()
colnames(train_market) <- c('m.1', 'm.2', 'm.3', 'm.4')
test_market <- model.matrix(~ test[,'Market'] - 1, test) %>% as.data.frame()
colnames(test_market) <- c('m.1', 'm.2', 'm.3', 'm.4')

# Merge with dummies
train %<>% cbind(train_days, train_market)
test %<>% cbind(test_days, test_market)

# Data Exploration ---------------------------------------------------------- 

# Get a more maneagble sample of data for visualization
sample <- train[sample(seq(1:nrow(train)), 20000),]
plot(density(sample$y))

# All the x's are positive and could benefit from a log transformation
plot(density(sample$x0))
plot(density(log(sample$x0)))

# Get a sample of one stock so we can more clearly visualize 
# relationships between variables
one_stock <- train[train$Stock == 1,]

# Using log transform helps to determine relationship with x vars
plot(log(one_stock$x0), one_stock$y) # negative / non-linear
plot(log(one_stock$x1), one_stock$y) # none / non-linear
plot(log(one_stock$x2), one_stock$y) # positive
plot(log(one_stock$x3A), one_stock$y) # negative
plot(log(one_stock$x3B), one_stock$y) # negative
plot(log(one_stock$x3C), one_stock$y) # negative
plot(log(one_stock$x3D), one_stock$y) # negative
plot(log(one_stock$x3E), one_stock$y) # negative / non-linear
plot(log(one_stock$x4), one_stock$y) # none / non-linear
plot(log(one_stock$x5), one_stock$y) # none / quadratic?

# convert day_of_week and Market back to numeric just to see if it helps
train$Market %<>% as.numeric()
test$Market %<>% as.numeric()
train$day_of_week %<>% as.numeric()
test$day_of_week %<>% as.numeric()

# Data Preparation --------------------------------------------------------------- 

# get train and test matrices needed for xgboost
x_train <- train[, c(3:15, 18:ncol(train))] %>% as.matrix()
x_test <- test[,3:ncol(test)] %>% as.matrix()

y_train <- train[, 16] %>% as.matrix()

# training weights
weights_train <- train[, 'Weight'] %>% as.matrix()

# replace na values
x_train[is.na(x_train)] <- 0
x_test[is.na(x_test)] <- 0

# normalize the data 
mean <- apply(x_train, 2, mean)
sd <- apply(x_train, 2, sd)
x_train %<>% scale(center = mean, scale = sd)
x_test %<>% scale(center = mean, scale = sd)

# Keras Deep Learning Model ---------------------------------------------------------------

# want to sample with respect to observation weights 
# but they're very skewed so we'll take the log and square it, so that some obvs aren't
log_weights_train_sq <- log(weights_train)^2

weighted_bootstrap_sample <- sample(seq(1:nrow(x_train)), 2*nrow(x_train), replace = TRUE, prob = log_weights_train_sq)
x_train_boot <- x_train[weighted_bootstrap_sample,]
y_train_boot <- y_train[weighted_bootstrap_sample,]
          

# initialize model
dnn <- keras_model_sequential()
dnn %>% 
  layer_dense(units = 16, activation = 'elu', input_shape = c(ncol(x_train)), kernel_regularizer = regularizer_l2(0.01)) %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'tanh') # using tanh b/c y is bounded

# Compile the model
dnn %>% compile(
  loss = 'mse',
  optimizer = 'rmsprop',
  metric = c('kullback_leibler_divergence')
)

callbacks_list <- list(
  callback_early_stopping(
    monitor = "loss",
    patience = 4
  ),
  callback_model_checkpoint(
    filepath = "my_model",
    monitor = "val_loss",
    save_best_only = TRUE
  ) )

# Train the model
dnn %>% fit(x_train_boot, y_train_boot,
            batch_size = 20000,
            epochs = 100,
            callbacks = callbacks_list,
            verbose = 1,
            validation_split = 0.1,
            shuffle = TRUE
)

# plot distribution of weights
weights <- dnn$get_weights()
weights[[1]] %>% as.numeric() %>% density() %>% plot() # layer 1 weights
weights[[2]] %>% as.numeric() %>% density() %>% plot() # hidden layer weights
weights[[3]] %>% as.numeric() %>% density() %>% plot() # hidden layer biases

# Evaluate the model with competition metric (weighted MSE)
log(weights_train * (as.numeric(y_train) - predict(dnn, x_train))^2) %>% density() %>% plot()

# Predictions -------------------------------------------------------------------- 

# boost
dnn_predictions <- cbind(test$Index, predict(dnn, x_test)) %>% as.data.frame()
colnames(dnn_predictions) <- c('Index', 'y')
dnn_predictions$Index %<>% as.integer() # submission doesn't accept numeric index
dnn_predictions <- dnn_predictions[order(dnn_predictions$Index),] # submission needs to be in original order

# Export Predictions
write.csv(dnn_predictions, '~/Desktop/gresearch/dnn.csv', na = '', row.names = FALSE)
