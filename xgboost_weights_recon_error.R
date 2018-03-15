# /////////////////////////////////////////////////////////////////////////// 
# G Research Financial Forecasting Competition 
# /////////////////////////////////////////////////////////////////////////// 
# - Carson Goeke

# Load Packages ------------------------------------------------------------- 
library(keras) # neural network
library(xgboost) # gradient boosting machines
library(randomForest) # random forest algorithm
library(dplyr) # data manipulation
library(readr) # reading data
library(DataCombine) # Creating lags
library(magrittr) # pipe operator
library(ggplot2) # visualization
library(DescTools) # Skew and Kurtosis

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

# log transform all x vars and then square
train_log2_xs <- sapply(train[,grepl('x',colnames(train))], function(x){log(x + 0.0000001)^2}) %>% as.data.frame()
test_log2_xs <- sapply(test[,grepl('x',colnames(test))], function(x){log(x + 0.0000001)^2}) %>% as.data.frame()
train_log2_xs <- train_log2_xs[,2:12] # don't need Index
test_log2_xs <- test_log2_xs[,2:12] # don't need Index
colnames(train_log2_xs) <- c('l2.x0', 'l2.x1', 'l2.x2', 'l2.x3A', 'l2.x3B', 'l2.x3C', 'l2.x3D', 'l2.x3E',  'l2.x4', 'l2.x5', 'l2.x6')
colnames(test_log2_xs) <- colnames(train_log2_xs)

# neg exp transform (only seems good fro x1 and x2)
train$neg_exp1 <- exp(-train$x1)
test$neg_exp1 <- exp(-test$x1)
train$neg_exp2 <- exp(-train$x2)
test$neg_exp2 <- exp(-test$x2)

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
 
#train %<>% interactions()
#test %<>% interactions()

# Merge with other transforms
train <- cbind(train, train_log_xs, train_log2_xs)
test <- cbind(test, test_log_xs, test_log2_xs)
rm(train_log_xs, test_log_xs, train_log2_xs, test_log2_xs)

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

# sparse vars
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
# days
train_days <- model.matrix(~ train[,'day_of_week'] - 1, train) %>% as.data.frame()
colnames(train_days) <- c('d.0', 'd.1', 'd.2', 'd.5', 'd.6')
test_days <- model.matrix(~ test[,'day_of_week'] - 1, test) %>% as.data.frame()
colnames(test_days) <- c('d.0', 'd.1', 'd.2', 'd.5', 'd.6')

# market
train_market <- model.matrix(~ train[,'Market'] - 1, train) %>% as.data.frame()
colnames(train_market) <- c('m.1', 'm.2', 'm.3', 'm.4')
test_market <- model.matrix(~ test[,'Market'] - 1, test) %>% as.data.frame()
colnames(test_market) <- c('m.1', 'm.2', 'm.3', 'm.4')

# Merge with dummies
train %<>% cbind(train_days, train_market)
test %<>% cbind(test_days, test_market)

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
med <- apply(x_train, 2, median) # data is unstable
mad <- apply(x_train, 2, mad) # median absolute deviation for same reason
x_train %<>% scale(center = med, scale = mad)
x_test %<>% scale(center = med, scale = mad)

y_med <- median(y_train)
y_mad <- mad(y_train)
y_scaled <- scale(y_train, center = y_med, scale = y_mad)

# clean up environment
rm(test_days, test_inv_xs, test_inv2_xs, test_market,
   train_days, test_inv2_xs, test_inv2_xs, test_market,
   y_stats)

# Reconstruction Error ------------------------------------------------------
ae <- keras_model_sequential()
ae %>% 
  #layer_gaussian_noise(stddev = 0.1, input_shape = c(ncol(x_train))) %>%
  layer_dense(units = 10, activation = 'elu', input_shape = c(ncol(x_train[,1:13]))) %>%
  layer_dense(units = 5, activation = 'elu') %>%
  layer_dense(units = 10, activation = 'elu') %>%
  layer_dense(units = ncol(x_train[,1:13]), activation = 'linear') # using tanh b/c y is bounded

# Compile the model
ae %>% compile(
  loss = 'mse',
  optimizer = 'rmsprop',
  metric = c('kullback_leibler_divergence')
)

# Train the model
ae %>% fit(x_train[,1:13], x_train[,1:13],
           batch_size = 10000,
           epochs = 30,
           verbose = 1,
           validation_split = 0.3,
           shuffle = TRUE
)

# Predict Weights -----------------------------------------------------------
set.seed(123)
weights_boost <- xgboost(data = x_train,
                        label = weights_train,
                        eta = 0.3, # learning rate
                        lambda = 0.3, # l2 regularization
                        alpha = 0.3, # l1 regularization
                        verbose = 1,
                        nrounds = 23,
                        early_stopping_rounds = 4, # stop after loss doesnt change for 5 rounds
                        weight = weights_train) # use weights provided for training

# Predict Weights to x_test -----------------------------------------------------
weights_test <- predict(weights_boost, x_test) %>% as.matrix()

# Predict reconstruction error to x_train and x_test ----------------------------
train_recon <- apply((x_train[,1:13] - predict(ae, x_train[,1:13]))^2, 1, sum)
test_recon <- apply((x_test[,1:13] - predict(ae, x_test[,1:13]))^2, 1, sum)

# Add reconstruction error and weights to datasets
x_train <- cbind(x_train, weights_train, train_recon)
x_test <- cbind(x_test, weights_test, test_recon)

# Simple GBM Model ---------------------------------------------------------------------------
# GBM
library(gbm)

# gaussian gbm

y_scaled <- mean(y_train)

set.seed(321)
gbm_gaussian <- gbm.fit(x = x_train,
                     y = as.numeric(y_scaled),
                     w = as.numeric(weights_train),
                     distribution = "gaussian",
                     interaction.depth = 2,
                     n.trees = 100,
                     bag.fraction = 0.5,
                     nTrain = nrow(x_train)*0.8,
                     response.name = 'y')


# boost predictions
boost_predictions <- cbind(test$Index, predict(gbm_gaussian, x_test)*y_mad + y_med) %>% as.data.frame()
colnames(boost_predictions) <- c('Index', 'y')
boost_predictions$Index %<>% as.integer() # submission doesn't accept numeric index
boost_predictions <- boost_predictions[order(boost_predictions$Index),] # submission needs to be in original order

# Export Predictions
write.csv(boost_predictions, '~/Desktop/gresearch/gbm.3.15.17.csv', na = '', row.names = FALSE)
