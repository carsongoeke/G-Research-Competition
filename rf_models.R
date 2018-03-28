# /////////////////////////////////////////////////////////////////////////// 
# G Research Financial Forecasting Competition 
# /////////////////////////////////////////////////////////////////////////// 

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

# Lagged Y ------------------------------------------------------------------
fake_test <- test
fake_test$y <- NA
test_train_combo <- rbind(fake_test, train[,1:16])
test_train_combo <- test_train_combo[order(test_train_combo$Day),]

test_train_combo <- slide(test_train_combo, 
                          Var = 'y', 
                          TimeVar = 'Day', 
                          GroupVar = 'Stock', 
                          NewVar = 'y_lag1',
                          slideBy = -1)
test_train_combo <- slide(test_train_combo, 
                          Var = 'y', 
                          TimeVar = 'Day', 
                          GroupVar = 'Stock', 
                          NewVar = 'y_lag2',
                          slideBy = -2)
test_train_combo <- slide(test_train_combo, 
                          Var = 'y', 
                          TimeVar = 'Day', 
                          GroupVar = 'Stock', 
                          NewVar = 'y_lag3',
                          slideBy = -3)
test_train_combo <- slide(test_train_combo, 
                          Var = 'y', 
                          TimeVar = 'Day', 
                          GroupVar = 'Stock', 
                          NewVar = 'y_lag4',
                          slideBy = -4)

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

# neg exp transform (only seems good fro x1 and x2)
train$neg_exp1 <- exp(-train$x1)
test$neg_exp1 <- exp(-test$x1)
train$neg_exp2 <- exp(-train$x2)
test$neg_exp2 <- exp(-test$x2)

train <- cbind(train, train_log_xs, train_log2_xs, train_square_xs, train_inv_xs, train_inv2_xs, train_cos_xs)
test <- cbind(test, test_log_xs, test_log2_xs, test_square_xs, test_inv_xs, test_inv2_xs, test_cos_xs)
rm(train_log_xs, test_log_xs, train_square_xs, test_square_xs, train_inv_xs, test_inv_xs, train_inv2_xs, test_in2v_xs)

#train %<>% interactions()
#test %<>% interactions()

# Merge with other transforms

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
                                                   mad_y = mad(y)) %>% as.data.frame()
y_stats[is.na(y_stats),] <- 0

x_stats <- train %>% group_by(Stock) %>% summarize(mean_x4 = mean(x4),
                                                   mean_x3E = mean(x3E),
                                                   mean_x0 = mean(x0),
                                                   median_x4 = median(x4),
                                                   median_x3E = median(x3E),
                                                   median_x0 = median(x0),
                                                   sd_x4 = sd(x4),
                                                   sd_x3E = sd(x3E),
                                                   sd_x0 = sd(x0),
                                                   mad_x4 = mad(x4),
                                                   mad_x3E = mad(x3E),
                                                   mad_x0 = mad(x0))


# sparse vars
y_stats$low_y <- (y_stats$median_y < quantile(y_stats$median_y)[2]) %>% as.numeric()
y_stats$high_y <- (y_stats$median_y > quantile(y_stats$median_y)[4]) %>% as.numeric()

# replace na values
y_stats[is.na(y_stats)] <- 0
x_stats[is.na(x_stats)] <- 0

# merge with train and test
train <- merge(train, y_stats, by = 'Stock', all.x = TRUE, sort = FALSE)
train <- merge(train, x_stats, by = 'Stock', all.x = TRUE, sort = FALSE)

test <- merge(test, y_stats, by = 'Stock', all.x = TRUE, sort = FALSE)
test <- merge(test, x_stats, by = 'Stock', all.x = TRUE, sort = FALSE)





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

# merge lagged ys
lagged <- test_train_combo[, c('Day', 'Stock', 'y_lag1', 'y_lag2', 'y_lag3', 'y_lag4')]
train <- merge(train, lagged, by = c('Day', 'Stock'), all.x = TRUE)
test <- merge(test, lagged, by = c('Day', 'Stock'), all.x = TRUE)

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
#mins <- apply(x_train, 2, min)
#maxs <- apply(x_train, 2, max)
#x_train %<>% scale(center = mins, scale = (maxs - mins))
#x_test %<>% scale(center = mins, scale = (maxs - mins))
means <- apply(x_train, 2, mean)
sds <- apply(x_train, 2, sd)
x_train %<>% scale(center = means, scale = sds)
x_test %<>% scale(center = means, scale = sds)

# Reconstruction Error ------------------------------------------------------
ae <- keras_model_sequential()
ae %>% 
  #layer_gaussian_noise(stddev = 0.1, input_shape = c(ncol(x_train))) %>%
  layer_dense(units = 15, activation = 'elu', input_shape = c(ncol(x_train[,1:26]))) %>%
  layer_dense(units = 7, activation = 'elu') %>%
  layer_dense(units = 15, activation = 'elu') %>%
  layer_dense(units = ncol(x_train[,1:26]), activation = 'linear') # using tanh b/c y is bounded

# Compile the model
ae %>% compile(
  loss = 'mse',
  optimizer = 'rmsprop',
  metric = c('kullback_leibler_divergence')
)

# Train the model
ae %>% fit(x_train[,1:26], x_train[,1:26],
           batch_size = 10000,
           epochs = 50,
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
                         nrounds = 50,
                         early_stopping_rounds = 2, # stop after loss doesnt change for 2 rounds
                         weight = weights_train) # use weights provided for training

# Predict Weights to x_test -----------------------------------------------------
weights_test <- predict(weights_boost, x_test) %>% as.matrix()

# Predict reconstruction error to x_train and x_test ----------------------------
train_recon <- apply((x_train[,1:26] - predict(ae, x_train[,1:26]))^2, 1, sum)
test_recon <- apply((x_test[,1:26] - predict(ae, x_test[,1:26]))^2, 1, sum)

# Add reconstruction error and weights to datasets and renormalize
x_train <- cbind(x_train, weights_train, train_recon)
x_test <- cbind(x_test, weights_test, test_recon)

# normalize the data 
means <- apply(x_train, 2, mean)
sds <- apply(x_train, 2, sd)
x_train %<>% scale(center = means, scale = sds)
x_test %<>% scale(center = means, scale = sds)

# get holdout data for stacking at the end
#holdout_ind <- sample(seq(1:nrow(x_train)),0.1*nrow(x_train))
#x_holdout <- x_train[holdout_ind,]
#y_holout <- y_train[holdout_ind,]
#weights_holdout <- weights_train[holdout_ind,]

# Clean up
rm(test_cos_xs, train_cos_xs, train_days, test_days, train_market, test_market, ae, 
   weights_boost, test_inv2_xs, test_log2_xs, y_stats, train_log2_xs)

# RF --------------------------------------------------------
library(ranger)
df_train <- as.data.frame(cbind(y_train, x_train))
colnames(df_train)[1] <- 'y'
df_test <- x_test %>% as.data.frame()
ranger <- ranger(formula = y~.,
                 data = df_train,
                # importance = 'impurity',
                 num.trees = 100,
                 replace = TRUE,
                 mtry = 10,
                 min.node.size = 3,
                 case.weights = weights_train,
                 verbose = TRUE)

#importance <- ranger$variable.importance
#most_imp <- colnames(x_train)[importance > quantile(importance)[4]]

train_preds <- predict(ranger, df_train, predict.all = FALSE, type = 'response')
# Evaluate the model with competition metric (weighted MSE)
wMSE <- sum(weights_train * abs(as.numeric(y_train) - train_preds$predictions))

# Predictions -------------------------------------------------------------------- 
colnames(df_test) <- colnames(df_train)[2:ncol(df_train)]
ranger_preds <- predict(ranger, df_test, predict.all = FALSE, type = 'response')
rf_predictions <- cbind(test$Index, ranger_preds$predictions) %>% as.data.frame()
colnames(rf_predictions) <- c('Index', 'y')
rf_predictions$Index %<>% as.integer() # submission doesn't accept numeric index
rf_predictions <- rf_predictions[order(rf_predictions$Index),] # submission needs to be in original order

# Export Predictions
write.csv(rf_predictions, '~/Desktop/gresearch/ranger_witxstats.csv', na = '', row.names = FALSE)
