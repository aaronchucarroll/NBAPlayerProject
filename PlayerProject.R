# import dataset
players_df <- read.csv("players.csv", na.strings = "-")

set.seed(98)

# years to be used
players_df <- players_df[players_df$draft_year > 1995 & players_df$draft_year < 2015, ]

# select desired attributes
players_df <- players_df[, c("X_id", "name", "career_AST", "career_PTS", "career_TRB", "career_eFG.", "career_FT.", "career_PER", "draft_pick")]

# rename columns 
colnames(players_df) <- c("id", "name", "assists", "points", "rebounds", "effective_FG", "free_throws", "PER", "draft_pick")

# uses a regex to filter out non-numeric characters from draft pick
players_df$draft_pick <- as.numeric(gsub("[^0-9]+", "", players_df$draft_pick))

# check all draft_pick values are not NA
# replace “-” in player-stat columns with mean or median

#Cleaning free throw column
ftMedian <- median(players_df$free_throws, na.rm = TRUE)
players_df$free_throws[is.na(players_df$free_throws)] <- ftMedian

#Cleaning effective field goal column
efgMedian <- median(players_df$effective_FG, na.rm = TRUE)
players_df$effective_FG[is.na(players_df$effective_FG)] <- efgMedian

#Cleaning PER column
perMedian <- median(players_df$PER, na.rm = TRUE)
players_df$PER[is.na(players_df$PER)] <- perMedian

#Removing NA Draft Picks
players_df <- players_df[!is.na(players_df$draft_pick),]

# predict w/ diff. classifiers draft number then flip for draft pick
# 	using bins of draft numbers 

players_df$actualBin <- ifelse(players_df$draft_pick <= 14, 1,
                            ifelse(players_df$draft_pick <= 30, 2, 3))

players_df$binaryBin <- ifelse(players_df$draft_pick <= 30, 1, 0)
players_df$binaryBin <- as.factor(players_df$binaryBin)
#players_df$actualBin <- as.factor(players_df$actualBin)


train_size <- floor(nrow(players_df) * 0.8)
train_idxs <- sample(nrow(players_df), train_size, replace = FALSE)
train <- players_df[train_idxs,]
test <- players_df[-train_idxs,]


#Logistic regression model of binomial family with no weights and 2 bins (good or bad)
biModel <- glm(binaryBin ~ assists + points + rebounds + effective_FG + free_throws + PER, data = train, family = "binomial")

test$binary_prediction <- predict(biModel, newdata  = test, type = "response")
test$binary_prediction <- ifelse(test$binary_prediction >= 0.5, 0, 1)

tabBinary <- table(test$binaryBin, test$binary_prediction)
sum(diag(tabBinary))/sum(tabBinary)

#Weighted binary model

trainScaled <- train
trainScaled$assists <- trainScaled$assists / max(trainScaled$assists)
trainScaled$points <- trainScaled$points / max(trainScaled$points)
trainScaled$rebounds <- trainScaled$rebounds / max(trainScaled$rebounds)
trainScaled$effective_FG <- trainScaled$effective_FG / max(trainScaled$effective_FG)
trainScaled$free_throws <- trainScaled$free_throws / max(trainScaled$free_throws)
trainScaled$PER <- trainScaled$PER / max(trainScaled$PER)

trainScaled$assists <- trainScaled$assists * 15
trainScaled$points <- trainScaled$points * 55
trainScaled$rebounds <- trainScaled$rebounds * 5
trainScaled$effective_FG <- trainScaled$effective_FG * 10
trainScaled$free_throws <- trainScaled$free_throws * 5
trainScaled$PER <- trainScaled$PER * 10

scaledModel <- glm(binaryBin ~ assists + points + rebounds + effective_FG + free_throws + PER, data = train, family = "binomial")
test$scaledPrediction <- predict(biModel, newdata  = test, type = "response")
test$scaledPrediction <- ifelse(test$binary_prediction >= 0.5, 0, 1)

scaledTable <- table(test$binaryBin, test$scaledPrediction)
sum(diag(scaledTable))/sum(scaledTable)



#Gaussian Model
model <- glm(actualBin ~ assists + points + rebounds + effective_FG + free_throws + PER, data = train, family = "gaussian")

test$predicted_bin <- predict(model, newdata = test, type = "response")

test$predicted_bin <- round(test$predicted_bin, 0)

test$predicted_bin <- ifelse(test$predicted_bin == 0, 1, test$predicted_bin)

tab <- table(test$actualBin, test$predicted_bin)
sum(diag(tab))/sum(tab)


# Classification Tree
library(rpart)
tree_model <- rpart(actualBin ~ assists + points + rebounds + effective_FG + free_throws + PER, data = train, method="class")
preds <- predict(tree_model, newdata = test, type="class")
tab <- table(test$actualBin, pred=preds)
sum(diag(tab))/sum(tab)


#GLM Draft Pick Prediction
model <- glm(draft_pick ~ assists + points + rebounds + effective_FG + free_throws + PER, data = train, family = "gaussian")
test$predictedPick <- predict(model, newdata = test, type = "response")
test$predictedPick <- round(test$predictedPick, 0)
test$pickDifference <- test$draft_pick - test$predictedPick
test$pickDifference <- abs(test$pickDifference)
mean(test$pickDifference)

#kknn

library(class)
params <- c("assists", "points", "rebounds", "effective_FG", "free_throws", "PER")
train2 <- train[, params]
test2 <- test[, params]

#Binary Bin Trial
pred = knn(train = train2, test = test2, cl = as.factor(train$binaryBin), k = 3, prob = TRUE)

cm = table(test$binaryBin, pred)

sum(diag(cm))/sum(cm)



#3 Bin trial
pred2 = knn(train = train2, test = test2, cl = as.factor(train$actualBin), k = 3, prob = TRUE)
cm2 = table(test$actualBin, pred2)
sum(diag(cm2))/sum(cm2)


# Naive Bayes
library(e1071)
nb_model <- naiveBayes(train[,3:8], train[,10])
pred = predict(nb_model, test, type = "class")
tab <- table(test$actualBin, pred=preds)
sum(diag(tab))/sum(tab)

