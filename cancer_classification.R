#############################
# Project 2: Classification #
#############################

# Dataset

# Breast cancer wisconsin (diagnostic) dataset
# Data Set Characteristics:
#   
#   Number of Instances: 569
# 
# Number of Attributes: 30 numeric, predictive attributes and the class
# 
# Attribute Information:
#   
#   - radius (mean of distances from center to points on the perimeter)
#   - texture (standard deviation of gray-scale values)
#   - perimeter
#   - area
#   - smoothness (local variation in radius lengths)
#   - compactness (perimeter^2 / area - 1.0)
#   - concavity (severity of concave portions of the contour)
#   - concave points (number of concave portions of the contour)
#   - symmetry 
#   - fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error, and "worst" or largest (mean of the three largest values) of these features 
# were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is 
# Radius SE, field 23 is Worst Radius.
# 
# class:
#   - WDBC-Malignant
#   - WDBC-Benign
#
# Missing Attribute Values: None
# 
# Class Distribution: 212 - Malignant (class 0), 357 - Benign (class 1)
# 
# Creator: Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
# 
# Donor: Nick Street
# 
# Date: November, 1995


# Importing the dataset:
data <- read.csv("data.csv")
View(data)

#class as factor
data$class <- factor(data$class)
str(data)

# (a) Splitting the Dataset into Training/Test Sets
library(caret)

set.seed(222)
inTraining   <- createDataPartition(data$class, p=0.8, list = FALSE)
training     <- data[ inTraining,]
testing      <- data[-inTraining,]


print("Proportion in whole data set:")
round(prop.table(table(data$class)),2)
print("Proportion in training data set:")
round(prop.table(table(training$class)),2)
print("Proportion in validation set:")
round(prop.table(table(testing$class)),2)
print("Overall split:")
paste("Training set: ", round(nrow(training)/nrow(data),2)*100, "%", sep = '')
paste("Testing set: " , round(nrow(testing)/nrow(data),2)*100, "%", sep = '')

# (b) Scaling the Training and Test Sets 
training <-  cbind(scale(training[-31]), training[31])
testing  <-  cbind(scale(testing[-31]), testing[31])

View(training)
View(testing)
# (c) Feature Selection
rfeCtrl   <- rfeControl(functions = rfFuncs, method = "repeatedcv", number = 6, repeats = 3)


set.seed(111)
rfeProfile <- rfe(training[,-31],      #train set without the class
                  training[,31],       #class vector
                  sizes=c(1:30),       #subset of features -all
                  rfeControl=rfeCtrl)  #controls

# summarizing results
print(rfeProfile)
plot(rfeProfile, type = c("g", "o"))

# listing of chosen features
selected_features      <- predictors(rfeProfile)
selected_features_top5 <- selected_features[1:5]

print("Recursive Feature Elimination with Random Forest - selected features:")
selected_features
print("Recursive Feature Elimination with Random Forest - top 5 selected features:")
selected_features_top5


# (d) fiting two different Classification models to the Training Set and predicting the unseen Test Set
# - Using the subset of features deemed optimal

#final subset used for training
training <- training[,c(selected_features_top5, "class")]

#training scheme
trainControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

#hyper parameter k for KNN grid
knnGrid <- expand.grid(k = c(seq(3,30)))

set.seed(777)
randomForestModel <- train(class ~ .,
                           data     = training,
                           method   = "rf",
                           control  = trainControl)

randomForestModel

set.seed(777)
knnModel          <- train(class ~ .,
                           data = training,
                           method   = "knn",
                           tuneGrid = knnGrid,
)

knnModel

#########predictions
rnfPredict <- predict(randomForestModel, testing[,-31])
knnPredict <- predict(knnModel, testing[,-31])

# (e) Confusion Matrix for each of the two classifiers predictions on the Test set
#Random Forest
table(testing$class, rnfPredict)

#KNN
table(testing$class, knnPredict)



#plots
library(yardstick)

cm1 <- data.frame(
  actual = testing$class,
  predicted = rnfPredict
)

cm1 <- conf_mat(cm1, predicted, actual)

autoplot(cm1, type = "heatmap") +
  scale_fill_gradient(low = "pink", high = "cyan") +
  labs(title = "Random Forest Confusion Matrix", x="Predicted", y="Actual") +
  scale_x_discrete(position = "top") 



cm2 <- data.frame(
  actual = testing$class,
  predicted = knnPredict
)

cm2 <- conf_mat(cm2, predicted, actual)

autoplot(cm1, type = "heatmap") +
  scale_fill_gradient(low = "pink", high = "cyan") +
  labs(title = "KNN Confusion Matrix", x="Predicted", y="Actual") +
  scale_x_discrete(position = "top") 


# (f) Performance of the two classification models
# What is the:
# - Accuracy
# - Precision
# - Recall
# - F1-Score
rnfCm <- confusionMatrix(testing$class, rnfPredict, mode = "everything", positive = "1")
knnCm <- confusionMatrix(testing$class, knnPredict, mode = "everything", positive = "1")


rnfCmMetrics <- c(rnfCm$overall["Accuracy"],rnfCm$byClass["Precision"],rnfCm$byClass["Recall"],rnfCm$byClass["F1"])
knnCmMetrics <- c(knnCm$overall["Accuracy"],knnCm$byClass["Precision"],knnCm$byClass["Recall"],knnCm$byClass["F1"])

print("Selected metrics for Random Forest Model:")
print(rnfCmMetrics)

print("Selected metrics for KNN Model:")
print(knnCmMetrics)


df <- data.frame(rnfCmMetrics, knnCmMetrics)
df

rnfCm
knnCm
#Summary
#judging purely by evaluation metrics both models perform quite the same honestly
#Although metrics suggest models are quite the same I cant get rid of the feeling that due to repeated cross validation
#Process used in Random Forest and the nature of algorithm (many trees and majority vote) the generalization if it is better
#I had different results I cant reproduce now where whole performance was higher and KNN was outperforming Random Forest
#Trying different seeds was affecting KNN more often so to say

#Random Forest has also slight edge over KNN because of sensitivity which is 1 percentage point higher
#We are dealing with diagnosis of breast cancer and we would like to have sensitivity metric better as we may potantially save lives
#I am leaning more towards Random Forest as a more reliable and more sensitive predictor of a breast cancer
