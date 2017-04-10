################
# Cadreon Data Analyst Application
# R Task

my.titanic = read.csv("titanic.csv")
my.predict = read.csv("predict.csv")

# Given the titanic.csv and predict.csv datasets:
# 1.	Using the titainc.csv dataset, create two partitions: train with 70% of the total records and test with the remaining 30%.

samp = sample(1:891,size = 891*0.7)


train = my.titanic[samp,]
test = my.titanic[-samp,]

# 2.	Before training a model, perform any variable transformation if necessary and briefly explain why.

# Data cleaning needs to be performed to remove any errors and missing values. Factor values and strings may also need to be converted to numerical values in order to be used in the modelling.

str(my.titanic)

# The variable Age contains non-numeric values, which need to be corrected if the data is to be used in the model. In this case, I will set all NA values to the median age.

my.titanic$Age[is.na(my.titanic$Age)] = median(my.titanic$Age, na.rm = TRUE)


# Sex of passengers is a factor. In order to perform modelling on sex as a feature, this need to be converted to a number, 1 = Female, 2 = Male

my.titanic$Sex <- as.numeric(my.titanic$Sex)

# Embarked is also a factor, we convert 1 = "", 2 = "C", 3 = "Q", 4 = "S"

my.titanic$Embarked = as.numeric(my.titanic$Embarked)

# Split cleaned data into train and test
# train = my.titanic[samp,]
# test = my.titanic[-samp,]


# 3.	Use the package rpart to train a decision tree model on the train partition, using the column "Survived" as the outcome.


# Recursive Partitioning and Regression Trees
library(rpart)

fit = rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train)

# fit = rpart(Survived ~ Pclass + Sex + Age, data=train)

# You can plot the regression tree using the following commands
plot(fit)
text(fit, cex = 0.7,pos = 2)


# 4.	After training the model, report the importance of each variable.

# The R command summary(fit) returns a number of details of the rpart model. The importance of each variable can be shown in the following component

fit$variable.importance

#       Sex      Fare    Pclass       Age     Parch     SibSp  Embarked 
# 44.896279 25.600646 12.210541  9.240344  5.799570  5.631461  3.221121

# Sex appears to be the most important variable in the model, followed by Fare, Pclass and Age

# 5.	Now, test the model on the test partition.

# Use the predict function to test the model on the test data

pre = predict(fit,newdata = test)

# predict comes back with the probability value of predictions, so we need to convert this into a survived/not survived value as follows:

pre =ifelse(pre<0.5, 0, 1)

# We can do an initial summary of results in the following tables

table(pre)
#   0   1 
# 189  79 
table(test$Survived)
#   0   1 
# 180  88 

# 6.	After testing the model report the overall model accuracy and misclassification rates using a confusion matrix.

# The confusion matrix displays a cross-tabulation of predicted and actual values

library(caret)

temp = confusionMatrix(reference = test$Survived,data = pre)
temp$table
#           Reference
# Prediction   0   1
#          0 162  27
#          1  18  61

# From this table we can see that 189 passengers are predicted to die, and 79 to survive.
# However, the test data shows that 180 die, and 88 survive, with 45 passengers misclassified as
# either false positive or false negative.

misclassified = (pre != test$Survived)
sum(misclassified)
# [1] 43

misclassError = mean(misclassified)
print(paste(("Accuracy = "), 1-misclassError))

# [1] "Accuracy =  0.832089552238806"

# 7.	Use your model to predict the survival of passengers in the predict.csv dataset (score the file) and save your results in the same file.

# We also have to remove NA values and convert factors into numeric variables for the predict.csv data

my.predict$Age[is.na(my.predict$Age)] = median(my.predict$Age, na.rm = TRUE)
my.predict$Sex = as.numeric(my.predict$Sex)
my.predict$Embarked = as.numeric(my.predict$Embarked)

# Run the prediction on the data, and adjust the resulting probabilities into 0 and 1 values

finalpre = predict(fit,newdata = my.predict)
PredSurvived = ifelse(finalpre<0.5, 0, 1)

table(PredSurvived)
# PredSurvived
#   0   1 
# 173 127 

# We can then add the results to the predict data and write it into a .csv file

my.predict = cbind(my.predict,PredSurvived)

write.csv(my.predict, "MyPredictions.csv", row.names = FALSE)

# 8.	Upload your code to a public repository (Github is recommended) and share the project link.
