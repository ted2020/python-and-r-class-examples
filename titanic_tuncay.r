
Sys.time()

library(tidyverse)
library(caret)
library(ggplot2)
library(olsrr)
library(car)
library(InformationValue)
library(heuristica)
library(stats)

train <- read.csv('train.csv',stringsAsFactors = F)
test <- read.csv('test.csv',stringsAsFactors = F)
gender_submission <- read.csv('gender_submission.csv',stringsAsFactors = F)

test <- left_join(test,gender_submission, by="PassengerId")

head(train,1)
head(test,1)
head(gender_submission,1)

str(train)
str(test)
str(gender_submission)

anyNA(train)
anyNA(test)
anyNA(gender_submission)

head(train[!complete.cases(train),])
head(test[!complete.cases(test),])

# looks like age data is missing for some
# those missing can be replaced by mean
# but i will skip it for now

# How many unique passengers are there in the training set and the testing set?
n_distinct(train$PassengerId)
n_distinct(test$PassengerId)

# What data is in the training set and what is in the testing set? 
colnames(train)
# head(train)
colnames(test)
# head(test)

# How many people are there in each of the classes (Pclass)? 

# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower

train %>% group_by(Pclass) %>% summarize(n())

test %>% group_by(Pclass) %>% summarize(n())

# How many males and how many females are there in the training set and the testing set?

train %>% group_by(Sex) %>% summarize(n())

test %>% group_by(Sex) %>% summarize(n())

# What is the age distribution of males and females?

head(train %>% group_by(Sex,Age) %>% summarize(n()) %>% arrange(desc(Age)))
train %>% group_by(Sex,Age) %>% ggplot(aes(Sex,Age, color=Pclass)) + geom_jitter() + facet_grid(.~Survived)

head(test %>% group_by(Sex,Age) %>% summarize(n()) %>% arrange(desc(Age)))
test %>% group_by(Sex,Age) %>% ggplot(aes(Sex,Age, color=Pclass)) + geom_jitter() + facet_grid(.~Survived)

# as shown above, some age data is missing, therefore there is a warning, but we can ignore for now

# On average how many siblings/spouse does each person have? 

# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)

head(train %>% group_by(PassengerId) %>% summarize(avg=SibSp/n()) %>% filter(avg>0)) %>% arrange(desc(avg))

head(test %>% group_by(PassengerId) %>% summarize(avg=SibSp/n()) %>% filter(avg>0)) %>% arrange(desc(avg))

# How many unique cabins are there in our training set and test set?

n_distinct(train$Cabin)
n_distinct(test$Cabin)

# Estimate a linear probability model relating the sex of a passenger to the likelihood of their survival. 
#Interpret the results. 

sexsurvival <- lm(Survived ~ Sex, train)
summary(sexsurvival)

# 1 unit change in x1(sex) generates a B1 unit change in y(survived)

# male passengers survival is 0.55313 less than female passengers'

#Estimate a linear probability model relating the class of a passenger to the likelihood of their survival. 
#Interpret the results.
classsurvival <- lm(Survived ~ as.factor(Pclass), train)
summary(classsurvival)

# 1 unit change in x1(Pclass) generates a B1 unit change in y(survived)

# compared to class 1, class 2 passengers survival is 0.15680 less
# compared to class 1, class 3 passengers survival is 0.38727 less

#Estimate a linear probability model relating the age of a passenger to the likelihood of their survival. 
#Interpret the results.

agesurvival <- lm(Survived ~ log(Age), train)
summary(agesurvival)

# 1 unit change in x1(age) generates a B1*100%  change in y(survived)

# 1 unit increase in age will reduce the survival by 8%

#Estimate a linear probability model relating the number of siblings/spouse of a passenger to the likelihood of their survival.
#Interpret the results.

sibspsurvival <- lm(Survived ~ SibSp, train)
summary(sibspsurvival)

# 1 unit change in x1(SibSp) generates a B1 unit change in y(survived)

# 1 unit increase in SibSp will reduce surviving by 0.01559

# Estimate a full linear probability model using as many predictors/features as you can. 
all_in <- lm(Survived ~ Sex+log(Age)+as.factor(Pclass)+SibSp+Parch+Fare+Embarked, train)
summary(all_in)
#anova(all_in, test = 'Chisq')

fit_ols <- ols_step_all_possible(all_in)
head(fit_ols)

best_aic <- arrange(fit_ols,aic)
best_aic[1,]
best_sbic <- arrange(fit_ols,sbic)
best_sbic[1,]

bestmodel <- lm(Survived ~ Sex+log(Age)+as.factor(Pclass)+SibSp, train)
summary(bestmodel) 

vif(bestmodel)

# Make predictions using your model on the training set and generate a confusion matrix. 
predicted <- predict(bestmodel, test)
predicted <- ifelse(predicted > 0.5,1,0)

confusMat <- table(predicted,test$Survived)
confusMat

# What is the accuracy of your model? 
accuracy <- sum(diag(as.matrix(confusMat)))/sum(confusMat)
accuracy

# What is the Type 1 error of your model?

type1 <- confusMat[2,1]/sum(confusMat[2,])
type1

# What is the Type 2 error of your model? 
type2 <- confusMat[1,2]/sum(confusMat[1,])
type2


