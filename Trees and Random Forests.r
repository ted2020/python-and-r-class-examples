library(tidyverse)
library(dslabs)
library(caret)

#Create a simple dataset where the outcome grows 0.75 units on average for every increase in a predictor, using this code:

library(rpart)
n <- 1000
sigma <- 0.25
x <- rnorm(n, 0, 1)
y <- 0.75 * x + rnorm(n, 0, sigma)
dat <- data.frame(x = x, y = y)

#Which code correctly uses rpart to fit a regression tree and saves the result to fit?

fit <- rpart(y ~ ., data = dat)

#Which of the following plots correctly shows the final tree obtained in Q1?

fit

plot(fit)
text(fit)

#Below is most of the code to make a scatter plot of y versus x along with the predicted values based on the fit.

dat %>% 
	mutate(y_hat = predict(fit)) %>% 
	ggplot() +
	geom_point(aes(x, y)) +
	BLANK

#Which line of code should be used to replace #BLANK in the code above?

    dat %>% 
        mutate(y_hat = predict(fit)) %>% 
        ggplot() +
        geom_point(aes(x, y)) +
        geom_step(aes(x, y_hat), col=2)

#Now run Random Forests instead of a regression tree using randomForest 
#from the __randomForest__ package, and remake the scatterplot with the prediction line. 
#Part of the code is provided for you below.



library(randomForest)
fit <- randomForest(y ~ x, data = dat)   #BLANK
dat %>% 
	mutate(y_hat = predict(fit)) %>% 
	ggplot() +
	geom_point(aes(x, y)) +
	geom_step(aes(x, y_hat), col = 2)

#What code should replace #BLANK in the provided code?

#Use the plot function to see if the Random Forest from Q4 has converged or if we need more trees.

#Which is the correct plot to assess whether the Random Forest has converged?

plot(fit)

#It seems that the default values for the Random Forest result in an estimate that is too flexible (unsmooth). 
#Re-run the Random Forest but this time with a node size of 50 and a maximum of 25 nodes. Remake the plot.

#Part of the code is provided for you below.

library(randomForest)
fit <-randomForest(y ~ x, data = dat, nodesize = 50, maxnodes = 25)     #BLANK
dat %>% 
	mutate(y_hat = predict(fit)) %>% 
	ggplot() +
	geom_point(aes(x, y)) +
	geom_step(aes(x, y_hat), col = 2)

#What code should replace #BLANK in the provided code?


