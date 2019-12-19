library(tidyverse)
library(dslabs)
library(caret)

#The exercises in Q1-Q8 work with a simulated dataset for 100 schools. 
#This pre-exercise setup walks you through the code needed to simulate the dataset.

#An education expert is advocating for smaller schools. 
#The expert bases this recommendation on the fact that among the best performing schools,
#many are small schools. Let's simulate a dataset for 100 schools. 
#First, let's simulate the number of students in each school, using the following code:

set.seed(1986)
n <- round(2^rnorm(1000, 8, 1))

#Now let's assign a true quality for each school that is completely independent from size.
#This is the parameter we want to estimate in our analysis. 
#The true quality can be assigned using the following code:

set.seed(1)
mu <- round(80 + 2*rt(1000, 5))
range(mu)
schools <- data.frame(id = paste("PS",1:100),
                      size = n,
                      quality = mu,
                      rank = rank(-mu))

#We can see the top 10 schools using this code: 

schools %>% top_n(10, quality) %>% arrange(desc(quality))

#Now let's have the students in the school take a test. 
#There is random variability in test taking, 
#so we will simulate the test scores as normally distributed with the average determined 
#by the school quality with a standard deviation of 30 percentage points. This code will simulate the test scores:

set.seed(1)
scores <- sapply(1:nrow(schools), function(i){
       scores <- rnorm(schools$size[i], schools$quality[i], 30)
       scores
})
schools <- schools %>% mutate(score = sapply(scores, mean))

#What are the top schools based on the average score? Show just the ID, size, and the average score.

#Report the ID of the top school and average score of the 10th school.

#What is the ID of the top school?

schools %>% top_n(10, score) %>% arrange(desc(score)) %>% select(id, size, score)

#Compare the median school size to the median school size of the top 10 schools based on the score.

#What is the median school size overall?
#What is the median school size of the of the top 10 schools based on the score?

median(schools$size)
schools %>% top_n(10, score) %>% .$size %>% median()

#According to this analysis, it appears that small schools produce better test scores than large schools. 
#Four out of the top 10 schools have 100 or fewer students. 
#But how can this be?
#We constructed the simulation so that quality and size were independent. 
#Repeat the exercise for the worst 10 schools.

#What is the median school size of the bottom 10 schools based on the score?

schools %>% top_n(-10, score)%>% .$size %>% median()

#From this analysis, we see that the worst schools are also small. 
#Plot the average score versus school size to see what's going on. 
#Highlight the top 10 schools based on the true quality. 
#Use a log scale to transform for the size.

#What do you observe?

#The standard error of the score has larger variability when the school is smaller, 
#which is why both the best and the worst schools are more likely to be small.


#We can see that the standard error of the score has larger variability when the school is smaller. 
#This is a basic statistical reality we learned in
#PH125.3x: Data Science: Probability and PH125.4x: Data Science: Inference and Modeling courses! 
#Note also that several of the top 10 schools based on true quality are also in the top 10 schools based on the exam score: 
#schools %>% top_n(10, score) %>% arrange(desc(score))


schools %>% ggplot(aes(size, score)) +
	geom_point(alpha = 0.5) +
	geom_point(data = filter(schools, rank<=10), col = 2) 

#Let's use regularization to pick the best schools. 
#Remember regularization shrinks deviations from the average towards 0. 
#To apply regularization here, we first need to define the overall average for all schools, using the following code:


overall <- mean(sapply(scores, mean))

#Then, we need to define, for each school, how it deviates from that average.

#Write code that estimates the score above the average for each school but dividing by n+alpha instead of n, 
#with n  the schools size and alpha a regularization parameters. Try alpha=25.

#What is the ID of the top school with regularization?
# 91

#What is the regularized score of the 10th school?
#86.9

alpha <- 25
score_reg <- sapply(scores, function(x)  overall + sum(x-overall)/(length(x)+alpha))
schools %>% mutate(score_reg = score_reg) %>%
	top_n(10, score_reg) %>% arrange(desc(score_reg))

#Notice that this improves things a bit. 
#The number of small schools that are not highly ranked is now lower.
#Is there a better ? 
#Find the  that minimizes the RMSE = 1/100 sum (quality-estimate)^2.

#What value of aplha gives the minimum RMSE?

alphas <- seq(10,250)
rmse <- sapply(alphas, function(alpha){
	score_reg <- sapply(scores, function(x) overall+sum(x-overall)/(length(x)+alpha))
	mean((score_reg - schools$quality)^2)
})
plot(alphas, rmse)
alphas[which.min(rmse)]  

#Rank the schools based on the average obtained with the best alpha. 
#Note that no small school is incorrectly included.

#What is the ID of the top school now?
#91

#What is the regularized average score of the 10th school now?
#85.4

alpha <- alphas[which.min(rmse)]  
score_reg <- sapply(scores, function(x)
	overall+sum(x-overall)/(length(x)+alpha))
schools %>% mutate(score_reg = score_reg) %>%
	top_n(10, score_reg) %>% arrange(desc(score_reg))

#A common mistake made when using regularization is shrinking values towards 0 that are not centered around 0. 
#For example, if we don't subtract the overall average before shrinking, 
#we actually obtain a very similar result. 
#Confirm this by re-running the code from the exercise in Q6 but without removing the overall mean.

#What value of aplha gives the minimum RMSE here?
#10

alphas <- seq(10,250)
rmse <- sapply(alphas, function(alpha){
	score_reg <- sapply(scores, function(x) sum(x)/(length(x)+alpha))
	mean((score_reg - schools$quality)^2)
})
plot(alphas, rmse)
alphas[which.min(rmse)]  


