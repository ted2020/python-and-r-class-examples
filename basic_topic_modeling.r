# Basic Topic Modeling

# install.packages(c("tm", "RTextTools", "descr", "ggplot2", "topicmodels"))
library(tm)
library(RTextTools)
library(descr); library(ggplot2)
library(topicmodels)

# Directory
setwd("~/Dropbox/...")

##########################
# Read in Kerry Speeches #
##########################

obama <- scan("obama_2012_eg.txt", what="raw", sep="\n")

# Then Text Stuff....

o_corp <- VCorpus(VectorSource(obama))
inspect(o_corp[1:4])

# Create DTM for Topic Model
o_immig4 <- tm_map(o_corp, removeNumbers)
o_immig4 <- tm_map(o_corp, removePunctuation)
o_immig4 <- tm_map(o_corp , stripWhitespace)
o_immig4 <- tm_map(o_corp, content_transformer(tolower))
o_immig4 <- tm_map(o_corp , removeWords, stopwords("english"))

o_immig_dtm <- DocumentTermMatrix(o_immig4)

# Set up the Topic Model
k <- 4
SEED <- 2010
num_terms <- 10
lda_immig4 <- LDA(o_immig_dtm, k = k, control = list(seed = SEED))
# Evlalaute terms
lda_terms4 <- terms(lda_immig4, num_terms)
lda_terms4
xtable::xtable (lda_terms4, caption="Terms describing topics in four category fit")
