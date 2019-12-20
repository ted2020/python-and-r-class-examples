# Very Basic New York Times Example			          #
###################################################

# What is Supervised Learning
browseURL("http://pan.oxfordjournals.org/content/early/2013/01/21/pan.mps028.full.pdf")

# LOAD THE RTextTools LIBRARY
#install.packages("RTextTools") #Needs to be R-2.14.1 or greater
library(RTextTools)

#CHANGE WORKING DIRECTORY TO YOUR WORKING DIRECTORY
#setwd("/Users/...")

# READ THE DATA from the RTextTools package
# This will be read in differently for your own data

data(NYTimes)
nyt_data <- NYTimes

# [OPTIONAL] SUBSET YOUR DATA TO GET A RANDOM SAMPLE
set.seed(10075)
nyt_data <- nyt_data[sample(1:3000,size=3000,replace=FALSE),]

#Examine the data
class(nyt_data) #make sure it is a data frame object
head(nyt_data) # Look at the first six lines or so
summary(nyt_data) #summarize the data
sapply(nyt_data, class) #look at the class of each column
dim(nyt_data) #Check the dimensions, rows and columns

# Convert Factor to character, text, this is not always the case
nyt_data$Title <- as.character(nyt_data$Title)
nyt_data$Subject <- as.character(nyt_data$Subject)

#What do the codes mean?
#http://www.policyagendas.org/page/topic-codebook

browseURL("http://www.policyagendas.org/page/topic-codebook")

# CREATE A TERM-DOCUMENT MATRIX THAT REPRESENTS WORD FREQUENCIES IN EACH DOCUMENT
# WE WILL TRAIN ON THE Title and Subject COLUMNS
nyt_matrix <- create_matrix(cbind(nyt_data$Title,nyt_data$Subject), language="english",
                            removeNumbers=TRUE, stemWords=TRUE, weighting=tm::weightTfIdf)
nyt_matrix # Sparse Matrix object

########################################
# 	  CORPUS AND CONTAINER CREATION	   #
########################################

# CREATE A CORPUS THAT IS SPLIT INTO A TRAINING SET AND A TESTING SET
# WE WILL BE USING Topic.Code AS THE CODE COLUMN. WE DEFINE A 2000
# ARTICLE TRAINING SET AND A 1000 ARTICLE TESTING SET.
corpus <- create_container(nyt_matrix,nyt_data$Topic.Code,trainSize=1:2600,testSize=2601:3000,virgin=FALSE)
names(attributes(corpus)) #class matrix_container

# Quick look at Document Term Matrix
example_mat <- corpus@training_matrix
example_names <- corpus@column_names
example_mat2 <- as.matrix(example_mat)
colnames(example_mat2) <- example_names
example_mat2[1:10,1:10]
# Look at original data
nyt_data[4,]

##########################################
#			   TRAIN MODELS				 #
##########################################
# THERE ARE TWO METHODS OF TRAINING AND CLASSIFYING DATA.
# ONE WAY IS TO DO THEM AS A BATCH (SEVERAL ALGORITHMS AT ONCE)
system.time(# You can throw this on to see how long it takes

suppressWarnings( #sometimes a nonsensical error will crop up
models <- train_models(corpus, algorithms=c("SVM","MAXENT", "GLMNET"))
)

)

##########################################
# 			  CLASSIFY MODELS		             #
##########################################

results <- classify_models(corpus, models)

##########################################
# VIEW THE RESULTS BY CREATING ANALYTICS #
##########################################
summary(analytics <- create_analytics(corpus, results))

# What objects can we extract from the analytics object?
names(attributes(analytics))

# RESULTS WILL BE REPORTED BACK IN THE analytics VARIABLE/Object
# analytics@algorithm_summary: SUMMARY OF PRECISION, RECALL, F-SCORES, AND ACCURACY SORTED BY TOPIC CODE FOR EACH ALGORITHM
# analytics@label_summary: SUMMARY OF LABEL (e.g. TOPIC) ACCURACY
# analytics@document_summary: RAW SUMMARY OF ALL DATA AND SCORING
# analytics@ensemble_summary: SUMMARY OF ENSEMBLE PRECISION/COVERAGE. USES THE n VARIABLE PASSED INTO create_analytics()

head(analytics@algorithm_summary)
head(analytics@label_summary)


#########################################################################
# Let's look at the documents to see how well the Consensus Score works #
#########################################################################

head(analytics@document_summary)
docs <- analytics@document_summary # Create new dataset object
dim(docs)
table(docs$CONSENSUS_AGREE) # We will want to subset the data to just the 3's

# SUBSET DATA TO JUST THE 3'S
nyt_test <- nyt_data[2601:3000,]
# Combine original document data with machine-learned data
final <- data.frame(nyt_test, docs)
head(final)

# One way to subset
final3 <- final[final$CONSENSUS_AGREE==3,]
dim(final3)
head(final3)

#write.csv(final3, "final_coded_data.csv", row.names=F)

# Unlabeled Data // Hand Code these datas
unlabel12 <- final[final$CONSENSUS_AGREE < 3,]
dim(unlabel12)
head(unlabel12)

#write.csv(unlabel12, "unlabeled_data.csv", row.names=F)

#######################
# VIRGIN DATA EXAMPLE #
#######################

nyt_data_virgin <- nyt_data
nyt_data_virgin$Topic.Code <- NA

# Create Bigger "virgin" dataset
nyt_data_big <- rbind(nyt_data_virgin,nyt_data_virgin,nyt_data_virgin,nyt_data_virgin)
dim(nyt_data_big)

nyt_final <- rbind(nyt_data, nyt_data_big)
dim(nyt_final)

nyt_matrix <- create_matrix(cbind(nyt_final$Title,nyt_final$Subject), language="english",
                            removeNumbers=TRUE, stemWords=TRUE, weighting=tm::weightTfIdf)
nyt_matrix # Sparse Matrix object

########################################
# 	  CORPUS AND CONTAINER CREATION	   #
########################################

# This time set virgin=TRUE
corpus <- create_container(nyt_matrix,nyt_data$Topic.Code,trainSize=1:3000,testSize=3001:nrow(nyt_final),virgin=TRUE)
names(attributes(corpus)) #class matrix_container

system.time(# You can throw this on to see how long it takes

  suppressWarnings( #sometimes a nonsensical error will crop up
    models <- train_models(corpus, algorithms=c("SVM","MAXENT", "GLMNET"))
  )

)

##########################################
# 			  CLASSIFY MODELS		             #
##########################################

results <- classify_models(corpus, models)

# Calculate Summary Statistics
summary(analytics <- create_analytics(corpus, results))

doc_sum <- analytics@document_summary

doc_sum3 <- doc_sum[doc_sum$CONSENSUS_AGREE==3,]
dim(doc_sum3)
