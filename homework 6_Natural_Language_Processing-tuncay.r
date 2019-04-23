
library(tidytext)
library(dplyr)
library(tm)
library(NLP)
library(text2vec)
library(stringr)
library(wordcloud)
library(qdapRegex)
library(RColorBrewer)

instagram_comments <- read.csv('instagram_comments_isle_of_dogs_cleaned_filtered.csv')
reddit_comments <- read.csv('reddit_isleofdogs_trailer1_cleaned.csv')
youtube_comments <- read.csv('youtube_isle_of_dogs_cleaned.csv')

length(instagram_comments$clean_comment)
#as.list(instagram_comments$clean_comment)

gsub("[A-Za-z0-9_:@'\"\"&!?'']", "", instagram_comments$clean_comment, perl=T) 
# this finds all unwanted characters, so that i can remove them

inst <- instagram_comments$clean_comment
inst2 <- instagram_comments$clean_comment_two

inst2 <- gsub("[^[:alnum:]:'@_]"," ", inst)  # this keeps all (@) and (') and (:), remove all other special characters
# to keep the chracters only, just remove @ and ' and :
#  @ refers to the users, and can be useful to keep them, in case i need to extract user names
#  ' refers to it's (grammarly correct), not it s
#  : and _ refer to emojis , i can differentiate emojis from the text.

inst2 <- gsub("â","`",inst2)  # this character used for punctuation
inst2 <- gsub("ï","",inst2)
inst2 <- gsub("â","",inst2)

inst2 <- gsub("youtube","",inst2)
inst2 <- gsub("youre","",inst2) # this is not included in stop words
inst2 <- gsub("http","",inst2)
inst2 <- gsub("https","",inst2)
inst2 <- gsub("www","",inst2)
inst2 <- gsub("com","",inst2)
inst2 <- gsub("ive","",inst2) # i've
inst2 <- gsub("ain","",inst2) #ain't
inst2 <- gsub("till","",inst2) # until
inst2 <- gsub("thats","",inst2) # that's
inst2 <- gsub("youtu","",inst2)
inst2 <- gsub("looks","",inst2) # this word doesnt reflect any sentiment, but somehow got picked up
inst2 <- gsub("wikipedia","",inst2)
inst2 <- gsub("org","",inst2)
inst2 <- gsub("whos","",inst2)

# these words removed because they are out context, therefore dont provide any meaning
inst2 <- gsub("dont","",inst2)
inst2 <- gsub("can","",inst2)
inst2 <- gsub("whos","",inst2)
inst2 <- gsub("just","",inst2)
inst2 <- gsub("one","",inst2)
inst2 <- gsub("stop","",inst2)


    # "the" is not included here, because it's a part of stop words

inst2 <- gsub("\\b\\w{1,2}\\b", "", inst2) # this removes 1 and 2 letter words
inst2 <- tolower(inst2)
inst2 <- str_squish(inst2)

instagram_comments  <- instagram_comments %>% mutate(clean_comment_two=inst2)
head(instagram_comments,1)

head(cbind(after=as.character(inst2),before=as.character(inst)))

# this finds all the user names in instagram comments
user_names_instagram <- str_subset(unlist(str_split(inst, pattern = " ")), "@")
head(as.list(user_names_instagram))

# this finds all the emojis in instagram comments
emojis_instagram <- str_subset(unlist(str_split(inst, pattern = " ")), ":")
head(as.list(emojis_instagram))

length(reddit_comments$clean_comment)
#head(reddit_comments$clean_comment,1)

#gsub("[A-Za-z0-9_:@'\"\"&!?'']", "", red, perl=T) 
# this finds all unwanted characters, so that i can remove them
# but since there are too many of them,
# therefore i will remove all of them

red <- reddit_comments$clean_comment
red2 <- reddit_comments$clean_comment_two

red2 <- gsub("'","", red)
red2 <- gsub("[^[A-Za-z]"," ", red2)  # keep characters only

red2 <- gsub("youtube","",red2)
red2 <- gsub("youre","",red2)
red2 <- gsub("http","",red2)
red2 <- gsub("https","",red2)
red2 <- gsub("www","",red2)
red2 <- gsub("com","",red2)
red2 <- gsub("ive","",red2) # i've
red2 <- gsub("ain","",red2) #ain't
red2 <- gsub("till","",red2) # until
red2 <- gsub("thats","",red2) # that's
red2 <- gsub("youtu","",red2)
red2 <- gsub("looks","",red2)
red2 <- gsub("wikipedia","",red2)
red2 <- gsub("org","",red2)
red2 <- gsub("whos","",red2)

# these words removed because they are out context, therefore dont provide any meaning
red2 <- gsub("dont","",red2)
red2 <- gsub("can","",red2)
red2 <- gsub("whos","",red2)
red2 <- gsub("just","",red2)
red2 <- gsub("one","",red2)
red2 <- gsub("stop","",red2)

    # "the" is not included here, because it's a part of stop words

red2 <- gsub("\\b\\w{1,2}\\b", "", red2)
red2 <- tolower(red2)
red2 <- str_squish(red2)

reddit_comments  <- reddit_comments %>% mutate(clean_comment_two=red2)
head(reddit_comments,1)


head(cbind(after=as.character(red2),before=as.character(red)))

# this finds all the links in reddit comments
#str_subset(unlist(str_split(red, pattern = " ")), "http://")

links_reddit <- as.data.frame((str_subset(unlist(rm_url(red,extract = TRUE)),"http:")))
names(links_reddit) <- c("links reddit")
head(links_reddit)

length(youtube_comments$clean_comment)
#head(youtube_comments$clean_comment,1)

#gsub("[A-Za-z0-9_:@'\"\"&!?'']", "", youtube, perl=T) 
# this finds all unwanted characters, so that i can remove them
# but since there are too many of them,
# therefore i will remove all of them

youtube <-youtube_comments$clean_comment
youtube2 <- youtube_comments$clean_comment_two

youtube2 <- gsub("'","", youtube)
youtube2 <- gsub("[^[A-Za-z]"," ", youtube2)  

youtube2 <- gsub("youtube","",youtube2)
youtube2 <- gsub("youre","",youtube2)
youtube2 <- gsub("http","",youtube2)
youtube2 <- gsub("https","",youtube2)
youtube2 <- gsub("www","",youtube2)
youtube2 <- gsub("com","",youtube2)
youtube2 <- gsub("ive","",youtube2) # i've
youtube2 <- gsub("ain","",youtube2) #ain't
youtube2 <- gsub("till","",youtube2) # until
youtube2 <- gsub("thats","",youtube2) # that's
youtube2 <- gsub("youtu","",youtube2)
youtube2 <- gsub("looks","",youtube2)
youtube2 <- gsub("wikipedia","",youtube2)
youtube2 <- gsub("org","",youtube2)
youtube2 <- gsub("whos","",youtube2)

# these words removed because they are out context, therefore dont provide any meaning
youtube2 <- gsub("dont","",youtube2)
youtube2 <- gsub("can","",youtube2)
youtube2 <- gsub("whos","",youtube2)
youtube2 <- gsub("just","",youtube2)
youtube2 <- gsub("one","",youtube2)
youtube2 <- gsub("stop","",youtube2)

    # "the" is not included here, because it's a part of stop words

youtube2 <- gsub("\\b\\w{1,2}\\b", "", youtube2)
youtube2 <- tolower(youtube2)
youtube2 <- str_squish(youtube2)

youtube_comments  <- youtube_comments %>% mutate(clean_comment_two=youtube2)
head(youtube_comments,1)

head(cbind(after=as.character(youtube2),before=as.character(youtube)))

data('stop_words')
table(stop_words$lexicon)
head(stop_words)
length(stop_words$word)
n_distinct(stop_words$word)
n_distinct(stop_words$lexicon)
class(stop_words$lexicon)
levels(as.factor(as.character(stop_words$lexicon)))

which(stop_words$lexicon == 'snowball')[1:10]

stop_words_for_removal <- as.vector((stop_words %>% filter(lexicon=="snowball"))[1])
as.character(stop_words_for_removal)

inst2_tm <- VCorpus(VectorSource(inst2))
red2_tm <- VCorpus(VectorSource(red2))
youtube2_tm <- VCorpus(VectorSource(youtube2))
getTransformations()

inst2_tm <- tm_map(inst2_tm,removeWords,stop_words_for_removal$word)
red2_tm <- tm_map(red2_tm,removeWords,stop_words_for_removal$word)
youtube2_tm <- tm_map(youtube2_tm,removeWords,stop_words_for_removal$word)

inst2_tm <- tm_map(inst2_tm,stripWhitespace)
red2_tm <- tm_map(red2_tm,stripWhitespace)
youtube2_tm <- tm_map(youtube2_tm,stripWhitespace)

# i kept some punctuation above to differentiate quotations and usernames and emojis and links, but now i'm removing them
inst2_tm <- tm_map(inst2_tm,removePunctuation)
red2_tm <- tm_map(red2_tm,removePunctuation)
youtube2_tm <- tm_map(youtube2_tm,removePunctuation)

inspect(inst2_tm[[10]])

inspect(red2_tm[[13]])

inspect(youtube2_tm[[5]])

NGramTokenizer <- function(x) {
  unlist(lapply(ngrams(words(x), GRAMS), paste, collapse = " "),
  use.names = FALSE)
}

GRAMS <- 1
NGramTokenizer(inst2_tm[[1]])
NGramTokenizer(red2_tm[[1]])
NGramTokenizer(youtube2_tm[[1]])

GRAMS <- 2
NGramTokenizer(inst2_tm[[1]])
NGramTokenizer(red2_tm[[1]])
NGramTokenizer(youtube2_tm[[1]])

GRAMS <- 3
NGramTokenizer(inst2_tm[[1]])
NGramTokenizer(red2_tm[[1]])
NGramTokenizer(youtube2_tm[[1]])

GRAMS <- 1
inst2_dtm_1 <- DocumentTermMatrix(inst2_tm,control = list(tokenize=NGramTokenizer))
inst2_dtm_1 <- removeSparseTerms(inst2_dtm_1,0.99)
dim(inst2_dtm_1)
head(inst2_dtm_1$dimnames$Terms)
tail(inst2_dtm_1$dimnames$Terms)

GRAMS <- 2
inst2_dtm_2 <- DocumentTermMatrix(inst2_tm,control = list(tokenize=NGramTokenizer))
inst2_dtm_2 <- removeSparseTerms(inst2_dtm_2,0.997)
dim(inst2_dtm_2)
head(inst2_dtm_2$dimnames$Terms)
tail(inst2_dtm_2$dimnames$Terms)

GRAMS <- 3
inst2_dtm_3 <- DocumentTermMatrix(inst2_tm,control = list(tokenize=NGramTokenizer))
inst2_dtm_3 <- removeSparseTerms(inst2_dtm_3,0.999)
dim(inst2_dtm_3)
head(inst2_dtm_3$dimnames$Terms)
tail(inst2_dtm_3$dimnames$Terms)

GRAMS <- 1
red2_dtm_1 <- DocumentTermMatrix(red2_tm,control = list(tokenize=NGramTokenizer))
red2_dtm_1 <- removeSparseTerms(red2_dtm_1,0.99)
dim(red2_dtm_1)
head(red2_dtm_1$dimnames$Terms)
tail(red2_dtm_1$dimnames$Terms)

GRAMS <- 2
red2_dtm_2 <- DocumentTermMatrix(red2_tm,control = list(tokenize=NGramTokenizer))
red2_dtm_2 <- removeSparseTerms(red2_dtm_2,0.997)
dim(red2_dtm_2)
head(red2_dtm_2$dimnames$Terms)
tail(red2_dtm_2$dimnames$Terms)

GRAMS <- 3
red2_dtm_3 <- DocumentTermMatrix(red2_tm,control = list(tokenize=NGramTokenizer))
red2_dtm_3 <- removeSparseTerms(red2_dtm_3,0.999)
dim(red2_dtm_3)
head(red2_dtm_3$dimnames$Terms)
tail(red2_dtm_3$dimnames$Terms)

GRAMS <- 1
youtube2_dtm_1 <- DocumentTermMatrix(youtube2_tm,control = list(tokenize=NGramTokenizer))
youtube2_dtm_1 <- removeSparseTerms(youtube2_dtm_1,0.99)
dim(youtube2_dtm_1)
head(youtube2_dtm_1$dimnames$Terms)
tail(youtube2_dtm_1$dimnames$Terms)

GRAMS <- 2
youtube2_dtm_2 <- DocumentTermMatrix(youtube2_tm,control = list(tokenize=NGramTokenizer))
youtube2_dtm_2 <- removeSparseTerms(youtube2_dtm_2,0.997)
dim(youtube2_dtm_2)
head(youtube2_dtm_2$dimnames$Terms)
tail(youtube2_dtm_2$dimnames$Terms)

GRAMS <- 3
youtube2_dtm_3 <- DocumentTermMatrix(youtube2_tm,control = list(tokenize=NGramTokenizer))
youtube2_dtm_3 <- removeSparseTerms(youtube2_dtm_3,0.999)
dim(youtube2_dtm_3)
head(youtube2_dtm_3$dimnames$Terms)
tail(youtube2_dtm_3$dimnames$Terms)

inst2_dtm_freq_1 <- colSums(as.matrix(inst2_dtm_1), na.rm = T)
inst2_dtm_freq_1 <- sort(inst2_dtm_freq_1,decreasing = T)
inst2_dtm_freq_1[1:20]
#barplot(inst2_dtm_freq_1[1:20])

inst2_dtm_freq_2 <- colSums(as.matrix(inst2_dtm_2), na.rm = T)
inst2_dtm_freq_2 <- sort(inst2_dtm_freq_2,decreasing = T)
inst2_dtm_freq_2[1:20]
#barplot(inst2_dtm_freq_2[1:20])

inst2_dtm_freq_3 <- colSums(as.matrix(inst2_dtm_3), na.rm = T)
inst2_dtm_freq_3 <- sort(inst2_dtm_freq_3,decreasing = T)
inst2_dtm_freq_3[1:20]
#barplot(inst2_dtm_freq_3[1:20])

red2_dtm_freq_1 <- colSums(as.matrix(red2_dtm_1), na.rm = T)
red2_dtm_freq_1 <- sort(red2_dtm_freq_1,decreasing = T)
red2_dtm_freq_1[1:20]
#barplot(red2_dtm_freq_1[1:20])

red2_dtm_freq_2 <- colSums(as.matrix(red2_dtm_2), na.rm = T)
red2_dtm_freq_2  <- sort(red2_dtm_freq_2,decreasing = T)
red2_dtm_freq_2[1:20]
#barplot(red2_dtm_freq_2[1:20])

red2_dtm_freq_3 <- colSums(as.matrix(red2_dtm_3), na.rm = T)
red2_dtm_freq_3 <- sort(red2_dtm_freq_3,decreasing = T)
red2_dtm_freq_3[1:20]
#barplot(red2_dtm_freq_3[1:20])

youtube2_dtm_freq_1 <- colSums(as.matrix(youtube2_dtm_1), na.rm = T)
youtube2_dtm_freq_1 <- sort(youtube2_dtm_freq_1,decreasing = T)
youtube2_dtm_freq_1[1:20]
#barplot(youtube2_dtm_freq_1[1:20])

youtube2_dtm_freq_2 <- colSums(as.matrix(youtube2_dtm_2), na.rm = T)
youtube2_dtm_freq_2 <- sort(youtube2_dtm_freq_2,decreasing = T)
youtube2_dtm_freq_2[1:20]
#barplot(youtube2_dtm_freq_2[1:20])

youtube2_dtm_freq_3 <- colSums(as.matrix(youtube2_dtm_3), na.rm = T)
youtube2_dtm_freq_3 <- sort(youtube2_dtm_freq_3,decreasing = T)
youtube2_dtm_freq_3[1:20]
#barplot(youtube2_dtm_freq_3[1:20])

colorlist = c("red","blue","green","red","pink","orange","grey","black","brown","navy","magenta","purple")
wordcloud(names(inst2_dtm_freq_1),inst2_dtm_freq_1, random.order = F, random.color = T, colors = colorlist,scale=c(2,.25))
wordcloud(names(inst2_dtm_freq_2),inst2_dtm_freq_2, random.order = F, random.color = T, colors = colorlist,scale=c(2,.25))
wordcloud(names(inst2_dtm_freq_3),inst2_dtm_freq_3, random.order = F, random.color = T, colors = colorlist,scale=c(2,.25))

wordcloud(names(red2_dtm_freq_1),red2_dtm_freq_1, random.order = F, random.color = T, colors = colorlist,scale=c(2,.25))
wordcloud(names(red2_dtm_freq_2),red2_dtm_freq_2, random.order = F, random.color = T, colors = colorlist,scale=c(2,.25))
wordcloud(names(red2_dtm_freq_3),red2_dtm_freq_3, random.order = F, random.color = T, colors = colorlist,scale=c(2,.25))

wordcloud(names(youtube2_dtm_freq_1),youtube2_dtm_freq_1, random.order = F, random.color = T, colors = colorlist,scale=c(2,.25))
wordcloud(names(youtube2_dtm_freq_2),youtube2_dtm_freq_2, random.order = F, random.color = T, colors = colorlist,scale=c(2,.25))
wordcloud(names(youtube2_dtm_freq_3),youtube2_dtm_freq_3, random.order = F, random.color = T, colors = colorlist,scale=c(2,.25))

bing <- sentiments %>% filter(lexicon == "bing") %>% dplyr::select(word,
sentiment)

head(bing)
sort(table(bing$sentiment))

tokens <- colnames(inst2_dtm_1)
tokenMat <- cbind.data.frame(inst2_dtm_1$i,inst2_dtm_1$v,tokens[inst2_dtm_1$j])
colnames(tokenMat) <- c("id","freq","word")
tokenMat$id <- as.character(tokenMat$id)
tokenMat$freq <- as.numeric(tokenMat$freq)
tokenMat$word <- as.character(tokenMat$word)
head(tokenMat)
dim(tokenMat)
word_count <- tapply(tokenMat$freq,tokenMat$id,sum)
tokenMat$n <- word_count[match(tokenMat$id,names(word_count))]

# i am able to tokenize 214 words from instagram comments about the movie Isle of Dogs
# for that 214, bing sentiment is able to match 30 for 85 rows of all clean_comment_two
# the randomized variable based on the movie's IMDB rating is taken as the predictor
# when actual tokenized word match run against the predictor,
# table outlines the overall match

by_id_sentiment_bing <- inner_join(bing,tokenMat,by="word")

# some users appear more than once.
# if a user has positive ranking,
# he/she will have positive ranking, no matter how many times the token catches the same person
# so, we can get rid of the same person's appereance that is more than once.
# one problem arises with this idea is, 
# what if the words extracted provide two contradicting sentiments.
# but, for the sake of this assignment, i will ignore this.

by_id_sentiment_bing.unique <- by_id_sentiment_bing[!duplicated(by_id_sentiment_bing$id),]
dim(by_id_sentiment_bing.unique)
head(by_id_sentiment_bing.unique)

count_sentiment_by_id <- matrix(NA,length(instagram_comments$clean_comment_two),length(unique(bing$sentiment)))
colnames(count_sentiment_by_id) <- unique(bing$sentiment)

for(i in 1:nrow(count_sentiment_by_id)){
    stp <- by_id_sentiment_bing.unique[by_id_sentiment_bing.unique$id == i,]
    sub_mat <- matrix(0,1,ncol(count_sentiment_by_id))
    colnames(sub_mat) <- colnames(count_sentiment_by_id)
    
  for(j in 1:nrow(stp)){
    sub_mat[1,(stp$sentiment[j-i])] <- sub_mat[1,(stp$sentiment[j-i])] + stp$freq[j]
  }
    
  count_sentiment_by_id[i,] <- sub_mat[1,]
}
count_sentiment_by_id<- as.data.frame(count_sentiment_by_id)
head(count_sentiment_by_id)
length(count_sentiment_by_id$negative)
c(negative=sum(count_sentiment_by_id$negative),positive=sum(count_sentiment_by_id$positive))


# since this is for the movie Isle of Dogs
# and its IMDB rating is 7.9 by 100000 people
# meaning that only 21 percent could be taken as negative and the rest is positive

n <- length(instagram_comments$clean_comment_two)

sentiment_hat <- rbinom(n, 1, 0.79) # 1 for positive, 0 for negative

sentiment_hat <- as.data.frame(sentiment_hat)
head(sentiment_hat)

dim(sentiment_hat)
sum(sentiment_hat$sentiment_hat)


classification_bing <- ifelse(count_sentiment_by_id$positive > count_sentiment_by_id$negative, 1, 0)
confusion_matrix_bing <- table(sentiment_hat$sentiment_hat,classification_bing)
confusion_matrix_bing

##Overall Accuracy of Bing Lexicon vs sentiment_hat
accuracy <- sum(diag(confusion_matrix_bing))/sum(confusion_matrix_bing)
print('--Overall Accuracy--')
accuracy

tokens <- colnames(red2_dtm_1)
tokenMat <- cbind.data.frame(red2_dtm_1$i,red2_dtm_1$v,tokens[red2_dtm_1$j])
colnames(tokenMat) <- c("id","freq","word")
tokenMat$id <- as.character(tokenMat$id)
tokenMat$freq <- as.numeric(tokenMat$freq)
tokenMat$word <- as.character(tokenMat$word)
head(tokenMat)
word_count <- tapply(tokenMat$freq,tokenMat$id,sum)
tokenMat$n <- word_count[match(tokenMat$id,names(word_count))]

by_id_sentiment_bing <- inner_join(bing,tokenMat,by="word")

by_id_sentiment_bing.unique <- by_id_sentiment_bing[!duplicated(by_id_sentiment_bing$id),]
dim(by_id_sentiment_bing.unique)
head(by_id_sentiment_bing.unique)

count_sentiment_by_id <- matrix(NA,length(reddit_comments$clean_comment_two),length(unique(bing$sentiment)))
colnames(count_sentiment_by_id) <- unique(bing$sentiment)

for(i in 1:nrow(count_sentiment_by_id)){
    stp <- by_id_sentiment_bing.unique[by_id_sentiment_bing.unique$id == i,]
    sub_mat <- matrix(0,1,ncol(count_sentiment_by_id))
    colnames(sub_mat) <- colnames(count_sentiment_by_id)
    
  for(j in 1:nrow(stp)){
    sub_mat[1,(stp$sentiment[j-i])] <- sub_mat[1,(stp$sentiment[j-i])] + stp$freq[j]
  }
    
  count_sentiment_by_id[i,] <- sub_mat[1,]
}
count_sentiment_by_id<- as.data.frame(count_sentiment_by_id)
head(count_sentiment_by_id)
length(count_sentiment_by_id$negative)
c(negative=sum(count_sentiment_by_id$negative),positive=sum(count_sentiment_by_id$positive))



# since this is for the movie Isle of Dogs
# and its IMDB rating is 7.9 by 100000 people
# meaning that only 21 percent could be taken as negative and the rest is positive

n <- length(reddit_comments$clean_comment_two)

sentiment_hat <- rbinom(n, 1, 0.79) # 1 for positive, 0 for negative

sentiment_hat <- as.data.frame(sentiment_hat)
head(sentiment_hat)

dim(sentiment_hat)
sum(sentiment_hat$sentiment_hat)


classification_bing <- ifelse(count_sentiment_by_id$positive > count_sentiment_by_id$negative, 1, 0)
confusion_matrix_bing <- table(sentiment_hat$sentiment_hat,classification_bing)
confusion_matrix_bing

##Overall Accuracy of Bing Lexicon vs sentiment_hat
accuracy <- sum(diag(confusion_matrix_bing))/sum(confusion_matrix_bing)
print('--Overall Accuracy--')
accuracy

tokens <- colnames(youtube2_dtm_1)
tokenMat <- cbind.data.frame(youtube2_dtm_1$i,youtube2_dtm_1$v,tokens[youtube2_dtm_1$j])
colnames(tokenMat) <- c("id","freq","word")
tokenMat$id <- as.character(tokenMat$id)
tokenMat$freq <- as.numeric(tokenMat$freq)
tokenMat$word <- as.character(tokenMat$word)
head(tokenMat)
word_count <- tapply(tokenMat$freq,tokenMat$id,sum)
tokenMat$n <- word_count[match(tokenMat$id,names(word_count))]

by_id_sentiment_bing <- inner_join(bing,tokenMat,by="word")

by_id_sentiment_bing.unique <- by_id_sentiment_bing[!duplicated(by_id_sentiment_bing$id),]
dim(by_id_sentiment_bing.unique)
head(by_id_sentiment_bing.unique)

count_sentiment_by_id <- matrix(NA,length(youtube_comments$clean_comment_two),length(unique(bing$sentiment)))
colnames(count_sentiment_by_id) <- unique(bing$sentiment)

for(i in 1:nrow(count_sentiment_by_id)){
    stp <- by_id_sentiment_bing.unique[by_id_sentiment_bing.unique$id == i,]
    sub_mat <- matrix(0,1,ncol(count_sentiment_by_id))
    colnames(sub_mat) <- colnames(count_sentiment_by_id)
    
  for(j in 1:nrow(stp)){
    sub_mat[1,(stp$sentiment[j-i])] <- sub_mat[1,(stp$sentiment[j-i])] + stp$freq[j]
  }
    
  count_sentiment_by_id[i,] <- sub_mat[1,]
}
count_sentiment_by_id<- as.data.frame(count_sentiment_by_id)
head(count_sentiment_by_id)
length(count_sentiment_by_id$negative)
c(negative=sum(count_sentiment_by_id$negative),positive=sum(count_sentiment_by_id$positive))



# since this is for the movie Isle of Dogs
# and its IMDB rating is 7.9 by 100000 people
# meaning that only 21 percent could be taken as negative and the rest is positive

n <- length(youtube_comments$clean_comment_two)

sentiment_hat <- rbinom(n, 1, 0.79) # 1 for positive, 0 for negative

sentiment_hat <- as.data.frame(sentiment_hat)
head(sentiment_hat)

dim(sentiment_hat)
sum(sentiment_hat$sentiment_hat)


classification_bing <- ifelse(count_sentiment_by_id$positive > count_sentiment_by_id$negative, 1, 0)
confusion_matrix_bing <- table(sentiment_hat$sentiment_hat,classification_bing)
confusion_matrix_bing

##Overall Accuracy of Bing Lexicon vs sentiment_hat
accuracy <- sum(diag(confusion_matrix_bing))/sum(confusion_matrix_bing)
print('--Overall Accuracy--')
accuracy
