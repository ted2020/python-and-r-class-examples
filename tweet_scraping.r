
# Libraries; some of these you probably have to install (best of luck)

library(twitteR)
library(ROAuth); library(RJSONIO); library(stringr)
library(devtools)
library(httk);library(httpuv)


###############################################
# Set up the OAuth; need to do this each time #
###############################################

options(RCurlOptions = list(cainfo = system.file("CurlSSL", "cacert.pem", package = "RCurl")))

# In 2016, RTwitter had changed, from 2014. So had to set some things up
# This was a good fix:
# http://stackoverflow.com/questions/25856394/r-twitter-package-authorization-error

consumerKey	<- ""
consumerSecret <- ""

# Set up, will take you to twitter api on browser. You need to have a twitter api account
setup_twitter_oauth(consumer_key = consumerKey,
                    consumer_secret = consumerSecret,
                    access_token=NULL, access_secret=NULL)

###################################################################
# Read in Latino Candidate dataset to gather twitter handle names #
###################################################################


# Just Latino Candidates
lat <- read.csv("~/Dropbox/research/rtwitter_stuff/lat_cands_2014.csv",
				stringsAsFactors=F, header=T)
str(lat)

#######################################################
# gather_store_tweet -- function used to loop over to #
# 					harvest and store candidate tweets#
#######################################################

gather_store_tweet <- function(twitter_user,n=3000) {
	#https://sites.google.com/site/miningtwitter/questions/talking-about/given-users

	cand_tweet <- userTimeline(twitter_user, n = n) # Collect up to 3000 tweets
	# Put into easier dataframe
	cand_df = twListToDF(cand_tweet)
	# Subset to tweets just from 2014
	just_2014 <- grep("2014", cand_df$created, value=T) # Select just 2014 tweets
	cand_df <- cand_df[as.character(cand_df$created) %in% just_2014,]
	# Extract Just the TExt
	cand_txt = cand_df$text
	# remove retweet entities -- you may not want to do this
	cand_clean = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", cand_txt)
	# remove Atpeople
	cand_clean = gsub("@\\w+", "", cand_clean)
	# remove punctuation symbols
	cand_clean = gsub("[[:punct:]]", "", cand_clean)
	# remove numbers
	cand_clean = gsub("[[:digit:]]", "", cand_clean)
	# remove links
	cand_clean = gsub("http\\w+", "", cand_clean)
	# Strip out any blanks
	cand_clean <- cand_clean[cand_clean !=""] # Remove candidates without twitter accounts
	# Trim leading/trailing white space
	trim <- function (x) gsub("^\\s+|\\s+$", "", x)
	cand_clean <- trim(cand_clean)
	# Get rid of any duplicates
	cand_clean <- cand_clean[!duplicated(cand_clean)]
	# Candidate column
	cand_clean_df <- data.frame(cand_clean, twitter_user)

	return(cand_clean_df)
	# Clean the user name of @
	#cand_name = gsub("@", "", twitter_user)
	# Write out the tweets for storing
	#write(cand_unique, paste(directory,cand_name,".txt",sep=""))

}

##############################
# Individual Tweet Gathering #
##############################

# Gallego (AZ-07) and Zamora (TX-15)

gal_zam <- c("RubenGallego","ElectEdZamora")

cand_tweet <- userTimeline(gal_zam[2], n=50) # Second element from gal_zam
# Put into easier dataframe
cand_df = twListToDF(cand_tweet)


################################################
# Prepare Data to send to Harvesting Function  #
################################################

twitter_cand_names <- lat$Twitter
# Clean Names
twitter_cand_names <- twitter_cand_names[!is.na(str_extract(twitter_cand_names, "[A-z]+"))]


################################
# Harvest All Candidate Tweets #
################################
twitter_list <- list()

for (i in 1:2) { # Just look at 2
	twitter_list[[i]] <- gather_store_tweet(twitter_cand_names[i])
}

twit_df <- plyr::ldply(twitter_list, rbind)


write.csv(twit_df, "twitter_database.csv", row.names=F)



################################

# Read the saved tweet back in #
################################
# Old sheetz
#con <- file("~/Dropbox/crm_articles/cces/external_validity/cand_tweets/RepBrady.txt", "r")
#tweets <- readLines(con)

# list all files in directory then use those to loop over and read in


#######################
#  Dictionary Coding  #
#######################
# Using techniques learned from Essex Class
library(quanteda)
library(xlsx)
library(gdata)
