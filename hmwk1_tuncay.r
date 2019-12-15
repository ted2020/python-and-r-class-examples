


P <- 5000
n <- seq(1:15)
R <- 11.5/100

A=P*(1+R/100)^n
as.list(A)



heights_cm <- c(180, 165, 160, 193)
heights_m <- c(180, 165, 160, 193)/100
weights <- c(87, 58, 65, 100)

BMI <- weights/heights_m^2
BMI

heights_feet <- c(180, 165, 160, 193)*0.0328084
heights_feet

ifelse(heights_feet>6,"tall","short")

data(cars)

head(cars,5)

state <- c("ny","ca","ct","tx")

set.seed(0)
random_state <- rep(state, each=length(cars$speed))
random_state <- sample(random_state)
random_state

set.seed(0)
random <- rnorm(n=length(cars$speed),mean=0,sd = 1)
random

random_df <- as.data.frame(random)
cars_new <- cbind(cars,random_df,random_state)
head(cars_new,1)

ratio <- cars_new$random_state/cars_new$speed
cars_new <- rbind(cars_new,ratio)

#cars_new <- cars_new %>% mutate(ratio=random/speed)
head(cars_new)

newnew <- data.frame(cars,random_df,random_state)
head(newnew)

ts_data = read.csv('https://s3.amazonaws.com/graderdata/TimesSquareSignage.csv', 
                   stringsAsFactors=FALSE,
                   na.strings=c(""," ","NA")
)

head(ts_data)

str(ts_data)

nrow(ts_data)
ncol(ts_data)

anyNA(ts_data)

# all missing

# ts_data %>% filter_all(any_vars(is.na(.))) 
ts_data_missing <- ts_data[!complete.cases(ts_data),]
ts_data_missing

length(ts_data_missing[,1])
#count(ts_data_missing[])

#which(!rowSums(is.na(ts_data))==ncol(ts_data))
as.list(rowSums(is.na(ts_data)))

sum(rowSums(is.na(ts_data)))


colSums(is.na(ts_data))
colnames(ts_data)[colSums(is.na(ts_data))==0]

getwd()

getwd <- getwd()
path <- paste0(getwd,"/data_hwmk1")
dir.create(path)

colnames(ts_data)

unique(ts_data$Location)

upper_bway<- ts_data[ts_data$Location=="Upper Bway",]
head(upper_bway,1)
#getwd <- getwd()
path2 <- paste0(path,"/upperbway.csv")
write.csv(upper_bway,path2)

mean_sqrf <- mean(ts_data$SF)
mean_sqrf

greater_than_average_square_footage <- ts_data[ts_data$SF>mean_sqrf,]
greater_than_average_square_footage

tsf_order <- ts_data[order(-ts_data$TOTAL.SF),]
head(tsf_order)
colnames(tsf_order)

tsf_order2 <- tsf_order[,c('Screen.Name..LED...Vinyl.Signs.','Building.Address','Location',"TOTAL.SF")]
topten <- head(tsf_order2,10)
head(topten)
#getwd <- getwd()
path3 <- paste0(path,"/topten.csv")
write.csv(topten,path3)


