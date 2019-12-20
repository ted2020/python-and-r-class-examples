library(dplyr)
library(ggplot2)

births <- read.csv("births.csv")

head(births)

bnames <- read.csv("bnames.csv")

head(bnames)

head(bnames[bnames$name=="Mike",])

bnames %>% filter(name=="Mike") %>% ggplot(aes(year,prop)) + geom_line()

bnames %>% filter(name=="Vivian") %>% ggplot(aes(year,prop)) + geom_line()

bnames %>% filter(name=="Vivian") %>% ggplot(aes(year,prop)) + geom_point()

bnames %>% filter(name=="Vivian") %>% ggplot(aes(year,prop,color=sex)) + geom_point()

head(bnames %>% filter (name=="Owen" & year %in% c(1950:1970)))

head(bnames %>% filter (name=="Michael" | year %in% c(1930:1940)))

head(bnames %>% select(., name,sex,year))

head(bnames %>% select(., -soundex))

head(bnames %>% select(., contains("d"),starts_with(("n"))))

head(bnames %>% arrange(., year, by_group = name))

head(bnames %>% arrange(., year, desc(prop)))

head(bnames %>% mutate (., prop2=prop*100))

thisyear <- as.numeric(format(Sys.time(), "%Y"))
head(bnames %>% transmute(., prop2 = prop*100, yeardiff=thisyear-year))

bnames %>% summarise(., sum(prop),mean(prop))

bnames %>% summarise(., n())
length(bnames$sex)

bnames %>% summarise(., n_distinct(name))

bnames %>% summarise(., nth(name,127))

bnames %>% filter(., name=="Ted", year %in% c(1980:1990))

bnames %>% filter(., name=="Ted", year %in% c(1980:1990), prop > c("0.00016"))
bnames %>% filter(., name=="Ted", year %in% c(1980:1990), prop > c("0.00016")) %>% summarise(.,n())

head(bnames %>% select(., starts_with(("y")),starts_with("p")))

head(bnames %>% select(., -year))

head(bnames %>% select(., year, name))

head(bnames %>% filter(., name=="Ted") %>% arrange(., desc(prop)))
bnames %>% filter(., name=="Ted") %>% arrange(., desc(prop)) %>% summarise(., n())

head(bnames %>% mutate(., mean=mean(prop),sd=sd(prop),min=min(prop),max=max(prop)))

str(births)
str(bnames)

head(left_join(births,bnames))

head(right_join(births,bnames))

head(inner_join(births,bnames))

head(full_join(births,bnames))

df2 <- inner_join(bnames,births)
head(df2)

head(bnames)

abc <- bnames %>%  filter(., year==1880 & sex=="boy")
head(abc %>% mutate(., klm=paste(name,sex,sep=" ")))

abc$sex <- as.character(abc$sex)
abc$sex[abc$sex == "boy"] <- "asdffd"

abc


