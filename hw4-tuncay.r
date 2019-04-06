
library(rvest)
library(dplyr)

url <- "https://www.cgu.edu/school/ssspe/division-of-politics-economics/economic-sciences-department/ "

webpage <- read_html(url)

cgu_econ <- webpage %>% html_nodes(xpath="//p")
cgu_econ

cgu_econ <- webpage %>% html_nodes(xpath="//p") %>% html_text()
cgu_econ[1:2]

faculty_names <-webpage%>%html_nodes(xpath="//h4")%>%html_text()

faculty_names <- faculty_names[1:10]
faculty_names

page_all_links <- webpage%>%html_nodes(xpath="//a")%>%html_attr("href")

page_all_links

faculty_links <- webpage%>%html_nodes(xpath='//*[@class="u-font-mercury"]')%>%html_attr("href")

identical(length(faculty_links),length(faculty_names))

faculty_links

faculty_df <- data.frame(faculty_names,faculty_links,stringsAsFactors = FALSE)

faculty_df

#Write a loop to go to each faculty members profile and extract everything with a pargraph tag.

store_content <- list()
for(i in 1:nrow(faculty_df)){
tryCatch({
    

p_content <- read_html(faculty_df[i,2])%>% html_nodes(xpath="//p")%>% html_text()

 store_content[[i]] <- p_content
},error = function(e){
print(e)
})
Sys.sleep(runif(1,5,10)) 
print(i/nrow(faculty_df))
}

i <- 1
read_html(faculty_df[i,2])%>% html_nodes(xpath="//p")%>% html_text()
