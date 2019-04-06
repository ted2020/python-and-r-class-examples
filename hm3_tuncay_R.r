
library(XML)
library(httr)

url1 <- "https://finance.yahoo.com/quote/CMG/financials?p=CMG"

html_data1 <- GET(url1)
html_text <- content(html_data1,as="text")
income_s <- readHTMLTable(html_text,header=T)
income_s

library(XML)
library(httr)

url2 <- "https://finance.yahoo.com/quote/CMG/balance-sheet?p=CMG"

html_data2 <- GET(url2)
html_text <- content(html_data2,as="text")
balance_s <- readHTMLTable(html_text,header=T)
balance_s

library(XML)
library(httr)

url3 <- "https://finance.yahoo.com/quote/CMG/cash-flow?p=CMG"

html_data3 <- GET(url3)
html_text <- content(html_data3,as="text")
cash_flow <- readHTMLTable(html_text,header=T)
cash_flow

library(quantmod)
symbols_tech <- c("AAPL", "GOOG", "FB","TWTR")
symbols_food <- c("CMG","MCD","DPZ")
grabFinancials <- function(symbols){
    env1 <- new.env()
    getSymbols(Symbols = symbols, env = env1)
    close_data <- do.call(merge, eapply(env1, Cl))
  
}
tail(grabFinancials(symbols_tech))
tail(grabFinancials(symbols_food))

library(httr)
library(XML)

k <- ("https://finance.yahoo.com/quote/")
#l <- ("/financials")
 l <- ("/balance-sheet")
# l <- ("/cash-flow")
m <- ("?p=")
n <- ("CMG")
url <- paste(k,n,l,m,n,sep="")

grabFinancials <- function(n){
    html_data3 <- GET(url)
    html_text <- content(html_data3,as="text")
    statement <- readHTMLTable(html_text,header=T)
    statement
  
}
grabFinancials()
