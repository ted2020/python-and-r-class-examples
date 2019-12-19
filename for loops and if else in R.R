for (i in 1:5){
  print(i)
}


m <- 25
for (i in 1:m){
  print(i)
}




compute_k<- function(n){
  x <- 1:n
  sum(x^2)
}
m <- 25
k <- vector(length = m)
for (n in 1:m){
  k[n] <- compute_k(n)
}
length(k)
k



a <- 5
if (a<=4){
  print(a^2)
  }else{
    print(a^3)
  }

head(murders3)

meanmt <- mean(murders3$total)
minmt <- which.min(murders3$total)
if (minmt<meanmt){
  print(murders3$state[minmt])
}else{
  print("more than average")
}
