library(MASS)
library(ggplot2)
library(gridExtra)
################ Functions ##################
perplexity <- function(x){
  y = x[x!=0]
  return(2^(-sum(y*log2(y))))
}

get_probabilities <- function(points, relevant_point, sigma){
  distances <- as.matrix(dist(points))
  p1 <- exp(-distances[relevant_point,]^2/ (2*sigma))
  p1[relevant_point] <-0
  p1<-p1/sum(p1)
  return(p1)
}

calculate_perplexity <- function(points, origin, sigma){
  prob <- get_probabilities(points, origin, sigma)
  return(perplexity(prob))
}

binary_search <- function(eval_fn, target){
  tol=1e-10
  max_iter=10000 
  lower=1e-20
  upper=1000

  for (i in rep(1,max_iter)){
    guess = (lower + upper) / 2
    val = eval_fn(guess)
    if(val > target){
      upper = guess
    }else{
      lower = guess
    }
    if(abs(val - target) <= tol){
      break
    }
  }
  return(guess)
}
#############################################


################ Generate data ##################

N <- 200

#Target parameters for univariate normal distributions
rho <- -0.6
mu1 <- 1; s1 <- 8
mu2 <- 1; s2 <- 8

# Parameters for bivariate normal distribution
mu <- c(mu1,mu2) # Mean
sigma <- matrix(c(s1^2, s1*s2*rho, s1*s2*rho, s2^2),
                2) # Covariance matrix


bvn1 <- mvrnorm(N, mu = mu, Sigma = sigma ) 
colnames(bvn1) <- c("X1","X2")
originaldata <- bvn1 
#############################################
rpoint <- 1
perp <- 50
sigma <- binary_search(function(sigma){calculate_perplexity(bvn1, rpoint, sigma)}, perp)

p1 <- get_probabilities(bvn1, rpoint, sigma)
bvn1 <- cbind(bvn1,p1)
colnames(bvn1) <- c("X1","X2","Probabilities")




rbPal <- colorRampPalette(c("royalblue", "springgreen", "yellow","red"))

Col <- rbPal(200)[as.numeric(cut(bvn1[,3],breaks = 200))]

bvn1 <- as.data.frame(bvn1)
plot1<-ggplot(bvn1, aes(X1, X2, color = Probabilities)) +
  ggtitle(paste0("Example with perplexity = ",perp)) +
  geom_point() +
  geom_point(data=bvn1[1,], aes(X1,X2), color='black') +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_color_gradientn(colours = c("royalblue", "springgreen", "yellow","red"))

orderedb <- bvn1[order(bvn1[,3], decreasing = TRUE),]
orderedb <- cbind(orderedb,cumsum(orderedb$Probabilities))
colnames(orderedb) <- c("X1","X2","Probabilities", "CP")
plot2<-ggplot(orderedb,aes(seq(1,200,1),CP))+
  geom_point() +
  ggtitle(paste0("Cumulative Probabilities with perplexity = ",perp)) +
  ylab("Cumulative Probabilities") + xlab("Index") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))





              


perp <- 30
sigma <- binary_search(function(sigma){calculate_perplexity(bvn1, rpoint, sigma)}, perp)
bvn1 <- originaldata
p1 <- get_probabilities(bvn1, rpoint, sigma)
bvn1 <- cbind(bvn1,p1)
colnames(bvn1) <- c("X1","X2","Probabilities")




rbPal <- colorRampPalette(c("royalblue", "springgreen", "yellow","red"))

Col <- rbPal(200)[as.numeric(cut(bvn1[,3],breaks = 200))]

bvn1 <- as.data.frame(bvn1)
plot3<-ggplot(bvn1, aes(X1, X2, color = Probabilities)) +
  ggtitle(paste0("Example with perplexity = ",perp)) +
  geom_point() +
  geom_point(data=bvn1[1,], aes(X1,X2), color='black') +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_color_gradientn(colours = c("royalblue", "springgreen", "yellow","red"))

orderedb <- bvn1[order(bvn1[,3], decreasing = TRUE),]
orderedb <- cbind(orderedb,cumsum(orderedb$Probabilities))
colnames(orderedb) <- c("X1","X2","Probabilities", "CP")
plot4<-ggplot(orderedb,aes(seq(1,200,1),CP))+
  geom_point() +
  ggtitle(paste0("Cumulative Probabilities with perplexity = ",perp)) +
  ylab("Cumulative Probabilities") + xlab("Index") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


grid.arrange(plot1, plot2, plot3, plot4, nrow = 2)