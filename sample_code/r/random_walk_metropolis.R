setwd("~/Desktop/Lancaster University Documents (n.abad@lancaster.ac.uk)/Term_Two/MATH.455_Bayesian_Inference")
library(MASS)
data <- read.table("newtonrigg.txt", header = F)
data$time <- seq(1,251,1)
names(data) <- c("year", "month", "mean", "rainfall", "sunlight")

### QUESTION 2# ##
mu <- function(b1, b2, b3, time){
  result <- b1 + (b2 * cos(pi * time / 6)) + (b3 * sin(pi * time / 6))
  return(result)
}


logLike <- function(b1,b2,b3,alpha,phi,data){
  n <- nrow(data)
  # Summation is initialized at the value when t = 1
  summation <- abs(data$mean[1] - mu(b1,b2,b3,1) - alpha *(5.5 - mu(b1,b2,b3,0)))
  
  # From t = 2:251, summation is added 
  for (i in 2:n){
    summation <- summation + (abs(data$mean[i] - mu(b1,b2,b3,i) - alpha * (data$mean[i-1] - mu(b1,b2,b3,i-1))))
  }
  
  loglikelihood <- (n * log(0.5 * phi)) + (-phi * summation)
  return(loglikelihood)
  
}







#### QUESTION 3 ####
logPrior <- function(theta, V.prior){
  result <- -0.5 * t(theta) %*% solve(V.prior) %*% theta
  return(result)
  }







#### QUESTION 4 ####

randomWalk <- function(iterations, theta.start, proposed.variance, data, prior.variance){
  
  p <- length(theta.start)
  numAccepted <- 0
  
  thetas <- matrix(nrow = iterations, ncol = p)
  
  current.theta <- theta.start
  
  current.logLikelihood <- logLike(
    current.theta[1],
    current.theta[2],
    current.theta[3],
    current.theta[4],
    current.theta[5],
    data # I have this instead of x and y because of my logLike function
  )
  
  current.prior <- logPrior(current.theta, prior.variance)
  
  for (i in 1:iterations){
    
    # Note that V.prop is the proposed variance
    proposed.theta <- mvrnorm(1, current.theta, proposed.variance)
    
    if (proposed.theta[5] > 0 & abs(proposed.theta[4]) < 1){
      proposed.logLike <- logLike(
        proposed.theta[1],
        proposed.theta[2],
        proposed.theta[3],
        proposed.theta[4],
        proposed.theta[5],
        data
      )
      
      proposed.logPrior <- logPrior(proposed.theta, prior.variance)
      
      log.alpha <- (proposed.logLike + proposed.logPrior) - (current.logLikelihood + current.prior)
      u <- runif(1)
      
      # Acceptance
        if (log(u) <= log.alpha){
          current.theta <- proposed.theta
          current.prior <- proposed.logPrior
          current.logLikelihood <- proposed.logLike
          numAccepted <- numAccepted + 1
        }
    }
        
      thetas[i, ] <- current.theta
  }
  return(thetas)
}






#### QUESTION 5 ####
V.prior <- diag(c(100,100,100,1/3,1/(0.01^2)))
theta <- c(0,0,0,0,10)

# First Iteration
V.proposed <- diag(rep(1,5))
first_iteration <- randomWalk(10000, theta, V.proposed, data, V.prior)
plot.ts(first_iteration)

# Second Iteration
V.proposed <- var(first_iteration)
second_iteration <- randomWalk(10000, theta, V.proposed, data, V.prior)
plot.ts(second_iteration)

# Third Iteration
V.proposed <- var(second_iteration)
third_iteration <- randomWalk(10000, theta, V.proposed, data, V.prior)
plot.ts(third_iteration)

# Fourth Iteration
V.proposed <- var(third_iteration)
fourth_iteration <- randomWalk(10000, theta, V.proposed, data, V.prior)
plot.ts(fourth_iteration)

# Fifth Iteration
V.proposed <- var(fourth_iteration)
fifth_iteration <- randomWalk(10000, theta, V.proposed, data, V.prior)
plot.ts(fifth_iteration)







#### QUESTION 7 ####
burn.in <- fifth_iteration[1001:10000,]
beta1.mean <- mean(burn.in[,1]) # 11.66546
beta2.mean <- mean(burn.in[,2]) # -5.388653
beta3.mean <- mean(burn.in[,3]) # -3.786601

mus <- data.frame(matrix(ncol = 2, nrow = 251))
names(mus) <- c("time", "mu")

for (i in 1:251){
  mus$time[i] <- i
  mus$mu[i] <- mu(beta1.mean, beta2.mean, beta3.mean, i)
}

# Plotting Y and MU
plot(data$mean, type = "l", xlab = "Time", ylab = "Y/Mu Values")
lines(mus$time, mus$mu, col = "red")
legend("topright", pch = c(1,1), col = c("black", "red"), legend = c("Y", "Mu"))
