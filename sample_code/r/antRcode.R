# setwd("~/Desktop/Lancaster University Documents (n.abad@lancaster.ac.uk)/Term_Two/MATH.455_Bayesian_Inference")

ants<-read.table("labants.dat",header=TRUE)

X<-matrix(data=1,nrow=nrow(ants),ncol=4)
X[,2]<-as.numeric(ants[,3])-1
X[,3]<-as.numeric(ants[,4])
X[,4]<-as.numeric(ants[,5])
y<-ants[,2]


log.prior<-function(beta,V.prior){
  result <- 0.5 * t(beta) %*% solve(V.prior) %*% beta
  return(result)
}


poisson.log.like<-function(beta,X,y) {
  loglam <- X %*% beta
  M <- dpois(y,exp(loglam), log = T)
  return(sum(M))
}

## Run an RWM algorithm for nits starting at beta.start,
#    using a MVN jump proposal with variance lambda.prop^2 V.prop
#    for data vector y, covariate matrix X,
#    iid priors with variance prior.var and mean zero
#    assuming the data are Poisson with log-link.


library(MASS)
RWM<-function(nits,beta.start,V.prop,lambda.prop,y,X,V.prior) {
  p<-length(beta.start)
  betas<-matrix(nrow=nits,ncol=p) # to store the MCMC output
  
  n.accept<-0    
  
  beta.curr<-beta.start

  log.like.curr<-poisson.log.like(beta.curr,X,y)
  log.prior.curr<-log.prior(beta.curr,V.prior)

  for (i in 1:nits) {
    beta.prop<- mvrnorm(1,beta.curr,V.prop)
    
    log.like.prop<- poisson.log.like(beta.prop, X, y)
    log.prior.prop<- log.prior(beta.prop, V.prior)

    log.alpha<- log.like.prop + log.prior.prop - (log.like.curr + log.prior.curr)
    
    u<-runif(1)
      
    if (log(u) <= log.alpha) { # accept
      beta.curr<-beta.prop
      log.like.curr<-log.like.prop
      log.prior.curr<-log.prior.prop
      n.accept<-n.accept+1
    }

    betas[i,]<-beta.curr
  }

  return(betas)
}


## Run an IS algorithm for nits starting at beta.start,
#    using a MVT proposal with mean mean.prop, variance v.prop, df df.prop
#    for data vector y, covariate matrix X,
#    iid priors with variance prior.var and mean zero
#    assuming the data are Poisson with log-link.

IS<-function(nits,beta.start,mean.prop,V.prop,df.prop,y,X,V.prior) {

  p<-length(beta.start)
  betas<-matrix(nrow=nits,ncol=p) # to store the MCMC output
  
  n.accept<-0    
  
  beta.curr<-beta.start

  log.like.curr<-poisson.log.like(beta.curr,X,y)
  log.prior.curr<-log.prior(beta.curr,V.prior)
  log.pdens.curr<-mvt.log.density(beta.curr,mean.prop,V.prop,df.prop)

  for (i in 1:nits) {
    beta.prop<- # FILL IN

    log.like.prop<-  # FILL IN
    log.prior.prop<- # FILL IN
    log.pdens.prop<- # FILL IN

    log.alpha<- # FILL IN
    
    u<-runif(1)
      
    if (log(u) <= log.alpha) { # accept
      beta.curr<-beta.prop
      log.like.curr<-log.like.prop
      log.prior.curr<-log.prior.prop
      log.pdens.curr<-log.pdens.prop
      n.accept<-n.accept+1
    }

    betas[i,]<-beta.curr
  }

  return(betas)
}

#
# Suggested starting point for RWM code
#

V<- 0.001 * diag(rep(1,4))  
V.prior <- diag(rep(100,4))
b1<-RWM(10000,c(0,0,0,0),V,.001,y,X,V.prior)
plot.ts(b1)

