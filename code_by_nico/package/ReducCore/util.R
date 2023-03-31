#--- Mean constraint normal distributions
rmixNormMul <- function(n, K, pik, p, mutild, sig2.1,sig2.2){
  #k: number of mixture components
  c1 <- (1-p)/sqrt(pow(p,2) + pow(1-p,2))
  c2 <- -p/sqrt(pow(p,2) + pow(1-p,2))
  Out = numeric(n)
  for(j in 1:n){
    pikval <- rmultinom(n=1,size=1,prob=pik) ## Gives a vector of length length(pik) =K
    p0 <- sapply(p,rbinom, n=1, size=1) 
    Res <- numeric(K)
    for(i in 1:K){
      p00 <- rbinom(n=1,size=1,prob = p0[i])
      Res[i] = p00*rnorm(n=1, mean=c1[i]*mutild[i], sd=sqrt(sig2.1[i])) + (1-p00)*rnorm(n=1, mean=c2[i]*mutild[i], sd=sqrt(sig2.2[i]))
    }
    Out[j] = pikval*Res
  }
  Out
}

B.basis <- function(x,knots){
    delta <- knots[2]-knots[1]
    n <- length(x)
    K <- length(knots)
    B <- matrix(0,n,K+1)
    for (jj in 1:(K-1))
    {
      act.inds <- (1:n)[(x>=knots[jj])&(x<=knots[jj+1])]
      act.x <- x[act.inds]
      resc.x <- (act.x-knots[jj])/(knots[jj+1]-knots[jj])
      
      B[act.inds,jj] <- (1/2)*(1-resc.x)^2
      B[act.inds,jj+1] <- -(resc.x^2)+resc.x+1/2
      B[act.inds,jj+2] <- (resc.x^2)/2
    }
    return(B)
  }

func <- function(Pik, n=1, size=1){
    which.max(rmultinom(n=1,size=1,prob = Pik))
  }

F2 <- function(Pik, n=1, size=1){
    
    apply(Pik,1,funcCmp)
  }

GamF <- function(gam, p0){
    
    (2*pnorm(-abs(gam))*exp(.5*gam^2) - p0)^2
  }
  
  #optimize(GamF,interval = c(-30,30), p0=1 - 0.05)
  GamBnd <- function(p0){
    
    Re1 <- optimize(GamF, interval = c(-30, 30), p0 = 1-p0)
    Re2 <- optimize(GamF, interval = c(-30, 30), p0 = p0)
    
    c( -abs(Re1$minimum),  abs(Re2$minimum), Re1$objective,Re2$objective)  
  }
  
  #--------------------------------------------
  #@ New parameters to use 
  #@parms si, ui, p, tau0, sig2k.e[k]; gamk[k], gamL; gamU 
  #@sig :  variance
  fGal.p2 <- function(e,sig2,gam, p,tau0){
    #sig2: is the standard deviation
    p.pos = p - 1.*(gam > 0)
    p.neg = p - 1.*(gam < 0)  
    est <- e/sig2 #sqrt(sig2) #sqrt(Thet$sig2k.e[Thet$Cik.e])
    
    a3 = p.pos*est/abs(gam)
    a2 = abs(gam)*p.neg/p.pos
    if(est/gam > 0){
      #(2*p*(1-p)/sqrt(sig2))*((pnorm(q = a2-a3) - pnorm(q = a2))*exp(-est*p.neg + .5*(gam^2)*(p.neg/p.pos)^2) + pnorm(q = -abs(gam) + a3)*exp(-p.pos*est + 0.5*(gam^2)))
      (2*p*(1-p)/sig2)*((pnorm(q = a2-a3) - pnorm(q = a2))*exp(-est*p.neg + .5*(gam^2)*(p.neg/p.pos)^2) + pnorm(q = -abs(gam) + a3)*exp(-p.pos*est + 0.5*(gam^2)))
    }else{
      (2*p*(1-p)/sig2)*(pnorm(q = -abs(gam))*exp(-p.pos*est + 0.5*(gam^2)) )
    }
    # else{
    #   #Val0 = ifelse(est/gam > 0, -abs(gam) + a3,  -abs(gam))
    #   #    (2*p*(1-p)/sqrt(sig2))*((pnorm(q = sign(est)*Inf) - 0.5)*exp(-est*p.neg + .5*(gam^2)*(p.neg^2)/(p.pos^2))*(est/gam > 0) + pnorm(q = -abs(gam) + Val0)*exp(-p.pos*est + 0.5*(gam^2)) )
    #   ald::dALD(y=e, mu = 0, p = tau0)
    # }
    
  }
  
  fGal.p0 <- function(e,sig2,gam, p,tau0){
    #sig2: is the standard deviation
    p.pos = p - 1.*(gam > 0)
    p.neg = p - 1.*(gam < 0)  
    est <- e/sig2 #sqrt(sig2) #sqrt(Thet$sig2k.e[Thet$Cik.e])
    
    a3 = p.pos*est/abs(gam)
    a2 = abs(gam)*p.neg/p.pos
    if(est/gam > 0){
      #(2*p*(1-p)/sqrt(sig2))*((pnorm(q = a2-a3) - pnorm(q = a2))*exp(-est*p.neg + .5*(gam^2)*(p.neg/p.pos)^2) + pnorm(q = -abs(gam) + a3)*exp(-p.pos*est + 0.5*(gam^2)))
      (2*p*(1-p)/sig2)*(exp(-est*p.neg + .5*(gam^2)*(p.neg/p.pos)^2 + pnorm(q = a2-a3, log.p = T) + Rmpfr::log1mexp(abs(pnorm(q = a2-a3, log.p = T) - pnorm(q = a2, log.p = T))) ) + exp(-p.pos*est + 0.5*(gam^2) + pnorm(-abs(gam) + a3, log.p = T)))
    }else{
      (2*p*(1-p)/sig2)*(exp(-p.pos*est + 0.5*(gam^2) + pnorm(q = -abs(gam), log.p = T )))
    }
    # else{
    #   #Val0 = ifelse(est/gam > 0, -abs(gam) + a3,  -abs(gam))
    #   #    (2*p*(1-p)/sqrt(sig2))*((pnorm(q = sign(est)*Inf) - 0.5)*exp(-est*p.neg + .5*(gam^2)*(p.neg^2)/(p.pos^2))*(est/gam > 0) + pnorm(q = -abs(gam) + Val0)*exp(-p.pos*est + 0.5*(gam^2)) )
    #   ald::dALD(y=e, mu = 0, p = tau0)
    # }
    
  }
  
  LogPlostGalLikOld <- function(parm,id, k, ei, Thet){
    #@parm is a vector: parm[1]=sqrt(sigke), parm[2] = gamk
    
    p = (parm[2] < 0) + (Thet$p0 - (parm[2] < 0))/(2*pnorm(-abs(parm[2]))*exp(parm[2]))
    
    sum(log(mapply(fGal.p0, e = ei[id], sig2 = parm[1], gam = parm[2], p = p, tau0 =Thet$tau0) + .Machine$double.eps))
    + dgamma( parm[1], shape = a.sig, scale = b.sig, log=T) + dunif(parm[2], min = Thet$gamL , max = Thet$gamU, log = T) 
    
  }
  
  LogPlostGalLik <- function(parm, ei, tau0){
    #@parm is a vector: parm[1]=sqrt(sigke), parm[2] = gamk
    
    p = (parm[2] < 0) + (tau0 - (parm[2] < 0))/(2*pnorm(-abs(parm[2]))*exp(.5*parm[2]*parm[2]))
    
    sum(log(mapply(fGal.p0, e = c(ei), sig2 = parm[1], gam = parm[2], p = p, tau0 =tau0) + .Machine$double.eps))
    #+ dgamma(parm[1], shape = a.sig, scale = b.sig, log=T) + dunif(parm[2], min = Thet$gamL , max = Thet$gamU, log = T) 
    
  }

  LogPlostGalLikBet <- function(parm, ei, X, tau0){
    #@parm is a vector: parm[1]=sqrt(sigke), parm[2] = gamk
    
    p = (parm[2] < 0) + (tau0 - (parm[2] < 0))/(2*pnorm(-abs(parm[2]))*exp(.5*parm[2]*parm[2]))
    ei2 <- ei - c(X%*%parm[2+c(1:ncol(X))])
    
    sum(log(mapply(fGal.p0, e = c(ei2), sig2 = parm[1], gam = parm[2], p = p, tau0 =tau0) + .Machine$double.eps))
    #+ dgamma(parm[1], shape = a.sig, scale = b.sig, log=T) + dunif(parm[2], min = Thet$gamL , max = Thet$gamU, log = T) 
    
  }