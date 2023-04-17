
#--- Rcpp functions
{#
# func <- function(Pik, n=1, size=1){
#   which.max(rmultinom(n=1,size=1,prob = Pik))
# }
# funcCmp <- cmpfun(func)
# 
# F2 <- function(Pik, n=1, size=1){
# 
#   apply(Pik,1,funcCmp)
# }
# F2Cmp <- cmpfun(F2)
# 
# {
#   #
#   {Code <-
#     '
#   arma::mat CikW(arma::mat W, arma::mat X, arma::mat Mu, arma::mat Sig, arma::vec Pk){
# 
#   const double log2pi = std::log(2.0 * M_PI);
# 
#   int n = W.n_rows;
#   int K = Mu.n_rows;
#   int P = Mu.n_cols;
#   double constants = -(static_cast<double>(P)/2.0) * log2pi;
#   arma::mat sigma(P, P);
#   arma::mat pik(n, K);
#   arma::mat rooti(P, P);
# 
#   for(int k=0; k < K; k++){
#   sigma = arma::reshape(Sig.row(k), P, P);
#   rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
#   double rootisum = arma::sum(log(rooti.diag()));
#   for(int i=0; i < n; i++){
#   arma::rowvec xtp = W.row(i) - X.row(i);
#   arma::vec z = rooti * arma::trans( xtp - Mu.row(k));
#   pik(i,k) = Pk(k)*exp(constants - 0.5 * arma::sum(z%z) + rootisum);
#   }
#   }
#   return(pik);
#   }
#   '
#   }
# 
#   {Code <-
#       "
#       #include <RcppArmadillo.h>
#      // [[Rcpp::depends(RcppArmadillo)]]
# 
#      // [[Rcpp::export]]
#   arma::mat CikW(arma::mat W, arma::mat X, arma::mat Mu, arma::mat Sig, arma::vec Pk){
# 
#   const double log2pi = std::log(2.0 * M_PI);
# 
#   int n = W.n_rows;
#   int K = Mu.n_rows;
#   int P = Mu.n_cols;
#   double constants = -(static_cast<double>(P)/2.0) * log2pi;
#   arma::mat sigma(P, P);
#   arma::mat pik(n, K);
#   arma::mat rooti(P, P);
# 
#   for(int k=0; k < K; k++){
#   sigma = arma::reshape(Sig.row(k), P, P);
#   rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
#   double rootisum = arma::sum(log(rooti.diag()));
#   for(int i=0; i < n; i++){
#   arma::rowvec xtp = W.row(i) - X.row(i);
#   arma::vec z = rooti * arma::trans( xtp - Mu.row(k));
#   pik(i,k) = Pk(k)*exp(constants - 0.5 * arma::sum(z%z) + rootisum);
#   }
#   }
#   return(pik);
#   }
#   "
#   }
#   sourceCpp(code = Code)
# 
#   {Code1 <- '
#   arma::mat CikMvec(arma::mat M, arma::mat X, arma::mat Mu, arma::mat Sig, arma::vec Pk, arma::vec Delt){
# 
#   const double log2pi = std::log(2.0 * M_PI);
# 
#   int n = M.n_rows;
#   int K = Mu.n_rows;
#   int P = Mu.n_cols;
#   double constants = -(static_cast<double>(P)/2.0) * log2pi;
#   arma::mat sigma(P, P);
#   arma::mat pik(n, K);
#   arma::mat rooti(P, P);
#   arma::mat diDelt = diagmat(Delt);
# 
#   for(int k=0; k < K; k++){
#   sigma = arma::reshape(Sig.row(k), P, P);
#   rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
#   double rootisum = arma::sum(log(rooti.diag()));
#   for(int i=0; i < n; i++){
#   arma::rowvec xtp = M.row(i) - X.row(i)*diDelt;
#   arma::vec z = rooti * arma::trans( xtp - Mu.row(k));
#   pik(i,k) = Pk(k)*exp(constants - 0.5 * arma::sum(z%z) + rootisum);
#   }
#   }
#   return(pik);
#   }
#   '
#   }
#   {Code2 <- '
#   arma::mat CikX(arma::mat X, arma::mat Mu, arma::mat Sig, arma::vec Pk){
# 
#   const double log2pi = std::log(2.0 * M_PI);
# 
#   int n = X.n_rows;
#   int K = Mu.n_rows;
#   int P = Mu.n_cols;
#   double constants = -(static_cast<double>(P)/2.0) * log2pi;
#   arma::mat sigma(P, P);
#   arma::mat pik(n, K);
#   arma::mat rooti(P, P);
# 
#   for(int k=0; k < K; k++){
#   sigma = arma::reshape(Sig.row(k), P, P);
#   rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
#   double rootisum = arma::sum(log(rooti.diag()));
#   for(int i=0; i < n; i++){
#   arma::rowvec xtp = X.row(i);
#   arma::vec z = rooti * arma::trans( xtp - Mu.row(k));
#   pik(i,k) = Pk(k)*exp(constants - 0.5 * arma::sum(z%z) + rootisum);
#   }
#   }
#   return(pik);
#   }
#   '
#   }
#   # #
#   {Code3 <- '
# 
#   arma::colvec Y = Rcpp::as<arma::vec>(Ys);
#   arma::mat X = Rcpp::as<arma::mat>(Xs);
#   arma::colvec  Mu = Rcpp::as<arma::vec>(Mus);
#   arma::colvec Sig = Rcpp::as<arma::vec>(Sigs);
#   arma::colvec Pk  = Rcpp::as<arma::vec>(Pks);
#   arma::colvec Gam = Rcpp::as<arma::vec>(Gams);
# 
# 
#   int n = X.n_rows;
#   int K = Mu.n_elem;
# 
#   arma::mat pik = arma::zeros<arma::mat>(n, K);
#   arma::colvec Ytld = Y - X*Gam;
# 
#   for(int k=0; k < K; k++){
#   for(int i=0; i < n; i++){
#   pik(i,k) = Pk(k)*R::dnorm(Ytld(i), Mu(k), sqrt(Sig(k)), 0);
#   }
#   }
#   return Rcpp::wrap(pik);
#   '
#   }
#   #
#   Cike <- cxxfunction(signature(Ys="numeric", Xs="numeric",Mus="numeric",Sigs="numeric",Pks="numeric",Gams="numeric"),
#                       Code3, plugin="RcppArmadillo")
# 
#   #cppFunction(Code,depends = "RcppArmadillo")
#   cppFunction(Code1,depends = "RcppArmadillo")
#   cppFunction(Code2,depends = "RcppArmadillo")
# 
#   {
#     #   Code4 <- '
#     #
#     # arma::mat UpdateX(arma::vec Yd, arma::mat Wd, arma::mat Md, arma::mat Mukx, arma::vec Sige, arma::mat Sigw, arma::mat Sigm, arma::mat Sigx,
#     # arma::vec Cike, arma::vec Cikw, arma::vec Cikm, arma::vec Cikx,arma::vec Gamma, double Delta, arma::vec A, arma::vec B){
#     # int n = Yd.n_elem;
#     # int J = Wd.n_cols;
#     # arma::mat Sigt(J,J);
#     # arma::vec Mut;
#     # arma::mat Gamsq = Gamma*Gamma.t();
#     # double Deltsq = Delta*Delta;
#     # arma::mat Res(n,J);
#     #
#     # // Obtaining namespace of Matrix package
#     # Environment pkg1 = Environment::namespace_env("LaplacesDemon");
#     # Function F1 = pkg1["as.symmetric.matrix"];
#     # Function F2 = pkg1["as.inverse"];
#     # Environment pkg2 = Environment::namespace_env("TruncatedNormal");
#     # Function F3 = pkg2["rtmvnorm"];
#     #
#     # for(int i=0; i < n; i++){
#     # int wi = Cikw(i)-1;
#     # int mi = Cikm(i)-1;
#     # int xi = Cikx(i)-1;
#     # int ei = Cike(i)-1;
#     # arma::mat Sigw0 = as<arma::mat>(F2(F1(reshape(Sigw.row(wi), J, J))));
#     # arma::mat Sigm0 = as<arma::mat>(F2(F1(reshape(Sigm.row(mi), J, J))));
#     # arma::mat Sigx0 = as<arma::mat>(F2(F1(reshape(Sigx.row(xi), J, J))));
#     #
#     # Sigt = as<arma::mat>(F2(F1((Gamsq/Sige(ei) + Sigw0 + Sigx0 + Gamsq*Sigm0))));
#     # Mut = Sigt*((Gamma*Yd[i]/Sige(ei)) + Sigw0*Wd.row(i).t() + Sigm0*Md.row(i).t() + Sigx0*Mukx.row(xi).t());
#     #
#     # Res.row(i) = as<arma::rowvec>(F3(1, Mut, Sigt, A, B));
#     #
#     # }
#     #
#     # return Res;
#     #
#     # }
#     # '
#   }
# 
#   #    //arma::mat UpdateX(arma::vec Yd, arma::mat Wd, arma::mat Md, arma::mat Sigw, arma::mat Sigm, arma::mat Sigx, arma::vec Cikw, arma::vec Cikm, arma::vec Cikx, arma::vec A, arma::vec B, arma::mat Mukx ){
#   {
#     Code4 <- '
#     arma::mat UpdateX(arma::mat Wd, arma::mat Md, arma::mat Sigw, arma::mat Sigm, arma::mat Sigx, arma::vec A, arma::vec B, arma::mat Mukx ){
#     int n = Wd.n_rows;
#     int J = Wd.n_cols;
#     arma::mat Sigt(J,J);
#     arma::mat Sigtp(J,J);
#     arma::vec Mut(J);
#     //arma::mat Gamsq = Gamma*Gamma.t();
#     //double Deltsq = Delta*Delta;
#     arma::mat Res(n,J);
# 
#     // Obtaining namespace of Matrix package
#     Environment pkg1 = Environment::namespace_env("LaplacesDemon");
#     Function F1 = pkg1["as.symmetric.matrix"];
#     Function F2 = pkg1["as.inverse"];
#     Function F2a = pkg1["rmvnp"];
#     Function F2b = pkg1["rmvn"];
#     Environment pkg2 = Environment::namespace_env("TruncatedNormal");
#     Function F3 = pkg2["rtmvnorm"];
# 
#     for(int i=0; i < n; i++){
#     //int wi = Cikw(i)-1;
#     //int mi = Cikm(i)-1;
#     //int xi = Cikx(i)-1;
#     //int ei = Cike(i)-1;
#     //arma::mat Sigw0 = as<arma::mat>(F2(F1(reshape(Sigw.row(wi), J, J))));
#     //arma::mat Sigm0 = as<arma::mat>(F2(F1(reshape(Sigm.row(mi), J, J))));
#     //arma::mat Sigx0 = as<arma::mat>(F2(F1(reshape(Sigx.row(xi), J, J))));
# 
#     arma::mat Sigw0 = as<arma::mat>(F1(reshape(Sigw.row(i), J, J)));
#     arma::mat Sigm0 = as<arma::mat>(F1(reshape(Sigm.row(i), J, J)));
#     arma::mat Sigx0 = as<arma::mat>(F1(reshape(Sigx.row(i), J, J)));
# 
#     //Sigt = as<arma::mat>(F2(F1((Gamsq/Sige(ei) + Sigw0 + Sigx0 + Gamsq*Sigm0))));
#      Sigt = as<arma::mat>(F2(Sigw0 + Sigx0 + Sigm0));
#      Sigtp = Sigw0 + Sigx0 + Sigm0;
#     Mut =Sigt*(Sigw0*Wd.row(i).t() + Sigm0*Md.row(i).t() + Sigx0*Mukx.row(i).t());
# 
#     Res.row(i) = as<arma::rowvec>(F3(1, Mut, Sigt, A, B));
#     //Res.row(i) = as<arma::rowvec>(F2a(1, Mut, Sigtp));
# 
#     }
# 
#     return Res;
# 
#     }
#     '
#   }
#   # UpdateX <- cxxfunction(signature(Ys="numeric", Xs="numeric",Mus="numeric",Sigs="numeric",Pks="numeric",Gams="numeric"),
#   #                     Code3, plugin="RcppArmadillo")
# 
#   cppFunction(Code4, depends = "RcppArmadillo")
#   #
# }


}


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

#save.image(paste(OrgPath,"Compile_Function.RData",sep=""))
#load(paste(OrgPath,"Compile_Function.RData",sep=""))
#-- Using the rtmvnorm approach for the truncated multivariate normal
#pn0,
BQBayes.ME_SoFRFunc <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0 = 0.5,X){
  
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
  met <- c("f1","f2","f3","f4","f5")
  
  
  func <- function(Pik, n=1, size=1){
    which.max(rmultinom(n=1,size=1,prob = Pik))
  }
  funcCmp <- cmpfun(func)
  
  F2 <- function(Pik, n=1, size=1){
    
    apply(Pik,1,funcCmp)
  }
  F2Cmp <- cmpfun(F2)
  
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
  
  
  
  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  #pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  #pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  pn = c(5, 6, 15, 20)[which(n == c(100, 200,500,1000))]
  alpha.e = alpx
  alpha.w = alpha.W = alpx
  alpha.x = alpha.X = alpx
  alpha.m = alpx
  pn = pn #min(15, ceiling(length(Y)^{1/3.8})+4)
  n = length(Y) 
  Zmod = cbind(1,Z)
  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt #c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) # 
  Mn_Mod <- M%*%diag(1/Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20,"knots"), 1) ))
  Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a) 
  Wn <- (W%*%bs2)/length(a)
  Xn <- (X%*%bs2)/length(a)
  
  cat("\n #--- Initial values...\n")  
  P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D 
    return(P)
  }
  
  
  InitialValues <- function(Y, W, M, Z, df0=3, pn, Gx, a){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- Z #cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    lmFit <- lm(Y ~ -1 + Zmod + W_i) 
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- residuals.lm(lmFit)
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  InitialValuesQR <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z+Wn)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    plot(a, bs2%*%lmFit$coefficients[-c(1:3)], type="l")
    
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    Tp0 = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd)), upper=c(30, 0.95*max(bnd) ))
    Res$Hess <- Tp0$hessian
    
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  #Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, W, M, Z, df0=3, pn, Gx=1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  #Kvalues
  #Kc <- 5
  
  Kx = Thet0$Kx               ## number of X  cluster
  Ke = Thet0$Ke               ## number of ei cluster
  Km = Thet0$Km               ## number of ei cluster
  Kw = Thet0$Kw              ## number of omega_i cluster 
  
  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta   #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n=pn) #as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ #rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del#0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  #Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1]); Thet$gamU <- max(bnd[2])
  
  #bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) #c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
  
  Thet$Cik.x =  sample.int(Kx,size=n, replace = T) #Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e =  sample.int(Ke,size=n, replace = T) #Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w = sample.int(Kw,size=n, replace = T) #Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m = sample.int(Km,size=n, replace = T) #Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km
  
  
  Thet$Pik.x = rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)#Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e = rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)#Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w = rdirch(n=1,alpha = rep(alpha.e, Kw)/Kw) #Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m = rdirch(n=1,alpha = rep(alpha.m, Km)/Km)#Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw
  
  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw
  
  Thet$sig2k.e  = nimble::rinvgamma(n=Ke, shape=5/2,scale=8/2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n=Ke, min = .95*Thet$gamL, max=.95*Thet$gamU)
  Thet$p <- 1*(Thet$gamk < 0) + (Thet$tau0 - 1*(Thet$gamk < 0))/(2*pnorm(-abs(Thet$gamk))*exp(0.5*Thet$gamk*Thet$gamk))
  Thet$Ak <- (1-2*Thet$p)/(Thet$p*(1-Thet$p)); Thet$Bk <- 2/(Thet$p*(1-Thet$p)); Thet$Ck <- 1/( 1*(Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean=0, sd=sqrt(Thet$sig2k.e[1])) ; Thet$nui <- rexp(n=n, rate = 1/Thet$sig2k.e[1]) 
  #Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix
  
  Thet$Sigk.w <- repmat(c(.5*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw*pn), nrow = Kw)          #- vectors of length Kw and Kw respectively
  
  Thet$Sigk.x <- repmat(c(.5*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$Muk.x  <- matrix(rnorm(n=Kx*pn), nrow=Kx)   #- vectors of length Kx and Kx respectively
  
  Thet$Sigk.m <- repmat(c(.5*diag(pn)), Km, 1)     #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2*diag(pn)), Km, 1)
  Thet$Muk.m  <- matrix(rnorm(n=Km*pn), nrow = Km)   #- vectors of length Kx and Kx respectively
  
  A <- min(c(c(Wn), c(Mn))) - .5*diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5*diff(range(c(Wn, Mn)))
  #Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  #matrix(rnorm(n = n*pn), ncol=pn)
  #Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5*(Mn+Wn) #Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))
  
  
  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
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
  
  
  #method = "L-BFGS-B"
  #method = "BFGS"
  
  #Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  #Res
  
  #Res$hessian
  #Res$hessian/n
  #solve(Res$hessian/n)
  
  #LogPlostGalLik(c(.05,), c(ei), Thet)
  
  #-------- Update Beta (vector of length pn) 
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #   
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma 
  #   
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #   
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #   
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #   
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)
  
  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together) 
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 = ncol(Z) +1
  Pgam <- P.mat(pn)*tauval
  Sig0.gam0 = diag(pn0); Mu0.gam0 = numeric(pn0+pn);
  #InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))
  
  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.gam0%*%Mu0.gam0))
    
    #c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-6,length(Mu.gam)), ub = rep(6,length(Mu.gam))))
    
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    
  }
  updateGam <- cmpfun(updateGam)
  
  #siggam ~ IG(al0, bet0)
  al0 = 0.001
  bet0 = .005
  al0 = 2.5978252 #1
  bet0 = 0.4842963 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    val <- nimble::rinvgamma(n = 1, shape = al, scale = bet)
    ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e 
  #alpha.e = 1.0
  updatePik.e <- function(Thet, alpha.e){
    
    nk <- numeric(length(Thet$Cik.e))
    for(i in 1:length(Thet$Cik.e)){ nk[i] = sum(Thet$Cik.e == i)}
    
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$Cik.e) + nk
    
    rdirichlet(n=1, alpha.enew)
    
  }
  
  #-------- Update Cik.e  
  
  #ei <- Y - Z%*%Thet$Betaz - Thet$X%*%Thet$Gamma
  updateCik.e <- function(i, Thet,ei){
    #Y - Thet$C*abs(Thet$gam)*Thet$si - Thet$A*Thet$nui
    
    #  Yi.tld <- Y[i] - c(crossprod(Thet$Gamma, Thet$X[i,]))
    #pi.k  <- Thet$Pik.e*mapply(dnorm, x = Yi.tld, mean=Thet$muk.e, sd = sqrt(Thet$sig2k.e)) + .Machine$double.eps
    pik <- Thet$Pik.e*mapply(fGal.p0, e = ei[i], sig2 = sqrt(Thet$sig2k.e), gam = Thet$gamk, p = Thet$p, tau0 = Thet$tau0) #+ .Machine$double.eps
    which(rmultinom(n=1, size=1, prob = pik) == 1)
  }
  updateCik.e <- Vectorize(updateCik.e,"i") 
  
  #--- update Si
  #-- calss id for individual i (Matrix with columns of nui and si)
  updatenui <- function(Thet, Y, Z){
    
    ki = Thet$Cik.e
    bi <- ( ((Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ck[ki]*abs(Thet$gamk[ki])*Thet$si)^2)/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])))
    ai = ((Thet$Ak[ki]*Thet$Ak[ki])/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])) + 2/sqrt(Thet$sig2k.e[ki]))
    
    #mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) 
    #val = mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    #ifelse(val < .001, rexp(n = length(muui), rate = 1/sqrt(Thet$sig2k.e[Thet$Cik.e] )), val)
  }
  
  updatesi <- function(Thet, Y, Z){   
    #-- update si
    ki = Thet$Cik.e
    sig2si <- 1 /( 1/Thet$sig2k.e[ki] +  ((Thet$Ck[ki]*Thet$gamk[ki])^2)/ (Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui))
    muui <- sig2si*(Thet$Ck[ki]*abs(Thet$gamk[ki])*(Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ak[ki]*Thet$nui))/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui)
    
    #mapply(truncnorm::rtruncnorm, n=1, a = 0.01, mean = muui, sd = sqrt(sig2si))
    mapply(truncnorm::rtruncnorm, n=1, a = 0.001, mean = muui, sd = sqrt(sig2si))
    
    
  }
  
  
  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  #a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig=5/2; b.sig = 8/2; sdsig.prop = 0.2; sdgam.prop = .3  #2.153640 1.167331
  a.sig=2.153640; b.sig = 1.167331; sdsig.prop = 0.2; sdgam.prop = .3 # 7.133445 4.303263
  a.sig=7.133445; b.sig = 4.303263; sdsig.prop = 0.2; sdgam.prop = .3 #
  updatesig_gam <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop){
    
    a0 = 0.001
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      sig_newk <- truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gam <- Vectorize(updatesig_gam,"k")
  
  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10,10,10)*1
  varxi <- c(10,10,10)*.25
  #varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi){
    
    a0 = 0.1
    ab = 5
    #varxi = 5
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]),Thet$gamk[k]),sigma = Sdsig.prop*varxi[k], lb = c(a0,0.95*Thet$gamL), ub = c(ab, .95*Thet$gamU)  )
      
      
      sig_newk <- Val[1] #truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] #truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu=c(sig_newk, gam_newk), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE) )
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes,"k")
  
  
  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices
  InvCov <- function(Mat){
    K <- nrow(Mat)
    J <- round(sqrt(ncol(Mat)))
    Res0 = matrix(0, nrow = K, ncol=ncol(Mat))
    for(k in 1:K){ Res0[k,] <- c(as.inverse(as.symmetric.matrix(matrix(Mat[k,],ncol=J, byrow=F)))) }
    Res0
  }
  
  
  #----------------------------------------------
  # Update things related to Wi 
  #--------------------------------------------
  #Piw
  #Cik.w
  #--- Update Pik.x  
  #alpha.W = alpha.w = 1.0
  updatePik.w <- function(Thet, alpha.W){
    Ku = nrow(Thet$Muk.w)
    nk <- numeric(Ku)
    for(k in 1:Ku){ nk[k] = sum(Thet$Cik.w == k)}
    
    #update the Pis
    alpha.wnew <- alpha.W/length(Thet$muk.w) + nk
    
    rdirichlet(n=1, alpha.wnew)
    
  }
  
  #--- Update Ci.k mixture of normals 
  dmvnormVec <- function(x, Mean, SigmaVc){
    Ku <- nrow(Mean)
    Val <- numeric(Ku)
    for(k in 1:Ku){ Val[k] <- mvtnorm::dmvnorm(x = x, mean = c(Mean[k,]), sigma = as.symmetric.matrix(matrix(SigmaVc[k,],ncol=length(x),byrow=F))) }
    Val
  }
  #dmvnormVec <- Vectorize(dmvnormVec,"k")
  #dmvnormVec(x = rep(0,6), Mn=Thet$Muk.w,SigmaVc = Thet$Sigk.w)
  
  updateCik.w <- function(i, Thet, W){ # mixture of normals 
    #Sig <- matrix(Thet$Sigk.w[id,], ncol = ncol(W), byrow = F)
    #kw = nrow(Thet$Muk.w)
    xi =  c(W[i,] - Thet$X[i,])
    #pi.k  = Thet$Pik.w*mapply(dmvnormVec, k = 1:Kw, MoreArgs = list(x = xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w))
    pi.k  = Thet$Pik.w*dmvnormVec(x=xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  
  #updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  #Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol=pn, nrow=pn));inSig0.w = as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.w <- function(Thet, W, Mu0.w, inSig0.w, Sig0.w){
    Ku = length(Thet$Pik.w)
    J = ncol(Thet$X)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol=J, nrow=J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = W - Thet$X 
    #Muk <- matrix(0, ncol=J,nrow=Ku)
    SigK <- list()
    for(k in 1:Ku){
      idk <- which(Thet$Cik.w == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow= nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.w)) ## sum of matrices inverse
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi.tld[idk,])) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
        Mu.k[k,] <- c(SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.w%*%Mu0.w))      ## some of mat
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(apply(t(Yi.tld[idk,]), 2, sum)) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
      }else{
        SigK[[k]] <- Sig0.w
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.w
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL; for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.w[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.w) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- 1.0*as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], sigma = SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.w[-Ku]) %*% SimMU.k)/Thet$Pik.w[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.w <- cmpfun(updateMuk.w)
  
  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w = pn + 2; Psi0.w = 0.5*(nu0.w-pn-1)*cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  updateSigk.w <- function(Thet, W, nu0.w, Psi0.w){
    
    Ku = nrow(Thet$Muk.w)
    m = ncol(W)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Wtl <- W - Thet$X - Thet$Muk.w[Thet$Cik.w,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.w == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.w  + nk
        Yi0 = matrix(c(Wtl[id,]), nrow= nk, byrow=F)
        Psin.w <- as.positive.definite(Psi0.w + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.w))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.w, Psi0.w))
      }
    }
    Res
  }
  updateSigk.w <- cmpfun(updateSigk.w)
  
  #----------------------------------------------
  # Update things related to M 
  #--------------------------------------------  
  
  #--- update Pik.m for epsilon_i
  #alpha.m = 1.0
  updatePik.m <- function(Thet, alpha.m){
    nk <- numeric(nrow(Thet$Muk.m))
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    
    #update the Pis
    alpha.mnew <- alpha.m/nrow(Thet$Muk.m) + nk
    
    c(rdirichlet(n=1, alpha.mnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  
  updateCik.m <- function(i, Thet, M){ # mixture of normals 
    #id <- Thet$Cik.m[i]
    #Sig <- matrix(Thet$Sigk.m[id,], ncol = ncol(M), byrow = F)
    Km = nrow(Thet$Muk.m)
    xi = M[i,] - Thet$Delta*Thet$X[i,]
    #pi.k  = Thet$Pik.m*mapply(dmvnormVec, k = 1:Km, MoreArgs = list(x = xi , Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m))
    pi.k  = Thet$Pik.m*dmvnormVec(x = xi, Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m = rep(0, pn);  Sig0.m = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.m = as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.m <- function(Thet, M, Mu0.m, inSig0.m, Sig0.m){
    #cat(dim(inSig0.m),"(--1)\n")
    Ku = length(Thet$Pik.m)
    J = ncol(M)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol = J, nrow = J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = M - Thet$X%*%diag(Thet$Betadel) 
    Muk <- matrix(0, ncol=J, nrow=Ku)
    for(k in 1:Ku){
      idk <- which(Thet$Cik.m == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow = nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.m)) ## sum of matrices inverse
        #cat(dim(SigK[[k]]),"(--2 \n")
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.m%*%Mu0.m)      ## some of mat
        #cat(dim(SigKWg),"(--3 \n")
      }else{
        SigK[[k]] <- Sig0.m
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.m
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL;  for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.m[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.m) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.m[-Ku]) %*% SimMU.k)/Thet$Pik.m[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.m <- cmpfun(updateMuk.m)
  
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m = pn + 2; Psi0.m = .5*(nu0.m-pn-1)*as.symmetric.matrix(cov(Wn))  #diag(pn)
  
  updateSigk.m <- function(Thet, M, nu0.m, Psi0.m){
    
    Ku = nrow(Thet$Muk.m)
    m = ncol(M)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Mtl <- M - Thet$X%*%diag(Thet$Betadel) - Thet$Muk.m[Thet$Cik.m,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.m == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.m  + nk
        Yi0 <- matrix(c(Mtl[id,]), nrow=nk, byrow=F)
        Psin.m <- as.positive.definite(Psi0.m + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.m))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.m, Psi0.m))
      }
    }
    Res
  }
  updateSigk.m <- cmpfun(updateSigk.m)
  
  #---- Update things related to delta
  quadFormVec <- function(i,X,sigma,Val){
    id = Val[i]
    as.symmetric.matrix(quad.form(as.inverse(as.symmetric.matrix(matrix(sigma[id,], ncol = length(X[i,]),byrow=F))), diag(c(X[i,])))) 
  }
  quadFormVec <- Vectorize(quadFormVec,"i",SIMPLIFY = F) ## Resulting in a list if precision matrices
  quadFormVec <- cmpfun(quadFormVec)
  
  quadFormVec3 <- function(i,Mi,X,sigma,Val){
    id = Val[i]
    c(quad.3form.inv(matrix(sigma[id,], ncol = length(X[i,]),byrow=F), diag(X[i,]),Mi[i,]))
  }
  quadFormVec3 <- Vectorize(quadFormVec3,"i", SIMPLIFY = F)
  quadFormVec3 <- cmpfun(quadFormVec3)
  
  #SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),   
  # prior values
  mu0.del = 1.0; #sig02.del = 100
  updateDelta <- function(Thet, M, mu0.del, sig02.del){
    n = nrow(M)
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X), Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    sig20 <- 1/( 1/sig02.del + sum(Res))
    mu <- sig20*(sum(Res3) + mu0.del/sig02.del)
    rtruncnorm(n=1, a=0, b=10, mean = mu, sd = sqrt(sig20))
    
    #c(sum(Res3), sum(Res),mu, sqrt(sig20), rtruncnorm(n=1, a=0, b=Inf, mean = mu, sd = sqrt(sig20)))
    
  }
  updateDelta <- cmpfun(updateDelta)
  
  
  # prior values
  mu0.del = rep(0,length(Thet$Betadel)); #sig02.del = 20
  Sig0.del = P.mat(length(Thet$Betadel))*.1   #Thet$siggam) # Inverse or precision covariance matrix
  
  updateFunDelta <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDelta <- cmpfun(updateFunDelta)
  
  
  SimDElVec <- function(i, Mi, X, Sigma, Val){
    id <- Val[i]
    Ot = mvtnorm::rmvnorm(n = 1,mean = Mi[i,],sigma = as.symmetric.matrix(matrix(Sigma[id,], ncol = length(X[i,]),byrow=F)))
  }
  SimDElVec <- Vectorize(SimDElVec, "i")
  
  #-- 
  DeltComp <- function(k,Xst, X){
    abs(sum(Xst[,k]*X[,k])/(sum(X[,k]^2)))
    #abs(diag(as.inverse(crossprod(X))%*%crossprod(X,Xst)))
  }
  DeltComp <- Vectorize(DeltComp,"k")
  
  updateFunDeltaSt <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Xst <- base::t(SimDElVec(1:nrow(M), Mitl, Thet$X, Thet$Sigk.m, Thet$Cik.m)) ## K by n matrix and now I get a n by K
    c(DeltComp(1:ncol(Thet$X), Xst, Thet$X))
    #c(DeltComp(Xst, Thet$X))
    
    # Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    # Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    # Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    # Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    # c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDeltaSt  <- cmpfun(updateFunDeltaSt )
  
  
  #----------------------------------------------
  # Update things related to X 
  #--------------------------------------------------  
  #--- update Pik.m for epsilon_i
  #alpha.x <- 1.0
  updatePik.x <- function(Thet, alpha.x){
    nk <- numeric(nrow(Thet$Muk.x))
    for(k in 1:nrow(Thet$Muk.x)){ nk[k] = sum(Thet$Cik.x == k)}
    
    #update the Pis
    alpha.xnew <- alpha.x/nrow(Thet$Muk.x) + nk
    
    c(rdirichlet(n=1, alpha.xnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  updateCik.x <- function(i, Thet){ # mixture of normals 
    #Kx = ncol(Thet$X) # J x J
    #id <- Thet$Cik.x[i]
    #Sig <- matrix(Thet$Sigk.x[id,], ncol = length(Thet$Gamma), byrow = F)
    Xi = Thet$X[i,]
    #pi.k  = Thet$Pik.x*mapply(dmvnormVec, k = 1:Kx, x = Xi,  Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x)
    pi.k  = Thet$Pik.x*dmvnormVec(x=Xi,Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x = numeric(pn); Sig0.x = .5*diag(pn); inSig0.x = 2*diag(pn)#as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x = rep(0, pn);  Sig0.x = .25*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w)+colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.x = as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.x <- function(Thet, Mu0.x, inSig0.x, Sig0.x){
    K = length(Thet$Pik.x)
    J = ncol(Thet$X)
    Res0 <- matrix(0, nrow = K, ncol = J)
    Mu.k <- matrix(0, nrow = K, ncol = J) # K x J
    #SigKWg <- matrix(0, ncol = ncol(Y),nrow=ncol(Y)) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = Thet$X 
    Muk <- matrix(0, ncol = J, nrow = K)
    for(k in 1:K){
      idk <- which(Thet$Cik.x == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 <- matrix(c(Yi.tld[idk,]), nrow=nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[k,], ncol=J, byrow=F)))
        SigK <- as.inverse(as.symmetric.matrix((nk*TpSi + inSig0.x))) ## sum of matrices inverse
        Muk <- SigK%*%(crossprod(TpSi, c(colSums(Yi0))) + inSig0.x%*%Mu0.x)      ## some of mat
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Muk, sigma = SigK))
      }else{
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Mu0.x, sigma = Sig0.x ))
      }
    }
    #--- we impose some constrainsts on the mean
    Mu.k
    
  }
  updateMuk.x <- cmpfun(updateMuk.x)
  
  nu0.x <- pn + 2 ; Psi0.x <- .5*diag(pn) #as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2 ; Psi0.x <- .5*(nu0.x-pn-1)*cov(.5*(Wn+Mn)) #as.symmetric.matrix(.25*cov(Wn))
  updateSigk.x <- function(Thet, nu0.x, Psi0.x){
    
    Ku = nrow(Thet$Muk.x)
    m = ncol(Thet$Muk.x)
    Xtl <- Thet$X - Thet$Muk.x[Thet$Cik.x,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.x == i)
      nk = length(id)
      if(nk > 0){
        nux <- nu0.x  + nk
        Yi0 <- matrix(c(Xtl[id,]), nrow=nk, byrow=F)
        Psin.x <- Psi0.x + crossprod(Yi0)
        Res[i, ] <- c(rinvwishart(nux, Psin.x))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.x, Psi0.x))
      }
    }
    Res
  }
  updateSigk.x <- cmpfun(updateSigk.x)
  # 
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn))) 
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn))) 
  
  A <- min(c(Wn, Mn)) - .1*diff(range(c(Wn, Mn))) 
  B <- max(c(Wn, Mn)) + .1*diff(range(c(Wn, Mn))) 
  
  udapateXi <- function(i, Thet, Y, W, M, Z, A=A, B=B){
    ie =Thet$Cik.e[i]
    iw = Thet$Cik.w[i]
    im = Thet$Cik.m[i]
    ix = Thet$Cik.x[i]
    J = ncol(Thet$X)
    
    Zmod <- Z #cbind(1,Z)
    Ytl = (Y - Zmod%*%Thet$BetaZ - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Wtl = W - Thet$Muk.w[Thet$Cik.w,]
    Mtl = (M - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
    Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
    Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
    
    #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
    Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
    
    Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))) 
    
    #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi <- cmpfun(udapateXi)
  
  
  udapateXi2A <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2A <- cmpfun(udapateXi2A)
  
  ## This approach of simulating Xi only uses W, M, and X
  
  udapateXi2 <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #Mu <- c(Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2 <- cmpfun(udapateXi2)
  
  
  udapateXi2V <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    #Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    #GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- (Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi2V <- Vectorize(udapateXi2V,"i")
  udapateXi2V <- cmpfun(udapateXi2V)
  
  #--- Write Rcpp function to speed things up
  {
    #
    
    
  }
  
  
  
  #Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)
  
  ##----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  #Nsim <- 5000
  #Nsim = 5000 
  MCMCSample <- list()
  
  #-- for X
  MCMCSample$X <- matrix(0,nrow = Nsim,ncol=length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined) 
  MCMCSample$Muk.x <- matrix(0,nrow = Nsim, ncol = Kx*pn)
  MCMCSample$Sigk.x <- matrix(0,nrow = Nsim,ncol=Kx*pn*pn)
  MCMCSample$Pik.x <- matrix(0,nrow = Nsim,ncol=Kx)
  MCMCSample$Cik.x <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for W
  MCMCSample$Muk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn)
  MCMCSample$Sigk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn*pn)
  MCMCSample$Pik.w <- matrix(0,nrow = Nsim,ncol=Kw)
  MCMCSample$Cik.w <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for M
  MCMCSample$Muk.m <- matrix(0,nrow = Nsim,ncol=Km*pn)
  MCMCSample$Sigk.m <- matrix(0,nrow = Nsim,ncol=Km*pn*pn)
  MCMCSample$Pik.m <- matrix(0,nrow = Nsim,ncol=Km)
  MCMCSample$Cik.m <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow=Nsim, ncol=length(Thet$Betadel))
  
  # For Y
  MCMCSample$gamk <- matrix(0,nrow = Nsim, ncol=Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Pik.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Cik.e <- matrix(0,nrow = Nsim, ncol=nrow(Wn))
  
  MCMCSample$Gamma <- matrix(0,nrow = Nsim, ncol=length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)
  
  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  # 
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  # 
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  # 
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  #Accept <- 
  # 
  MCMCSample$AcceptX <- matrix(0,nrow = Nsim, ncol=Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  #stop()
  #------------------------------------------------------------------
  #Nsim = 5000 
  #Nsim = 1000
  #Nsim = 3000
  #library(profvis)
  #profvis(
  #system.time(
  #cat("\n #--- MCMC is starting ----> \n")
  
  for(iter in 1:Nsim){
    if(iter %% 500 == 0){cat("iter=",iter,"\n")}
    #----------------------------------------------
    # Update things related to X 
    #--------------------------------------------  
    #--- update Pik.m for epsilon_i
    Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    
    #--- update Cik.m for epsilon_i based on mixture of normals
    Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x)) 
    
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    
    Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    Thet$InvSigk.x <- InvCov(Thet$Sigk.x)
    
    #Ytd = Y - Zmod%*%Thet$BetaZ
    #if(iter%%10 == 0){
    #Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B) # Xn #
    #Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    #Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    #Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w, 
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)
    
    Thet$X <- UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,]) 
    
    #}
    #dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))
    
    #Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    #Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    #Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    #Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w, 
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J))) 
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Kw)
    for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    #update the Pis
    alpha.wnew <- alpha.W/Kw + nk
    Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    
    #--- Update Ci.k mixture of normals
    #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Km)
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    #update the Pis
    alpha.mnew <- alpha.m/Kw + nk
    Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    
    #--- Update Ci.k mixture of normals
    #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))  
    #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    #if(iter%%50 == 0){
    Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    MCMCSample$Betdel[iter,] <- Thet$Betadel
    #}
    #---------------------------------------------
    #cat("#--- update things related to Y or epsilon ")
    #--------------------------------------------- 
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for(k in 1:length(Thet$sig2k.e)){ nk[k] = sum(Thet$Cik.e == k)}
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n=1, alpha.enew))
    
    #update the Cik.e
    ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
    #Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- rep(1,n) #updateCik.e(1:length(Y), Thet, ei) #rep(1,n) #
    #Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    #update nui  and si
    #Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    #nuisi <- updatenuisi(Thet, Y, Zmod)
    #Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    #update Sigk.e
    #Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    #update sigk and gamk
    #--- update jointly sig2ke and gamke
    #Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi)
    Thet$sig2k.e <- Temp[1,];  Thet$gamk <- Temp[2,]; Thet$p <- Temp[3,]; MCMCSample$AcceptX[iter,] <- Temp[4,]; Thet$Ak <- Temp[5,]; Thet$Bk <- Temp[6,]; Thet$Ck <- Temp[7,];
    
    #update Gamma
    #Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    #Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <-  sig02.del#updateSigGam(Thet, al0, bet0, order = 2)# sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#
    
    #Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    # 
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    # 
    # #--- upodate beta_{z,m}
    # #--  Prior 
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    # 
    # #--- update X_i 
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX) 
    # 
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]); 
    # 
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb 
    
    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x))  ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise
    
    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w
    
    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    #MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e
    
    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  } 
  
  #)
  #)
  
  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample
}

#--- Separate update of the function paramaters 
#-- Using the rtmvnorm approach for the truncated multivariate normal
#-- update beta(t) and beta_z separately
#Nsim=1000; sig02.del = 0.1; tauval = .1; alpx=1; tau0 = 0.5;pn=15
#-- Fixed x (we know X) Fx
BQBayes.ME_SoFRFuncIndivFx <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0 = 0.5, X,g0){
  
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
  met <- c("f1","f2","f3","f4","f5")
  
  
  func <- function(Pik, n=1, size=1){
    which.max(rmultinom(n=1,size=1,prob = Pik))
  }
  funcCmp <- cmpfun(func)
  
  F2 <- function(Pik, n=1, size=1){
    
    apply(Pik,1,funcCmp)
  }
  F2Cmp <- cmpfun(F2)
  
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
  

  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  #pn = pn #min(15, ceiling(length(Y)^{1/3.8})+4)
  n = length(Y) 
  Zmod = cbind(1,Z)
  #------------------
  #pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  #pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  pn = c(5, 6, 20, 20)[which(n == c(100, 200,500,1000))]
  alpha.e = alpx
  alpha.w = alpha.W = alpx
  alpha.x = alpha.X = alpx
  alpha.m = alpx
  #-------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt #c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) # 
  Mn_Mod <- M%*%diag(1/Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20,"knots"), 1) ))
  Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a) 
  Wn <- (W%*%bs2)/length(a)
  Xn <- (X%*%bs2)/length(a)
  
  cat("\n #--- Initial values...\n")  
  P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D 
    return(P)
  }
  
  
  InitialValues <- function(Y, W, M, Z, df0=3, pn, Gx, a){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- Z #cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    lmFit <- lm(Y ~ -1 + Zmod + W_i) 
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- residuals.lm(lmFit)
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  InitialValuesQR <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z+Wn)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    plot(a, bs2%*%lmFit$coefficients[-c(1:3)], type="l")
    
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    Tp0 = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd)), upper=c(30, 0.95*max(bnd) ))
    Res$Hess <- Tp0$hessian
    
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  #--- Another version
  InitialValuesQR2 <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    #plot(a, bs2%*%lmFit$coefficients[-c(1:ncol(Z))], type="l")
    
    c_hatW <- lmFit$coefficients
    #Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    #Tp0 = optim(c(.2,-.1, rnorm(n=ncol(Wn))), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd), rep(-7, ncol(Wn)))
    #            , upper=c(30, 0.95*max(bnd),rep(7, ncol(Wn)) ))
    Tp0 = optim(c(.2,-.1, rnorm(n=ncol(Wn))), LogPlostGalLikBet, method = method, ei = c(ei), X = Wn, tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd), rep(-7, ncol(Wn)))
                , upper=c(30, 0.95*max(bnd),rep(7, ncol(Wn)) ))
    Res$Hess <- Tp0$hessian[1:2,1:2]
    Res$HessBet <- Tp0$hessian[-c(1:2),-c(1:2)]
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  #Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  #Thet0 <- InitialValuesQR(Y, W, M, Z, df0=3, pn, Gx=1, a, tau0)
  
  Thet0 <- InitialValuesQR2(Y, W, M, Z, df0=3, pn, Gx=1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  #Kvalues
  #Kc <- 5
  
  Kx = Thet0$Kx               ## number of X  cluster
  Ke = Thet0$Ke               ## number of ei cluster
  Km = Thet0$Km               ## number of ei cluster
  Kw = Thet0$Kw              ## number of omega_i cluster 
  
  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta   #- The intercepts vector of length J
  Thet$Gamma <- runif(n = pn, -4,4)  #rnorm(n=pn) #as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ #rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del#0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  #Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1]); Thet$gamU <- max(bnd[2])
  
  #bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) #c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
  
  Thet$Cik.x =  sample.int(Kx,size=n, replace = T) #Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e =  sample.int(Ke,size=n, replace = T) #Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w = sample.int(Kw,size=n, replace = T) #Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m = sample.int(Km,size=n, replace = T) #Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km
  
  
  Thet$Pik.x = rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)#Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e = rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)#Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w = rdirch(n=1,alpha = rep(alpha.e, Kw)/Kw) #Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m = rdirch(n=1,alpha = rep(alpha.m, Km)/Km)#Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw
  
  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw
  
  Thet$sig2k.e  = nimble::rinvgamma(n=Ke, shape=5/2,scale=8/2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n=Ke, min = .95*Thet$gamL, max=.95*Thet$gamU)
  Thet$p <- 1*(Thet$gamk < 0) + (Thet$tau0 - 1*(Thet$gamk < 0))/(2*pnorm(-abs(Thet$gamk))*exp(0.5*Thet$gamk*Thet$gamk))
  Thet$Ak <- (1-2*Thet$p)/(Thet$p*(1-Thet$p)); Thet$Bk <- 2/(Thet$p*(1-Thet$p)); Thet$Ck <- 1/( 1*(Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean=0, sd=sqrt(Thet$sig2k.e[1])) ; Thet$nui <- rexp(n=n, rate = 1/Thet$sig2k.e[1]) 
  #Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix
  
  Thet$Sigk.w <- repmat(c(.5*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw*pn), nrow = Kw)          #- vectors of length Kw and Kw respectively
  
  Thet$Sigk.x <- repmat(c(.5*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$Muk.x  <- matrix(rnorm(n=Kx*pn), nrow=Kx)   #- vectors of length Kx and Kx respectively
  
  Thet$Sigk.m <- repmat(c(.5*diag(pn)), Km, 1)     #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2*diag(pn)), Km, 1)
  Thet$Muk.m  <- matrix(rnorm(n=Km*pn), nrow = Km)   #- vectors of length Kx and Kx respectively
  
  A <- min(c(c(Wn), c(Mn))) - .5*diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5*diff(range(c(Wn, Mn)))
  #Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  #matrix(rnorm(n = n*pn), ncol=pn)
  #Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5*(Mn+Wn) #Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))
  
  
  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
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
  
  
  #method = "L-BFGS-B"
  #method = "BFGS"
  
  #Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  #Res
  
  #Res$hessian
  #Res$hessian/n
  #solve(Res$hessian/n)
  
  #LogPlostGalLik(c(.05,), c(ei), Thet)
  
  #-------- Update Beta (vector of length pn) 
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #   
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma 
  #   
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #   
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #   
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #   
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)
  
  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together) 
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 = ncol(Z) +1
  Pgam <- P.mat(pn)*tauval
  Sig0.gam0 = diag(pn0); Mu0.gam0 = numeric(pn0+pn);
  #InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))
  
  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.gam0%*%Mu0.gam0))
    
    #c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-6,length(Mu.gam)), ub = rep(6,length(Mu.gam))))
    
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    
  }
  updateGam <- cmpfun(updateGam)
  
  #--- Update Beta (error free covariates)
  #--- Prior parms
  pn0 = ncol(Z) +1
  #Pgam <- P.mat(pn)*tauval
  Sig0.bet0 = 100*diag(pn0); InvSig0.bet0 <- 1.0*as.inverse(Sig0.bet0); Mu0.bet0 = numeric(pn0);
  updateGamBeta <- function(Thet, Y, Zmod, InvSig0.bet0, Mu0.bet0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Thet$X%*%Thet$Gamma - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    #InvSig0.gam0 <- 1.0*as.inverse(Sig0.gam0) #, P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- Zmod 
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.bet0))
     Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.bet0%*%Mu0.bet0))
    
    c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-6,length(Mu.gam)), ub = rep(6,length(Mu.gam))))
    
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    
  }
  updateGamBeta <- cmpfun(updateGamBeta)
  
  #--- Update Gamma individually
  #--- Prior parms
  #pn0 = ncol(Z) +1
  #Pgam <- P.mat(pn)*tauval
   InvSig0.gam0 <- P.mat(pn); Mu0.gam0 = numeric(pn);
  updateGamma0 <- function(Thet, Y, Zmod, InvSig0.gam0, Mu0.gam0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Zmod%*%Thet$BetaZ - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    #InvSig0.gam0 <- 1.0*as.inverse(Sig0.gam0) #, P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- Thet$X 
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0*(1/Thet$siggam)))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.gam0%*%Mu0.gam0))
    
    #c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-6,length(Mu.gam)), ub = rep(6,length(Mu.gam))))
    
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    
  }
  updateGamma0 <- cmpfun(updateGamma0)
  
  #---- update beta using the GAL density instead
  K0 = P.mat(pn)
  SigmaProp <- solve(-Thet0$HessBet)
  diag(SigmaProp) <- abs(diag(SigmaProp)) ; c0 = .5
  SigmaProp <- as.positive.definite(as.symmetric.matrix(SigmaProp))
  #SigmaProp <- diag(pn); c0 = .5
  updateGamma <- function(Thet, Y, Zmod, K0,SigmaProp, c0, g0){
    
    #g0 = 15
    idk = Thet$Cik.e
    #Zmod <-  Z #cbind(1,Z)
    Ydi = Y - Zmod%*%Thet$BetaZ #- Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    ei_old = Ydi - Thet$X%*%Thet$Gamma

    Gam_newk <- TruncatedNormal::rtmvnorm(n=1, mu = Thet$Gamma, sigma = c0*SigmaProp, lb = rep(-g0,length(Thet$Gam)), ub = rep(g0,length(Thet$Gam)))
    ei_new = Ydi - Thet$X%*%Gam_newk  
    
    logLkold <- sum(log(mapply(fGal.p0, e = ei_old, sig2 = sqrt(Thet$sig2k.e[idk]), gam = Thet$gamk[idk], p =Thet$p[idk], tau0 =Thet$tau0) + .Machine$double.eps))
                 - .5*quad.form(K0*(1/Thet$siggam), Gam_newk) - TruncatedNormal::dtmvnorm(x = Gam_newk, mu = Thet$Gamma, sigma = c0*SigmaProp, lb = rep(-g0,length(Thet$Gam)), ub = rep(g0,length(Thet$Gam)), log=T)
      #LaplacesDemon::dmvnp(x = Gam_newk, mu = rep(0, length(Thet$Gamma)), Omega = K0*(1/Thet$siggam), log = T)
                   
                 #  (sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                 # - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
    
    logLknew <- sum(log(mapply(fGal.p0, e = ei_new, sig2 = sqrt(Thet$sig2k.e[idk]), gam = Thet$gamk[idk], p =Thet$p[idk], tau0 =Thet$tau0) + .Machine$double.eps))
             - .5*quad.form(K0*(1/Thet$siggam), Thet$Gamma) - TruncatedNormal::dtmvnorm(x = Thet$Gamma , mu = Gam_newk, sigma = c0*SigmaProp, lb = rep(-g0,length(Thet$Gam)), ub = rep(g0,length(Thet$Gam)), log=T)   
      #+ LaplacesDemon::dmvnp(x = Thet$Gamma, mu = rep(0, length(Thet$Gamma)), Omega = K0*(1/Thet$siggam), log = T)
                 #+ dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                 #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
    
    uv =  logLknew - logLkold 
    jump <- runif(n=1)
    if(log(jump) < uv){
      res <- c(1, Gam_newk)
    }else{
      res <- c(0, Thet$Gamma)
    }
     res
  }
  updateGamma <- cmpfun(updateGamma)
  
  
  
  #siggam ~ IG(al0, bet0)
  al0 = 0.001
  bet0 = .005
  al0 = 2.5978252 #1
  bet0 = 0.4842963 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    val <- nimble::rinvgamma(n = 1, shape = al, scale = bet)
    ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e 
  #alpha.e = 1.0
  updatePik.e <- function(Thet, alpha.e){
    
    nk <- numeric(length(Thet$Cik.e))
    for(i in 1:length(Thet$Cik.e)){ nk[i] = sum(Thet$Cik.e == i)}
    
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$Cik.e) + nk
    
    rdirichlet(n=1, alpha.enew)
    
  }
  
  #-------- Update Cik.e  
  
  #ei <- Y - Z%*%Thet$Betaz - Thet$X%*%Thet$Gamma
  updateCik.e <- function(i, Thet,ei){
    #Y - Thet$C*abs(Thet$gam)*Thet$si - Thet$A*Thet$nui
    
    #  Yi.tld <- Y[i] - c(crossprod(Thet$Gamma, Thet$X[i,]))
    #pi.k  <- Thet$Pik.e*mapply(dnorm, x = Yi.tld, mean=Thet$muk.e, sd = sqrt(Thet$sig2k.e)) + .Machine$double.eps
    pik <- Thet$Pik.e*mapply(fGal.p0, e = ei[i], sig2 = sqrt(Thet$sig2k.e), gam = Thet$gamk, p = Thet$p, tau0 = Thet$tau0) #+ .Machine$double.eps
    which(rmultinom(n=1, size=1, prob = pik) == 1)
  }
  updateCik.e <- Vectorize(updateCik.e,"i") 
  
  #--- update Si
  #-- calss id for individual i (Matrix with columns of nui and si)
  updatenui <- function(Thet, Y, Z){
    
    ki = Thet$Cik.e
    bi <- ( ((Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ck[ki]*abs(Thet$gamk[ki])*Thet$si)^2)/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])))
    ai = ((Thet$Ak[ki]*Thet$Ak[ki])/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])) + 2/sqrt(Thet$sig2k.e[ki]))
    
    #mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) 
    #val = mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    #ifelse(val < .001, rexp(n = length(muui), rate = 1/sqrt(Thet$sig2k.e[Thet$Cik.e] )), val)
  }
  
  updatesi <- function(Thet, Y, Z){   
    #-- update si
    ki = Thet$Cik.e
    sig2si <- 1 /( 1/Thet$sig2k.e[ki] +  ((Thet$Ck[ki]*Thet$gamk[ki])^2)/ (Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui))
    muui <- sig2si*(Thet$Ck[ki]*abs(Thet$gamk[ki])*(Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ak[ki]*Thet$nui))/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui)
    
    #mapply(truncnorm::rtruncnorm, n=1, a = 0.01, mean = muui, sd = sqrt(sig2si))
    mapply(truncnorm::rtruncnorm, n=1, a = 0.001, mean = muui, sd = sqrt(sig2si))
    
    
  }
  
  
  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  #a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig=5/2; b.sig = 8/2; sdsig.prop = 0.2; sdgam.prop = .3  #2.153640 1.167331
  a.sig=2.153640; b.sig = 1.167331; sdsig.prop = 0.2; sdgam.prop = .3 # 7.133445 4.303263
  a.sig=7.133445; b.sig = 4.303263; sdsig.prop = 0.2; sdgam.prop = .3 #
  updatesig_gam <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop){
    
    a0 = 0.001
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      sig_newk <- truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gam <- Vectorize(updatesig_gam,"k")
  
  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10,10,10)*1
  varxi <- c(10,10,10)*.25
  #varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi){
    
    a0 = 0.1
    ab = 5
    #varxi = 5
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]),Thet$gamk[k]),sigma = Sdsig.prop*varxi[k], lb = c(a0,0.95*Thet$gamL), ub = c(ab, .95*Thet$gamU)  )
      
      
      sig_newk <- Val[1] #truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] #truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu=c(sig_newk, gam_newk), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE) )
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes,"k")
  
  
  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices
  InvCov <- function(Mat){
    K <- nrow(Mat)
    J <- round(sqrt(ncol(Mat)))
    Res0 = matrix(0, nrow = K, ncol=ncol(Mat))
    for(k in 1:K){ Res0[k,] <- c(as.inverse(as.symmetric.matrix(matrix(Mat[k,],ncol=J, byrow=F)))) }
    Res0
  }
  
  
  #----------------------------------------------
  # Update things related to Wi 
  #--------------------------------------------
  #Piw
  #Cik.w
  #--- Update Pik.x  
  #alpha.W = alpha.w = 1.0
  updatePik.w <- function(Thet, alpha.W){
    Ku = nrow(Thet$Muk.w)
    nk <- numeric(Ku)
    for(k in 1:Ku){ nk[k] = sum(Thet$Cik.w == k)}
    
    #update the Pis
    alpha.wnew <- alpha.W/length(Thet$muk.w) + nk
    
    rdirichlet(n=1, alpha.wnew)
    
  }
  
  #--- Update Ci.k mixture of normals 
  dmvnormVec <- function(x, Mean, SigmaVc){
    Ku <- nrow(Mean)
    Val <- numeric(Ku)
    for(k in 1:Ku){ Val[k] <- mvtnorm::dmvnorm(x = x, mean = c(Mean[k,]), sigma = as.symmetric.matrix(matrix(SigmaVc[k,],ncol=length(x),byrow=F))) }
    Val
  }
  #dmvnormVec <- Vectorize(dmvnormVec,"k")
  #dmvnormVec(x = rep(0,6), Mn=Thet$Muk.w,SigmaVc = Thet$Sigk.w)
  
  updateCik.w <- function(i, Thet, W){ # mixture of normals 
    #Sig <- matrix(Thet$Sigk.w[id,], ncol = ncol(W), byrow = F)
    #kw = nrow(Thet$Muk.w)
    xi =  c(W[i,] - Thet$X[i,])
    #pi.k  = Thet$Pik.w*mapply(dmvnormVec, k = 1:Kw, MoreArgs = list(x = xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w))
    pi.k  = Thet$Pik.w*dmvnormVec(x=xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  
  #updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  #Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol=pn, nrow=pn));inSig0.w = as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.w <- function(Thet, W, Mu0.w, inSig0.w, Sig0.w){
    Ku = length(Thet$Pik.w)
    J = ncol(Thet$X)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol=J, nrow=J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = W - Thet$X 
    #Muk <- matrix(0, ncol=J,nrow=Ku)
    SigK <- list()
    for(k in 1:Ku){
      idk <- which(Thet$Cik.w == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow= nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.w)) ## sum of matrices inverse
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi.tld[idk,])) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
        Mu.k[k,] <- c(SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.w%*%Mu0.w))      ## some of mat
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(apply(t(Yi.tld[idk,]), 2, sum)) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
      }else{
        SigK[[k]] <- Sig0.w
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.w
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL; for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.w[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.w) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- 1.0*as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], sigma = SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.w[-Ku]) %*% SimMU.k)/Thet$Pik.w[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.w <- cmpfun(updateMuk.w)
  
  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w = pn + 2; Psi0.w = 0.5*(nu0.w-pn-1)*cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  updateSigk.w <- function(Thet, W, nu0.w, Psi0.w){
    
    Ku = nrow(Thet$Muk.w)
    m = ncol(W)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Wtl <- W - Thet$X - Thet$Muk.w[Thet$Cik.w,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.w == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.w  + nk
        Yi0 = matrix(c(Wtl[id,]), nrow= nk, byrow=F)
        Psin.w <- as.positive.definite(Psi0.w + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.w))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.w, Psi0.w))
      }
    }
    Res
  }
  updateSigk.w <- cmpfun(updateSigk.w)
  
  #----------------------------------------------
  # Update things related to M 
  #--------------------------------------------  
  
  #--- update Pik.m for epsilon_i
  #alpha.m = 1.0
  updatePik.m <- function(Thet, alpha.m){
    nk <- numeric(nrow(Thet$Muk.m))
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    
    #update the Pis
    alpha.mnew <- alpha.m/nrow(Thet$Muk.m) + nk
    
    c(rdirichlet(n=1, alpha.mnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  
  updateCik.m <- function(i, Thet, M){ # mixture of normals 
    #id <- Thet$Cik.m[i]
    #Sig <- matrix(Thet$Sigk.m[id,], ncol = ncol(M), byrow = F)
    Km = nrow(Thet$Muk.m)
    xi = M[i,] - Thet$Delta*Thet$X[i,]
    #pi.k  = Thet$Pik.m*mapply(dmvnormVec, k = 1:Km, MoreArgs = list(x = xi , Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m))
    pi.k  = Thet$Pik.m*dmvnormVec(x = xi, Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m = rep(0, pn);  Sig0.m = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.m = as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.m <- function(Thet, M, Mu0.m, inSig0.m, Sig0.m){
    #cat(dim(inSig0.m),"(--1)\n")
    Ku = length(Thet$Pik.m)
    J = ncol(M)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol = J, nrow = J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = M - Thet$X%*%diag(Thet$Betadel) 
    Muk <- matrix(0, ncol=J, nrow=Ku)
    for(k in 1:Ku){
      idk <- which(Thet$Cik.m == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow = nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.m)) ## sum of matrices inverse
        #cat(dim(SigK[[k]]),"(--2 \n")
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.m%*%Mu0.m)      ## some of mat
        #cat(dim(SigKWg),"(--3 \n")
      }else{
        SigK[[k]] <- Sig0.m
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.m
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL;  for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.m[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.m) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.m[-Ku]) %*% SimMU.k)/Thet$Pik.m[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.m <- cmpfun(updateMuk.m)
  
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m = pn + 2; Psi0.m = .5*(nu0.m-pn-1)*as.symmetric.matrix(cov(Wn))  #diag(pn)
  
  updateSigk.m <- function(Thet, M, nu0.m, Psi0.m){
    
    Ku = nrow(Thet$Muk.m)
    m = ncol(M)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Mtl <- M - Thet$X%*%diag(Thet$Betadel) - Thet$Muk.m[Thet$Cik.m,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.m == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.m  + nk
        Yi0 <- matrix(c(Mtl[id,]), nrow=nk, byrow=F)
        Psin.m <- as.positive.definite(Psi0.m + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.m))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.m, Psi0.m))
      }
    }
    Res
  }
  updateSigk.m <- cmpfun(updateSigk.m)
  
  #---- Update things related to delta
  quadFormVec <- function(i,X,sigma,Val){
    id = Val[i]
    as.symmetric.matrix(quad.form(as.inverse(as.symmetric.matrix(matrix(sigma[id,], ncol = length(X[i,]),byrow=F))), diag(c(X[i,])))) 
  }
  quadFormVec <- Vectorize(quadFormVec,"i",SIMPLIFY = F) ## Resulting in a list if precision matrices
  quadFormVec <- cmpfun(quadFormVec)
  
  quadFormVec3 <- function(i,Mi,X,sigma,Val){
    id = Val[i]
    c(quad.3form.inv(matrix(sigma[id,], ncol = length(X[i,]),byrow=F), diag(X[i,]),Mi[i,]))
  }
  quadFormVec3 <- Vectorize(quadFormVec3,"i", SIMPLIFY = F)
  quadFormVec3 <- cmpfun(quadFormVec3)
  
  #SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),   
  # prior values
  mu0.del = 1.0; #sig02.del = 100
  updateDelta <- function(Thet, M, mu0.del, sig02.del){
    n = nrow(M)
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X), Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    sig20 <- 1/( 1/sig02.del + sum(Res))
    mu <- sig20*(sum(Res3) + mu0.del/sig02.del)
    rtruncnorm(n=1, a=0, b=10, mean = mu, sd = sqrt(sig20))
    
    #c(sum(Res3), sum(Res),mu, sqrt(sig20), rtruncnorm(n=1, a=0, b=Inf, mean = mu, sd = sqrt(sig20)))
    
  }
  updateDelta <- cmpfun(updateDelta)
  
  
  # prior values
  mu0.del = rep(0,length(Thet$Betadel)); #sig02.del = 20
  Sig0.del = P.mat(length(Thet$Betadel))*.1   #Thet$siggam) # Inverse or precision covariance matrix
  
  updateFunDelta <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDelta <- cmpfun(updateFunDelta)
  
  
  SimDElVec <- function(i, Mi, X, Sigma, Val){
    id <- Val[i]
    Ot = mvtnorm::rmvnorm(n = 1,mean = Mi[i,],sigma = as.symmetric.matrix(matrix(Sigma[id,], ncol = length(X[i,]),byrow=F)))
  }
  SimDElVec <- Vectorize(SimDElVec, "i")
  
  #-- 
  DeltComp <- function(k,Xst, X){
    abs(sum(Xst[,k]*X[,k])/(sum(X[,k]^2)))
    #abs(diag(as.inverse(crossprod(X))%*%crossprod(X,Xst)))
  }
  DeltComp <- Vectorize(DeltComp,"k")
  
  updateFunDeltaSt <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Xst <- base::t(SimDElVec(1:nrow(M), Mitl, Thet$X, Thet$Sigk.m, Thet$Cik.m)) ## K by n matrix and now I get a n by K
    c(DeltComp(1:ncol(Thet$X), Xst, Thet$X))
    #c(DeltComp(Xst, Thet$X))
    
    # Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    # Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    # Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    # Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    # c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDeltaSt  <- cmpfun(updateFunDeltaSt )
  
  
  #----------------------------------------------
  # Update things related to X 
  #--------------------------------------------------  
  #--- update Pik.m for epsilon_i
  #alpha.x <- 1.0
  updatePik.x <- function(Thet, alpha.x){
    nk <- numeric(nrow(Thet$Muk.x))
    for(k in 1:nrow(Thet$Muk.x)){ nk[k] = sum(Thet$Cik.x == k)}
    
    #update the Pis
    alpha.xnew <- alpha.x/nrow(Thet$Muk.x) + nk
    
    c(rdirichlet(n=1, alpha.xnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  updateCik.x <- function(i, Thet){ # mixture of normals 
    #Kx = ncol(Thet$X) # J x J
    #id <- Thet$Cik.x[i]
    #Sig <- matrix(Thet$Sigk.x[id,], ncol = length(Thet$Gamma), byrow = F)
    Xi = Thet$X[i,]
    #pi.k  = Thet$Pik.x*mapply(dmvnormVec, k = 1:Kx, x = Xi,  Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x)
    pi.k  = Thet$Pik.x*dmvnormVec(x=Xi,Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x = numeric(pn); Sig0.x = .5*diag(pn); inSig0.x = 2*diag(pn)#as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x = rep(0, pn);  Sig0.x = .25*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w)+colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.x = as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.x <- function(Thet, Mu0.x, inSig0.x, Sig0.x){
    K = length(Thet$Pik.x)
    J = ncol(Thet$X)
    Res0 <- matrix(0, nrow = K, ncol = J)
    Mu.k <- matrix(0, nrow = K, ncol = J) # K x J
    #SigKWg <- matrix(0, ncol = ncol(Y),nrow=ncol(Y)) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = Thet$X 
    Muk <- matrix(0, ncol = J, nrow = K)
    for(k in 1:K){
      idk <- which(Thet$Cik.x == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 <- matrix(c(Yi.tld[idk,]), nrow=nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[k,], ncol=J, byrow=F)))
        SigK <- as.inverse(as.symmetric.matrix((nk*TpSi + inSig0.x))) ## sum of matrices inverse
        Muk <- SigK%*%(crossprod(TpSi, c(colSums(Yi0))) + inSig0.x%*%Mu0.x)      ## some of mat
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Muk, sigma = SigK))
      }else{
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Mu0.x, sigma = Sig0.x ))
      }
    }
    #--- we impose some constrainsts on the mean
    Mu.k
    
  }
  updateMuk.x <- cmpfun(updateMuk.x)
  
  nu0.x <- pn + 2 ; Psi0.x <- .5*diag(pn) #as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2 ; Psi0.x <- .5*(nu0.x-pn-1)*cov(.5*(Wn+Mn)) #as.symmetric.matrix(.25*cov(Wn))
  updateSigk.x <- function(Thet, nu0.x, Psi0.x){
    
    Ku = nrow(Thet$Muk.x)
    m = ncol(Thet$Muk.x)
    Xtl <- Thet$X - Thet$Muk.x[Thet$Cik.x,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.x == i)
      nk = length(id)
      if(nk > 0){
        nux <- nu0.x  + nk
        Yi0 <- matrix(c(Xtl[id,]), nrow=nk, byrow=F)
        Psin.x <- Psi0.x + crossprod(Yi0)
        Res[i, ] <- c(rinvwishart(nux, Psin.x))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.x, Psi0.x))
      }
    }
    Res
  }
  updateSigk.x <- cmpfun(updateSigk.x)
  # 
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn))) 
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn))) 
  
  A <- min(c(Wn, Mn)) - .1*diff(range(c(Wn, Mn))) 
  B <- max(c(Wn, Mn)) + .1*diff(range(c(Wn, Mn))) 
  
  udapateXi <- function(i, Thet, Y, W, M, Z, A=A, B=B){
    ie =Thet$Cik.e[i]
    iw = Thet$Cik.w[i]
    im = Thet$Cik.m[i]
    ix = Thet$Cik.x[i]
    J = ncol(Thet$X)
    
    Zmod <- Z #cbind(1,Z)
    Ytl = (Y - Zmod%*%Thet$BetaZ - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Wtl = W - Thet$Muk.w[Thet$Cik.w,]
    Mtl = (M - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
    Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
    Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
    
    #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
    Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
    
    Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))) 
    
    #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi <- cmpfun(udapateXi)
  
  
  udapateXi2A <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2A <- cmpfun(udapateXi2A)
  
  ## This approach of simulating Xi only uses W, M, and X
  
  udapateXi2 <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #Mu <- c(Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2 <- cmpfun(udapateXi2)
  
  
  udapateXi2V <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    #Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    #GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- (Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi2V <- Vectorize(udapateXi2V,"i")
  udapateXi2V <- cmpfun(udapateXi2V)
  
  #--- Write Rcpp function to speed things up
  {
    #
    
    
  }
  
  
  
  #Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)
  
  ##----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  #Nsim <- 5000
  #Nsim = 5000 
  MCMCSample <- list()
  
  #-- for X
  MCMCSample$X <- matrix(0,nrow = Nsim,ncol=length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined) 
  MCMCSample$Muk.x <- matrix(0,nrow = Nsim, ncol = Kx*pn)
  MCMCSample$Sigk.x <- matrix(0,nrow = Nsim,ncol=Kx*pn*pn)
  MCMCSample$Pik.x <- matrix(0,nrow = Nsim,ncol=Kx)
  MCMCSample$Cik.x <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for W
  MCMCSample$Muk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn)
  MCMCSample$Sigk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn*pn)
  MCMCSample$Pik.w <- matrix(0,nrow = Nsim,ncol=Kw)
  MCMCSample$Cik.w <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for M
  MCMCSample$Muk.m <- matrix(0,nrow = Nsim,ncol=Km*pn)
  MCMCSample$Sigk.m <- matrix(0,nrow = Nsim,ncol=Km*pn*pn)
  MCMCSample$Pik.m <- matrix(0,nrow = Nsim,ncol=Km)
  MCMCSample$Cik.m <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow=Nsim, ncol=length(Thet$Betadel))
  
  # For Y
  MCMCSample$gamk <- matrix(0,nrow = Nsim, ncol=Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Pik.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Cik.e <- matrix(0,nrow = Nsim, ncol=nrow(Wn))
  
  MCMCSample$Gamma <- matrix(0,nrow = Nsim, ncol=pn) #length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)
  MCMCSample$AcceptGam <- rep(NA, Nsim)
  
  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  # 
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  # 
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  # 
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  #Accept <- 
  # 
  MCMCSample$AcceptX <- matrix(0,nrow = Nsim, ncol=Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  #stop()
  #------------------------------------------------------------------
  #Nsim = 5000 
  #Nsim = 1000
  #Nsim = 3000
  #library(profvis)
  #profvis(
  #system.time(
  #cat("\n #--- MCMC is starting ----> \n")
  #Thet$X <- 
  
  for(iter in 1:Nsim){
    if(iter %% 500 == 0){cat("iter=",iter,"\n")}
    #----------------------------------------------
    # Update things related to X 
    #--------------------------------------------  
    #--- update Pik.m for epsilon_i
    Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    
    #--- update Cik.m for epsilon_i based on mixture of normals
    Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x)) 
    
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    
    Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    Thet$InvSigk.x <- InvCov(Thet$Sigk.x)
    
    #Ytd = Y - Zmod%*%Thet$BetaZ
    #if(iter%%10 == 0){
    #Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B) # Xn #
    #Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    #Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    #Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w, 
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)
    
    Thet$X <- Xn#UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,]) 
    
    #}
    #dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))
    
    #Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    #Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    #Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    #Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w, 
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J))) 
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Kw)
    for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    #update the Pis
    alpha.wnew <- alpha.W/Kw + nk
    Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    
    #--- Update Ci.k mixture of normals
    #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Km)
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    #update the Pis
    alpha.mnew <- alpha.m/Kw + nk
    Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    
    #--- Update Ci.k mixture of normals
    #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m)) 
    #Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))  
    #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    #if(iter%%50 == 0){
    Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    MCMCSample$Betdel[iter,] <- Thet$Betadel
    #}
    #---------------------------------------------
    #cat("#--- update things related to Y or epsilon ")
    #--------------------------------------------- 
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for(k in 1:length(Thet$sig2k.e)){ nk[k] = sum(Thet$Cik.e == k)}
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n=1, alpha.enew))
    
    #update the Cik.e
    ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
    #Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) #rep(1,n) #
    #Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    #update nui  and si
    #Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    #nuisi <- updatenuisi(Thet, Y, Zmod)
    #Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    #update Sigk.e
    #Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    #update sigk and gamk
    #--- update jointly sig2ke and gamke
    #Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi)
    Thet$sig2k.e <- Temp[1,];  Thet$gamk <- Temp[2,]; Thet$p <- Temp[3,]; MCMCSample$AcceptX[iter,] <- Temp[4,]; Thet$Ak <- Temp[5,]; Thet$Bk <- Temp[6,]; Thet$Ck <- Temp[7,];
    
    #update Gamma
    #Thet$siggam <- sig02.del
    #Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0)
    # Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    # Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    #Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    
    Thet$BetaZ <- updateGamBeta(Thet, Y, Zmod, InvSig0.bet0, Mu0.bet0)#Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Tep <- updateGamma(Thet, Y, Zmod, K0,SigmaProp, c0, g0)
    #print(Tep[length(Tep)])
    GamAccep <- Tep[1]
    Thet$Gamma <- Tep[-1]  #updateGamma(Thet, Y, Zmod, K0,SigmaProp, c0)
      #updateGamma(Thet, Y, Zmod, InvSig0.gam0, Mu0.gam0)  # Tp0[-c(1:ncol(Zmod))]
    
    Thet$siggam <-  updateSigGam(Thet, al0, bet0, order = 2)# sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#
    
    #Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    # 
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    # 
    # #--- upodate beta_{z,m}
    # #--  Prior 
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    # 
    # #--- update X_i 
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX) 
    # 
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]); 
    # 
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb 
    
    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x))  ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise
    
    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w
    
    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    #MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e
    MCMCSample$AcceptGam[iter] <- GamAccep 
    
    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  } 
  
  #)
  #)
  
  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample
}

#-- Fixed x (we know X) Fx

BQBayes.ME_SoFRFuncIndiv <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0 = 0.5, X, g0){
  
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
  met <- c("f1","f2","f3","f4","f5")
  
  
  func <- function(Pik, n=1, size=1){
    which.max(rmultinom(n=1,size=1,prob = Pik))
  }
  funcCmp <- cmpfun(func)
  
  F2 <- function(Pik, n=1, size=1){
    
    apply(Pik,1,funcCmp)
  }
  F2Cmp <- cmpfun(F2)
  
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
  
  
  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  #pn = pn #min(15, ceiling(length(Y)^{1/3.8})+4)
  n = length(Y) 
  Zmod = cbind(1,Z)
  #------------------
  #pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  #pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  pn = c(5, 6, 20, 20)[which(n == c(100, 200,500,1000))]
  alpha.e = alpx
  alpha.w = alpha.W = alpx
  alpha.x = alpha.X = alpx
  alpha.m = alpx
  #-------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt #c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) # 
  Mn_Mod <- M%*%diag(1/Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20,"knots"), 1) ))
  Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a) 
  Wn <- (W%*%bs2)/length(a)
  Xn <- (X%*%bs2)/length(a)
  
  cat("\n #--- Initial values...\n")  
  P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D 
    return(P)
  }
  
  
  InitialValues <- function(Y, W, M, Z, df0=3, pn, Gx, a){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- Z #cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    lmFit <- lm(Y ~ -1 + Zmod + W_i) 
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- residuals.lm(lmFit)
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  InitialValuesQR <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z+Wn)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    plot(a, bs2%*%lmFit$coefficients[-c(1:3)], type="l")
    
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    Tp0 = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd)), upper=c(30, 0.95*max(bnd) ))
    Res$Hess <- Tp0$hessian
    
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  #--- Another version
  InitialValuesQR2 <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    #plot(a, bs2%*%lmFit$coefficients[-c(1:ncol(Z))], type="l")
    
    c_hatW <- lmFit$coefficients
    #Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    #Tp0 = optim(c(.2,-.1, rnorm(n=ncol(Wn))), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd), rep(-7, ncol(Wn)))
    #            , upper=c(30, 0.95*max(bnd),rep(7, ncol(Wn)) ))
    Tp0 = optim(c(.2,-.1, rnorm(n=ncol(Wn))), LogPlostGalLikBet, method = method, ei = c(ei), X = Wn, tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd), rep(-7, ncol(Wn)))
                , upper=c(30, 0.95*max(bnd),rep(7, ncol(Wn)) ))
    Res$Hess <- Tp0$hessian[1:2,1:2]
    Res$HessBet <- Tp0$hessian[-c(1:2),-c(1:2)]
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  #Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  #Thet0 <- InitialValuesQR(Y, W, M, Z, df0=3, pn, Gx=1, a, tau0)
  
  Thet0 <- InitialValuesQR2(Y, W, M, Z, df0=3, pn, Gx=1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  #Kvalues
  #Kc <- 5
  
  Kx = Thet0$Kx               ## number of X  cluster
  Ke = Thet0$Ke               ## number of ei cluster
  Km = Thet0$Km               ## number of ei cluster
  Kw = Thet0$Kw              ## number of omega_i cluster 
  
  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta   #- The intercepts vector of length J
  Thet$Gamma <- runif(n = pn, -4,4)  #rnorm(n=pn) #as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ #rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del#0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  #Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1]); Thet$gamU <- max(bnd[2])
  
  #bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) #c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
  
  Thet$Cik.x =  sample.int(Kx,size=n, replace = T) #Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e =  sample.int(Ke,size=n, replace = T) #Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w = sample.int(Kw,size=n, replace = T) #Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m = sample.int(Km,size=n, replace = T) #Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km
  
  
  Thet$Pik.x = rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)#Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e = rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)#Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w = rdirch(n=1,alpha = rep(alpha.e, Kw)/Kw) #Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m = rdirch(n=1,alpha = rep(alpha.m, Km)/Km)#Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw
  
  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw
  
  Thet$sig2k.e  = nimble::rinvgamma(n=Ke, shape=5/2,scale=8/2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n=Ke, min = .95*Thet$gamL, max=.95*Thet$gamU)
  Thet$p <- 1*(Thet$gamk < 0) + (Thet$tau0 - 1*(Thet$gamk < 0))/(2*pnorm(-abs(Thet$gamk))*exp(0.5*Thet$gamk*Thet$gamk))
  Thet$Ak <- (1-2*Thet$p)/(Thet$p*(1-Thet$p)); Thet$Bk <- 2/(Thet$p*(1-Thet$p)); Thet$Ck <- 1/( 1*(Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean=0, sd=sqrt(Thet$sig2k.e[1])) ; Thet$nui <- rexp(n=n, rate = 1/Thet$sig2k.e[1]) 
  #Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix
  
  Thet$Sigk.w <- repmat(c(.5*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw*pn), nrow = Kw)          #- vectors of length Kw and Kw respectively
  
  Thet$Sigk.x <- repmat(c(.5*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$Muk.x  <- matrix(rnorm(n=Kx*pn), nrow=Kx)   #- vectors of length Kx and Kx respectively
  
  Thet$Sigk.m <- repmat(c(.5*diag(pn)), Km, 1)     #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2*diag(pn)), Km, 1)
  Thet$Muk.m  <- matrix(rnorm(n=Km*pn), nrow = Km)   #- vectors of length Kx and Kx respectively
  
  A <- min(c(c(Wn), c(Mn))) - .5*diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5*diff(range(c(Wn, Mn)))
  #Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  #matrix(rnorm(n = n*pn), ncol=pn)
  #Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5*(Mn+Wn) #Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))
  
  
  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
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
  
  
  #method = "L-BFGS-B"
  #method = "BFGS"
  
  #Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  #Res
  
  #Res$hessian
  #Res$hessian/n
  #solve(Res$hessian/n)
  
  #LogPlostGalLik(c(.05,), c(ei), Thet)
  
  #-------- Update Beta (vector of length pn) 
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #   
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma 
  #   
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #   
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #   
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #   
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)
  
  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together) 
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 = ncol(Z) +1
  Pgam <- P.mat(pn)*tauval
  Sig0.gam0 = diag(pn0); Mu0.gam0 = numeric(pn0+pn);
  #InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))
  
  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.gam0%*%Mu0.gam0))
    
    #c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-6,length(Mu.gam)), ub = rep(6,length(Mu.gam))))
    
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    
  }
  updateGam <- cmpfun(updateGam)
  
  #--- Update Beta (error free covariates)
  #--- Prior parms
  pn0 = ncol(Z) +1
  #Pgam <- P.mat(pn)*tauval
  Sig0.bet0 = 100*diag(pn0); InvSig0.bet0 <- 1.0*as.inverse(Sig0.bet0); Mu0.bet0 = numeric(pn0);
  updateGamBeta <- function(Thet, Y, Zmod, InvSig0.bet0, Mu0.bet0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Thet$X%*%Thet$Gamma - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    #InvSig0.gam0 <- 1.0*as.inverse(Sig0.gam0) #, P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- Zmod 
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.bet0))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.bet0%*%Mu0.bet0))
    
    c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-6,length(Mu.gam)), ub = rep(6,length(Mu.gam))))
    
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    
  }
  updateGamBeta <- cmpfun(updateGamBeta)
  
  #--- Update Gamma individually
  #--- Prior parms
  #pn0 = ncol(Z) +1
  #Pgam <- P.mat(pn)*tauval
  InvSig0.gam0 <- P.mat(pn); Mu0.gam0 = numeric(pn);
  updateGamma0 <- function(Thet, Y, Zmod, InvSig0.gam0, Mu0.gam0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Zmod%*%Thet$BetaZ - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    #InvSig0.gam0 <- 1.0*as.inverse(Sig0.gam0) #, P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- Thet$X 
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0*(1/Thet$siggam)))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.gam0%*%Mu0.gam0))
    
    #c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-6,length(Mu.gam)), ub = rep(6,length(Mu.gam))))
    
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    
  }
  updateGamma0 <- cmpfun(updateGamma0)
  
  #---- update beta using the GAL density instead
  K0 = P.mat(pn)
  SigmaProp <- solve(-Thet0$HessBet)
  diag(SigmaProp) <- abs(diag(SigmaProp)) ; c0 = .75
  SigmaProp <- as.positive.definite(as.symmetric.matrix(SigmaProp))
  #SigmaProp <- diag(pn); c0 = .5
  updateGamma <- function(Thet, Y, Zmod, K0,SigmaProp, c0,g0){
    
    #g0 = 2
    idk = Thet$Cik.e
    #Zmod <-  Z #cbind(1,Z)
    Ydi = Y - Zmod%*%Thet$BetaZ #- Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    ei_old = Ydi - Thet$X%*%Thet$Gamma
    
    Gam_newk <- TruncatedNormal::rtmvnorm(n=1, mu = Thet$Gamma, sigma = c0*SigmaProp, lb = rep(-g0,length(Thet$Gam)), ub = rep(g0,length(Thet$Gam)))
    ei_new = Ydi - Thet$X%*%Gam_newk  
    
    logLkold <- sum(log(mapply(fGal.p0, e = ei_old, sig2 = sqrt(Thet$sig2k.e[idk]), gam = Thet$gamk[idk], p =Thet$p[idk], tau0 =Thet$tau0) + .Machine$double.eps))
    - .5*quad.form(K0*(1/Thet$siggam), Gam_newk) - TruncatedNormal::dtmvnorm(x = Gam_newk, mu = Thet$Gamma, sigma = c0*SigmaProp, lb = rep(-g0,length(Thet$Gam)), ub = rep(g0,length(Thet$Gam)), log=T)
    #LaplacesDemon::dmvnp(x = Gam_newk, mu = rep(0, length(Thet$Gamma)), Omega = K0*(1/Thet$siggam), log = T)
    
    #  (sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
    # - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
    
    logLknew <- sum(log(mapply(fGal.p0, e = ei_new, sig2 = sqrt(Thet$sig2k.e[idk]), gam = Thet$gamk[idk], p =Thet$p[idk], tau0 =Thet$tau0) + .Machine$double.eps))
    - .5*quad.form(K0*(1/Thet$siggam), Thet$Gamma) - TruncatedNormal::dtmvnorm(x = Thet$Gamma , mu = Gam_newk, sigma = c0*SigmaProp, lb = rep(-g0,length(Thet$Gam)), ub = rep(g0,length(Thet$Gam)), log=T)   
    #+ LaplacesDemon::dmvnp(x = Thet$Gamma, mu = rep(0, length(Thet$Gamma)), Omega = K0*(1/Thet$siggam), log = T)
    #+ dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
    #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
    
    uv =  logLknew - logLkold 
    jump <- runif(n=1)
    if(log(jump) < uv){
      res <- c(1, Gam_newk)
    }else{
      res <- c(0, Thet$Gamma)
    }
    res
  }
  updateGamma <- cmpfun(updateGamma)
  
  
  
  #siggam ~ IG(al0, bet0)
  al0 = 0.001
  bet0 = .005
  al0 = 2.5978252 #1
  bet0 = 0.4842963 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    val <- nimble::rinvgamma(n = 1, shape = al, scale = bet)
    ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e 
  #alpha.e = 1.0
  updatePik.e <- function(Thet, alpha.e){
    
    nk <- numeric(length(Thet$Cik.e))
    for(i in 1:length(Thet$Cik.e)){ nk[i] = sum(Thet$Cik.e == i)}
    
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$Cik.e) + nk
    
    rdirichlet(n=1, alpha.enew)
    
  }
  
  #-------- Update Cik.e  
  
  #ei <- Y - Z%*%Thet$Betaz - Thet$X%*%Thet$Gamma
  updateCik.e <- function(i, Thet,ei){
    #Y - Thet$C*abs(Thet$gam)*Thet$si - Thet$A*Thet$nui
    
    #  Yi.tld <- Y[i] - c(crossprod(Thet$Gamma, Thet$X[i,]))
    #pi.k  <- Thet$Pik.e*mapply(dnorm, x = Yi.tld, mean=Thet$muk.e, sd = sqrt(Thet$sig2k.e)) + .Machine$double.eps
    pik <- Thet$Pik.e*mapply(fGal.p0, e = ei[i], sig2 = sqrt(Thet$sig2k.e), gam = Thet$gamk, p = Thet$p, tau0 = Thet$tau0) #+ .Machine$double.eps
    which(rmultinom(n=1, size=1, prob = pik) == 1)
  }
  updateCik.e <- Vectorize(updateCik.e,"i") 
  
  #--- update Si
  #-- calss id for individual i (Matrix with columns of nui and si)
  updatenui <- function(Thet, Y, Z){
    
    ki = Thet$Cik.e
    bi <- ( ((Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ck[ki]*abs(Thet$gamk[ki])*Thet$si)^2)/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])))
    ai = ((Thet$Ak[ki]*Thet$Ak[ki])/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])) + 2/sqrt(Thet$sig2k.e[ki]))
    
    #mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) 
    #val = mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    #ifelse(val < .001, rexp(n = length(muui), rate = 1/sqrt(Thet$sig2k.e[Thet$Cik.e] )), val)
  }
  
  updatesi <- function(Thet, Y, Z){   
    #-- update si
    ki = Thet$Cik.e
    sig2si <- 1 /( 1/Thet$sig2k.e[ki] +  ((Thet$Ck[ki]*Thet$gamk[ki])^2)/ (Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui))
    muui <- sig2si*(Thet$Ck[ki]*abs(Thet$gamk[ki])*(Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ak[ki]*Thet$nui))/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui)
    
    #mapply(truncnorm::rtruncnorm, n=1, a = 0.01, mean = muui, sd = sqrt(sig2si))
    mapply(truncnorm::rtruncnorm, n=1, a = 0.001, mean = muui, sd = sqrt(sig2si))
    
    
  }
  
  
  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  #a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig=5/2; b.sig = 8/2; sdsig.prop = 0.2; sdgam.prop = .3  #2.153640 1.167331
  a.sig=2.153640; b.sig = 1.167331; sdsig.prop = 0.2; sdgam.prop = .3 # 7.133445 4.303263
  a.sig=7.133445; b.sig = 4.303263; sdsig.prop = 0.2; sdgam.prop = .3 #
  updatesig_gam <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop){
    
    a0 = 0.001
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      sig_newk <- truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gam <- Vectorize(updatesig_gam,"k")
  
  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10,10,10)*1
  varxi <- c(10,10,10)*.25
  #varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi){
    
    a0 = 0.1
    ab = 5
    #varxi = 5
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]),Thet$gamk[k]),sigma = Sdsig.prop*varxi[k], lb = c(a0,0.95*Thet$gamL), ub = c(ab, .95*Thet$gamU)  )
      
      
      sig_newk <- Val[1] #truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] #truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu=c(sig_newk, gam_newk), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE) )
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes,"k")
  
  
  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices
  InvCov <- function(Mat){
    K <- nrow(Mat)
    J <- round(sqrt(ncol(Mat)))
    Res0 = matrix(0, nrow = K, ncol=ncol(Mat))
    for(k in 1:K){ Res0[k,] <- c(as.inverse(as.symmetric.matrix(matrix(Mat[k,],ncol=J, byrow=F)))) }
    Res0
  }
  
  
  #----------------------------------------------
  # Update things related to Wi 
  #--------------------------------------------
  #Piw
  #Cik.w
  #--- Update Pik.x  
  #alpha.W = alpha.w = 1.0
  updatePik.w <- function(Thet, alpha.W){
    Ku = nrow(Thet$Muk.w)
    nk <- numeric(Ku)
    for(k in 1:Ku){ nk[k] = sum(Thet$Cik.w == k)}
    
    #update the Pis
    alpha.wnew <- alpha.W/length(Thet$muk.w) + nk
    
    rdirichlet(n=1, alpha.wnew)
    
  }
  
  #--- Update Ci.k mixture of normals 
  dmvnormVec <- function(x, Mean, SigmaVc){
    Ku <- nrow(Mean)
    Val <- numeric(Ku)
    for(k in 1:Ku){ Val[k] <- mvtnorm::dmvnorm(x = x, mean = c(Mean[k,]), sigma = as.symmetric.matrix(matrix(SigmaVc[k,],ncol=length(x),byrow=F))) }
    Val
  }
  #dmvnormVec <- Vectorize(dmvnormVec,"k")
  #dmvnormVec(x = rep(0,6), Mn=Thet$Muk.w,SigmaVc = Thet$Sigk.w)
  
  updateCik.w <- function(i, Thet, W){ # mixture of normals 
    #Sig <- matrix(Thet$Sigk.w[id,], ncol = ncol(W), byrow = F)
    #kw = nrow(Thet$Muk.w)
    xi =  c(W[i,] - Thet$X[i,])
    #pi.k  = Thet$Pik.w*mapply(dmvnormVec, k = 1:Kw, MoreArgs = list(x = xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w))
    pi.k  = Thet$Pik.w*dmvnormVec(x=xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  
  #updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  #Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol=pn, nrow=pn));inSig0.w = as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.w <- function(Thet, W, Mu0.w, inSig0.w, Sig0.w){
    Ku = length(Thet$Pik.w)
    J = ncol(Thet$X)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol=J, nrow=J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = W - Thet$X 
    #Muk <- matrix(0, ncol=J,nrow=Ku)
    SigK <- list()
    for(k in 1:Ku){
      idk <- which(Thet$Cik.w == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow= nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.w)) ## sum of matrices inverse
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi.tld[idk,])) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
        Mu.k[k,] <- c(SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.w%*%Mu0.w))      ## some of mat
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(apply(t(Yi.tld[idk,]), 2, sum)) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
      }else{
        SigK[[k]] <- Sig0.w
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.w
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL; for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.w[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.w) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- 1.0*as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], sigma = SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.w[-Ku]) %*% SimMU.k)/Thet$Pik.w[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.w <- cmpfun(updateMuk.w)
  
  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w = pn + 2; Psi0.w = 0.5*(nu0.w-pn-1)*cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  updateSigk.w <- function(Thet, W, nu0.w, Psi0.w){
    
    Ku = nrow(Thet$Muk.w)
    m = ncol(W)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Wtl <- W - Thet$X - Thet$Muk.w[Thet$Cik.w,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.w == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.w  + nk
        Yi0 = matrix(c(Wtl[id,]), nrow= nk, byrow=F)
        Psin.w <- as.positive.definite(Psi0.w + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.w))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.w, Psi0.w))
      }
    }
    Res
  }
  updateSigk.w <- cmpfun(updateSigk.w)
  
  #----------------------------------------------
  # Update things related to M 
  #--------------------------------------------  
  
  #--- update Pik.m for epsilon_i
  #alpha.m = 1.0
  updatePik.m <- function(Thet, alpha.m){
    nk <- numeric(nrow(Thet$Muk.m))
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    
    #update the Pis
    alpha.mnew <- alpha.m/nrow(Thet$Muk.m) + nk
    
    c(rdirichlet(n=1, alpha.mnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  
  updateCik.m <- function(i, Thet, M){ # mixture of normals 
    #id <- Thet$Cik.m[i]
    #Sig <- matrix(Thet$Sigk.m[id,], ncol = ncol(M), byrow = F)
    Km = nrow(Thet$Muk.m)
    xi = M[i,] - Thet$Delta*Thet$X[i,]
    #pi.k  = Thet$Pik.m*mapply(dmvnormVec, k = 1:Km, MoreArgs = list(x = xi , Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m))
    pi.k  = Thet$Pik.m*dmvnormVec(x = xi, Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m = rep(0, pn);  Sig0.m = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.m = as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.m <- function(Thet, M, Mu0.m, inSig0.m, Sig0.m){
    #cat(dim(inSig0.m),"(--1)\n")
    Ku = length(Thet$Pik.m)
    J = ncol(M)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol = J, nrow = J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = M - Thet$X%*%diag(Thet$Betadel) 
    Muk <- matrix(0, ncol=J, nrow=Ku)
    for(k in 1:Ku){
      idk <- which(Thet$Cik.m == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow = nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.m)) ## sum of matrices inverse
        #cat(dim(SigK[[k]]),"(--2 \n")
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.m%*%Mu0.m)      ## some of mat
        #cat(dim(SigKWg),"(--3 \n")
      }else{
        SigK[[k]] <- Sig0.m
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.m
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL;  for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.m[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.m) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.m[-Ku]) %*% SimMU.k)/Thet$Pik.m[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.m <- cmpfun(updateMuk.m)
  
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m = pn + 2; Psi0.m = .5*(nu0.m-pn-1)*as.symmetric.matrix(cov(Wn))  #diag(pn)
  
  updateSigk.m <- function(Thet, M, nu0.m, Psi0.m){
    
    Ku = nrow(Thet$Muk.m)
    m = ncol(M)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Mtl <- M - Thet$X%*%diag(Thet$Betadel) - Thet$Muk.m[Thet$Cik.m,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.m == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.m  + nk
        Yi0 <- matrix(c(Mtl[id,]), nrow=nk, byrow=F)
        Psin.m <- as.positive.definite(Psi0.m + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.m))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.m, Psi0.m))
      }
    }
    Res
  }
  updateSigk.m <- cmpfun(updateSigk.m)
  
  #---- Update things related to delta
  quadFormVec <- function(i,X,sigma,Val){
    id = Val[i]
    as.symmetric.matrix(quad.form(as.inverse(as.symmetric.matrix(matrix(sigma[id,], ncol = length(X[i,]),byrow=F))), diag(c(X[i,])))) 
  }
  quadFormVec <- Vectorize(quadFormVec,"i",SIMPLIFY = F) ## Resulting in a list if precision matrices
  quadFormVec <- cmpfun(quadFormVec)
  
  quadFormVec3 <- function(i,Mi,X,sigma,Val){
    id = Val[i]
    c(quad.3form.inv(matrix(sigma[id,], ncol = length(X[i,]),byrow=F), diag(X[i,]),Mi[i,]))
  }
  quadFormVec3 <- Vectorize(quadFormVec3,"i", SIMPLIFY = F)
  quadFormVec3 <- cmpfun(quadFormVec3)
  
  #SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),   
  # prior values
  mu0.del = 1.0; #sig02.del = 100
  updateDelta <- function(Thet, M, mu0.del, sig02.del){
    n = nrow(M)
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X), Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    sig20 <- 1/( 1/sig02.del + sum(Res))
    mu <- sig20*(sum(Res3) + mu0.del/sig02.del)
    rtruncnorm(n=1, a=0, b=10, mean = mu, sd = sqrt(sig20))
    
    #c(sum(Res3), sum(Res),mu, sqrt(sig20), rtruncnorm(n=1, a=0, b=Inf, mean = mu, sd = sqrt(sig20)))
    
  }
  updateDelta <- cmpfun(updateDelta)
  
  
  # prior values
  mu0.del = rep(0,length(Thet$Betadel)); #sig02.del = 20
  Sig0.del = P.mat(length(Thet$Betadel))*.1   #Thet$siggam) # Inverse or precision covariance matrix
  
  updateFunDelta <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDelta <- cmpfun(updateFunDelta)
  
  
  SimDElVec <- function(i, Mi, X, Sigma, Val){
    id <- Val[i]
    Ot = mvtnorm::rmvnorm(n = 1,mean = Mi[i,],sigma = as.symmetric.matrix(matrix(Sigma[id,], ncol = length(X[i,]),byrow=F)))
  }
  SimDElVec <- Vectorize(SimDElVec, "i")
  
  #-- 
  DeltComp <- function(k,Xst, X){
    abs(sum(Xst[,k]*X[,k])/(sum(X[,k]^2)))
    #abs(diag(as.inverse(crossprod(X))%*%crossprod(X,Xst)))
  }
  DeltComp <- Vectorize(DeltComp,"k")
  
  updateFunDeltaSt <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Xst <- base::t(SimDElVec(1:nrow(M), Mitl, Thet$X, Thet$Sigk.m, Thet$Cik.m)) ## K by n matrix and now I get a n by K
    c(DeltComp(1:ncol(Thet$X), Xst, Thet$X))
    #c(DeltComp(Xst, Thet$X))
    
    # Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    # Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    # Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    # Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    # c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDeltaSt  <- cmpfun(updateFunDeltaSt )
  
  
  #----------------------------------------------
  # Update things related to X 
  #--------------------------------------------------  
  #--- update Pik.m for epsilon_i
  #alpha.x <- 1.0
  updatePik.x <- function(Thet, alpha.x){
    nk <- numeric(nrow(Thet$Muk.x))
    for(k in 1:nrow(Thet$Muk.x)){ nk[k] = sum(Thet$Cik.x == k)}
    
    #update the Pis
    alpha.xnew <- alpha.x/nrow(Thet$Muk.x) + nk
    
    c(rdirichlet(n=1, alpha.xnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  updateCik.x <- function(i, Thet){ # mixture of normals 
    #Kx = ncol(Thet$X) # J x J
    #id <- Thet$Cik.x[i]
    #Sig <- matrix(Thet$Sigk.x[id,], ncol = length(Thet$Gamma), byrow = F)
    Xi = Thet$X[i,]
    #pi.k  = Thet$Pik.x*mapply(dmvnormVec, k = 1:Kx, x = Xi,  Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x)
    pi.k  = Thet$Pik.x*dmvnormVec(x=Xi,Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x = numeric(pn); Sig0.x = .5*diag(pn); inSig0.x = 2*diag(pn)#as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x = rep(0, pn);  Sig0.x = .25*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w)+colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.x = as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.x <- function(Thet, Mu0.x, inSig0.x, Sig0.x){
    K = length(Thet$Pik.x)
    J = ncol(Thet$X)
    Res0 <- matrix(0, nrow = K, ncol = J)
    Mu.k <- matrix(0, nrow = K, ncol = J) # K x J
    #SigKWg <- matrix(0, ncol = ncol(Y),nrow=ncol(Y)) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = Thet$X 
    Muk <- matrix(0, ncol = J, nrow = K)
    for(k in 1:K){
      idk <- which(Thet$Cik.x == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 <- matrix(c(Yi.tld[idk,]), nrow=nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[k,], ncol=J, byrow=F)))
        SigK <- as.inverse(as.symmetric.matrix((nk*TpSi + inSig0.x))) ## sum of matrices inverse
        Muk <- SigK%*%(crossprod(TpSi, c(colSums(Yi0))) + inSig0.x%*%Mu0.x)      ## some of mat
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Muk, sigma = SigK))
      }else{
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Mu0.x, sigma = Sig0.x ))
      }
    }
    #--- we impose some constrainsts on the mean
    Mu.k
    
  }
  updateMuk.x <- cmpfun(updateMuk.x)
  
  nu0.x <- pn + 2 ; Psi0.x <- .5*diag(pn) #as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2 ; Psi0.x <- .5*(nu0.x-pn-1)*cov(.5*(Wn+Mn)) #as.symmetric.matrix(.25*cov(Wn))
  updateSigk.x <- function(Thet, nu0.x, Psi0.x){
    
    Ku = nrow(Thet$Muk.x)
    m = ncol(Thet$Muk.x)
    Xtl <- Thet$X - Thet$Muk.x[Thet$Cik.x,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.x == i)
      nk = length(id)
      if(nk > 0){
        nux <- nu0.x  + nk
        Yi0 <- matrix(c(Xtl[id,]), nrow=nk, byrow=F)
        Psin.x <- Psi0.x + crossprod(Yi0)
        Res[i, ] <- c(rinvwishart(nux, Psin.x))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.x, Psi0.x))
      }
    }
    Res
  }
  updateSigk.x <- cmpfun(updateSigk.x)
  # 
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn))) 
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn))) 
  
  A <- min(c(Wn, Mn)) - .1*diff(range(c(Wn, Mn))) 
  B <- max(c(Wn, Mn)) + .1*diff(range(c(Wn, Mn))) 
  
  udapateXi <- function(i, Thet, Y, W, M, Z, A=A, B=B){
    ie =Thet$Cik.e[i]
    iw = Thet$Cik.w[i]
    im = Thet$Cik.m[i]
    ix = Thet$Cik.x[i]
    J = ncol(Thet$X)
    
    Zmod <- Z #cbind(1,Z)
    Ytl = (Y - Zmod%*%Thet$BetaZ - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Wtl = W - Thet$Muk.w[Thet$Cik.w,]
    Mtl = (M - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
    Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
    Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
    
    #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
    Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
    
    Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))) 
    
    #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi <- cmpfun(udapateXi)
  
  
  udapateXi2A <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2A <- cmpfun(udapateXi2A)
  
  ## This approach of simulating Xi only uses W, M, and X
  
  udapateXi2 <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #Mu <- c(Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2 <- cmpfun(udapateXi2)
  
  
  udapateXi2V <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    #Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    #GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- (Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi2V <- Vectorize(udapateXi2V,"i")
  udapateXi2V <- cmpfun(udapateXi2V)
  
  #--- Write Rcpp function to speed things up
  {
    #
    
    
  }
  
  
  
  #Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)
  
  ##----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  #Nsim <- 5000
  #Nsim = 5000 
  MCMCSample <- list()
  
  #-- for X
  MCMCSample$X <- matrix(0,nrow = Nsim,ncol=length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined) 
  MCMCSample$Muk.x <- matrix(0,nrow = Nsim, ncol = Kx*pn)
  MCMCSample$Sigk.x <- matrix(0,nrow = Nsim,ncol=Kx*pn*pn)
  MCMCSample$Pik.x <- matrix(0,nrow = Nsim,ncol=Kx)
  MCMCSample$Cik.x <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for W
  MCMCSample$Muk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn)
  MCMCSample$Sigk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn*pn)
  MCMCSample$Pik.w <- matrix(0,nrow = Nsim,ncol=Kw)
  MCMCSample$Cik.w <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for M
  MCMCSample$Muk.m <- matrix(0,nrow = Nsim,ncol=Km*pn)
  MCMCSample$Sigk.m <- matrix(0,nrow = Nsim,ncol=Km*pn*pn)
  MCMCSample$Pik.m <- matrix(0,nrow = Nsim,ncol=Km)
  MCMCSample$Cik.m <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow=Nsim, ncol=length(Thet$Betadel))
  
  # For Y
  MCMCSample$gamk <- matrix(0,nrow = Nsim, ncol=Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Pik.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Cik.e <- matrix(0,nrow = Nsim, ncol=nrow(Wn))
  
  MCMCSample$Gamma <- matrix(0,nrow = Nsim, ncol=pn) #length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)
  MCMCSample$AcceptGam <- rep(NA, Nsim)
  
  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  # 
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  # 
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  # 
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  #Accept <- 
  # 
  MCMCSample$AcceptX <- matrix(0,nrow = Nsim, ncol=Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  #stop()
  #------------------------------------------------------------------
  #Nsim = 5000 
  #Nsim = 1000
  #Nsim = 3000
  #library(profvis)
  #profvis(
  #system.time(
  #cat("\n #--- MCMC is starting ----> \n")
  
  for(iter in 1:Nsim){
    if(iter %% 500 == 0){cat("iter=",iter,"\n")}
    #----------------------------------------------
    # Update things related to X 
    #--------------------------------------------  
    #--- update Pik.m for epsilon_i
    Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    
    #--- update Cik.m for epsilon_i based on mixture of normals
    Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x)) 
    
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    
    Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    Thet$InvSigk.x <- InvCov(Thet$Sigk.x)
    
    #Ytd = Y - Zmod%*%Thet$BetaZ
    #if(iter%%10 == 0){
    Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B) # Xn #
    #Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    #Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    #Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w, 
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)
    
    #Thet$X <- UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,]) 
    
    #}
    #dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))
    
    #Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    #Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    #Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    #Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w, 
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J))) 
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Kw)
    for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    #update the Pis
    alpha.wnew <- alpha.W/Kw + nk
    Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    
    #--- Update Ci.k mixture of normals
    #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Km)
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    #update the Pis
    alpha.mnew <- alpha.m/Kw + nk
    Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    
    #--- Update Ci.k mixture of normals
    #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m)) 
    #Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))  
    #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    #if(iter%%50 == 0){
    Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    MCMCSample$Betdel[iter,] <- Thet$Betadel
    #}
    #---------------------------------------------
    #cat("#--- update things related to Y or epsilon ")
    #--------------------------------------------- 
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for(k in 1:length(Thet$sig2k.e)){ nk[k] = sum(Thet$Cik.e == k)}
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n=1, alpha.enew))
    
    #update the Cik.e
    ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
    #Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) #rep(1,n) #
    #Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    #update nui  and si
    #Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    #nuisi <- updatenuisi(Thet, Y, Zmod)
    #Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    #update Sigk.e
    #Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    #update sigk and gamk
    #--- update jointly sig2ke and gamke
    #Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi)
    Thet$sig2k.e <- Temp[1,];  Thet$gamk <- Temp[2,]; Thet$p <- Temp[3,]; MCMCSample$AcceptX[iter,] <- Temp[4,]; Thet$Ak <- Temp[5,]; Thet$Bk <- Temp[6,]; Thet$Ck <- Temp[7,];
    
    #update Gamma
    #Thet$siggam <- sig02.del
    #Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0)
    # Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    # Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    #Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    
    Thet$BetaZ <- updateGamBeta(Thet, Y, Zmod, InvSig0.bet0, Mu0.bet0)#Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Tep <- updateGamma(Thet, Y, Zmod, K0,SigmaProp, c0, g0)
    #print(Tep[length(Tep)])
    GamAccep <- Tep[1]
    Thet$Gamma <- Tep[-1]  #updateGamma(Thet, Y, Zmod, K0,SigmaProp, c0)
    #updateGamma(Thet, Y, Zmod, InvSig0.gam0, Mu0.gam0)  # Tp0[-c(1:ncol(Zmod))]
    
    Thet$siggam <-  updateSigGam(Thet, al0, bet0, order = 2)# sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#
    
    #Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    # 
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    # 
    # #--- upodate beta_{z,m}
    # #--  Prior 
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    # 
    # #--- update X_i 
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX) 
    # 
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]); 
    # 
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb 
    
    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x))  ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise
    
    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w
    
    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    #MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e
    MCMCSample$AcceptGam[iter] <- GamAccep 
    
    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  } 
  
  #)
  #)
  
  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample
}




#-- using FUI approach 
#-- Fixed x (we know X) Fx
#sig02.del = 0.1; tauval = .1;  Delt= rep(1, ncol(M)); alpx=1; tau0 = 0.5
BQBayes.ME_SoFRFuncIndivFxFUI <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0 = 0.5, X,g0){
  
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
  met <- c("f1","f2","f3","f4","f5")
  
  
  func <- function(Pik, n=1, size=1){
    which.max(rmultinom(n=1,size=1,prob = Pik))
  }
  funcCmp <- cmpfun(func)
  
  F2 <- function(Pik, n=1, size=1){
    
    apply(Pik,1,funcCmp)
  }
  F2Cmp <- cmpfun(F2)
  
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
      (2*p*(1-p)/sig2)*(exp(-est*p.neg + .5*(gam^2)*(p.neg/p.pos)^2 + pnorm(q = a2-a3, log.p = T) + Rmpfr::log1mexp( abs(pnorm(q = a2-a3, log.p = T) - pnorm(q = a2, log.p = T)) ) ) + exp(-p.pos*est + 0.5*(gam^2) + pnorm(-abs(gam) + a3, log.p = T)))
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
  
  
  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  #pn = pn #min(15, ceiling(length(Y)^{1/3.8})+4)
  n = length(Y) 
  Zmod = cbind(1,Z)
  #------------------
  #pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  #pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  pn = c(5, 6, 20, 20)[which(n == c(100, 200,500,1000))]
  alpha.e = alpx
  alpha.w = alpha.W = alpx
  alpha.x = alpha.X = alpx
  alpha.m = alpx
  #-------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt #c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) # 
  Mn_Mod <- M%*%diag(1/Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20,"knots"), 1) ))
  Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a) 
  Wn <- (W[,,1]%*%bs2)/length(a)
  Xn <- (X%*%bs2)/length(a)
  
  cat("\n #--- Initial values...\n")  
  P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D 
    return(P)
  }
  
  
  InitialValues <- function(Y, W, M, Z, df0=3, pn, Gx, a){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- Z #cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    lmFit <- lm(Y ~ -1 + Zmod + W_i) 
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- residuals.lm(lmFit)
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  InitialValuesQR <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z+Wn)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    plot(a, bs2%*%lmFit$coefficients[-c(1:3)], type="l")
    
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    Tp0 = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd)), upper=c(30, 0.95*max(bnd) ))
    Res$Hess <- Tp0$hessian
    
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  #--- Another version
  InitialValuesQR2 <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    #plot(a, bs2%*%lmFit$coefficients[-c(1:ncol(Z))], type="l")
    
    c_hatW <- lmFit$coefficients
    #Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    #Tp0 = optim(c(.2,-.1, rnorm(n=ncol(Wn))), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd), rep(-7, ncol(Wn)))
    #            , upper=c(30, 0.95*max(bnd),rep(7, ncol(Wn)) ))
    Tp0 = optim(c(.2,-.1, rnorm(n=ncol(Wn))), LogPlostGalLikBet, method = method, ei = c(ei), X = Wn, tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd), rep(-7, ncol(Wn)))
                , upper=c(30, 0.95*max(bnd),rep(7, ncol(Wn)) ))
    Res$Hess <- Tp0$hessian[1:2,1:2]
    Res$HessBet <- Tp0$hessian[-c(1:2),-c(1:2)]
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  #Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  #Thet0 <- InitialValuesQR(Y, W, M, Z, df0=3, pn, Gx=1, a, tau0)
  W01 <- apply(W, c(1,2), mean)
  
  Thet0 <- InitialValuesQR2(Y, W01, M, Z, df0=3, pn, Gx=1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  #Kvalues
  #Kc <- 5
  
  Kx = Thet0$Kx               ## number of X  cluster
  Ke = Thet0$Ke               ## number of ei cluster
  Km = Thet0$Km               ## number of ei cluster
  Kw = Thet0$Kw              ## number of omega_i cluster 
  
  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta   #- The intercepts vector of length J
  Thet$Gamma <- runif(n = pn, -4,4)  #rnorm(n=pn) #as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ #rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del#0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  #Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1]); Thet$gamU <- max(bnd[2])
  
  #bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) #c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
  
  Thet$Cik.x =  sample.int(Kx,size=n, replace = T) #Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e =  sample.int(Ke,size=n, replace = T) #Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w = sample.int(Kw,size=n, replace = T) #Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m = sample.int(Km,size=n, replace = T) #Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km
  
  
  Thet$Pik.x = rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)#Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e = rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)#Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w = rdirch(n=1,alpha = rep(alpha.e, Kw)/Kw) #Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m = rdirch(n=1,alpha = rep(alpha.m, Km)/Km)#Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw
  
  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw
  
  Thet$sig2k.e  = nimble::rinvgamma(n=Ke, shape=5/2,scale=8/2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n=Ke, min = .95*Thet$gamL, max=.95*Thet$gamU)
  Thet$p <- 1*(Thet$gamk < 0) + (Thet$tau0 - 1*(Thet$gamk < 0))/(2*pnorm(-abs(Thet$gamk))*exp(0.5*Thet$gamk*Thet$gamk))
  Thet$Ak <- (1-2*Thet$p)/(Thet$p*(1-Thet$p)); Thet$Bk <- 2/(Thet$p*(1-Thet$p)); Thet$Ck <- 1/( 1*(Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean=0, sd=sqrt(Thet$sig2k.e[1])) ; Thet$nui <- rexp(n=n, rate = 1/Thet$sig2k.e[1]) 
  #Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix
  
  Thet$Sigk.w <- repmat(c(.5*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw*pn), nrow = Kw)          #- vectors of length Kw and Kw respectively
  
  Thet$Sigk.x <- repmat(c(.5*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$Muk.x  <- matrix(rnorm(n=Kx*pn), nrow=Kx)   #- vectors of length Kx and Kx respectively
  
  Thet$Sigk.m <- repmat(c(.5*diag(pn)), Km, 1)     #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2*diag(pn)), Km, 1)
  Thet$Muk.m  <- matrix(rnorm(n=Km*pn), nrow = Km)   #- vectors of length Kx and Kx respectively
  
  A <- min(c(c(Wn), c(Mn))) - .5*diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5*diff(range(c(Wn, Mn)))
  #Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  #matrix(rnorm(n = n*pn), ncol=pn)
  #Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5*(Mn+Wn) #Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))
  
  
  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
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
  
  
  #method = "L-BFGS-B"
  #method = "BFGS"
  
  #Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  #Res
  
  #Res$hessian
  #Res$hessian/n
  #solve(Res$hessian/n)
  
  #LogPlostGalLik(c(.05,), c(ei), Thet)
  
  #-------- Update Beta (vector of length pn) 
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #   
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma 
  #   
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #   
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #   
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #   
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)
  
  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together) 
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 = ncol(Z) +1
  Pgam <- P.mat(pn)*tauval
  Sig0.gam0 = diag(pn0); Mu0.gam0 = numeric(pn0+pn);
  #InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))
  
  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.gam0%*%Mu0.gam0))
    
    #c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-6,length(Mu.gam)), ub = rep(6,length(Mu.gam))))
    
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    
  }
  updateGam <- cmpfun(updateGam)
  
  #--- Update Beta (error free covariates)
  #--- Prior parms
  pn0 = ncol(Z) +1
  #Pgam <- P.mat(pn)*tauval
  Sig0.bet0 = 100*diag(pn0); InvSig0.bet0 <- 1.0*as.inverse(Sig0.bet0); Mu0.bet0 = numeric(pn0);
  updateGamBeta <- function(Thet, Y, Zmod, InvSig0.bet0, Mu0.bet0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Thet$X%*%Thet$Gamma - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    #InvSig0.gam0 <- 1.0*as.inverse(Sig0.gam0) #, P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- Zmod 
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.bet0))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.bet0%*%Mu0.bet0))
    
    c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-6,length(Mu.gam)), ub = rep(6,length(Mu.gam))))
    
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    
  }
  updateGamBeta <- cmpfun(updateGamBeta)
  
  #--- Update Gamma individually
  #--- Prior parms
  #pn0 = ncol(Z) +1
  #Pgam <- P.mat(pn)*tauval
  InvSig0.gam0 <- P.mat(pn); Mu0.gam0 = numeric(pn);
  updateGamma0 <- function(Thet, Y, Zmod, InvSig0.gam0, Mu0.gam0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Zmod%*%Thet$BetaZ - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    #InvSig0.gam0 <- 1.0*as.inverse(Sig0.gam0) #, P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- Thet$X 
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0*(1/Thet$siggam)))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.gam0%*%Mu0.gam0))
    
    #c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-6,length(Mu.gam)), ub = rep(6,length(Mu.gam))))
    
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    
  }
  updateGamma0 <- cmpfun(updateGamma0)
  
  #---- update beta using the GAL density instead
  K0 = P.mat(pn)
  SigmaProp <- solve(-Thet0$HessBet)
  diag(SigmaProp) <- abs(diag(SigmaProp)) ; c0 = .5
  SigmaProp <- as.positive.definite(as.symmetric.matrix(SigmaProp))
  #SigmaProp <- diag(pn); c0 = .5
  updateGamma <- function(Thet, Y, Zmod, K0,SigmaProp, c0, g0){
    
    #g0 = 15
    idk = Thet$Cik.e
    #Zmod <-  Z #cbind(1,Z)
    Ydi = Y - Zmod%*%Thet$BetaZ #- Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    ei_old = Ydi - Thet$X%*%Thet$Gamma
    
    Gam_newk <- TruncatedNormal::rtmvnorm(n=1, mu = Thet$Gamma, sigma = c0*SigmaProp, lb = rep(-g0,length(Thet$Gam)), ub = rep(g0,length(Thet$Gam)))
    ei_new = Ydi - Thet$X%*%Gam_newk  
    
    logLkold <- sum(log(mapply(fGal.p0, e = ei_old, sig2 = sqrt(Thet$sig2k.e[idk]), gam = Thet$gamk[idk], p =Thet$p[idk], tau0 =Thet$tau0) + .Machine$double.eps))
    - .5*quad.form(K0*(1/Thet$siggam), Gam_newk) - TruncatedNormal::dtmvnorm(x = Gam_newk, mu = Thet$Gamma, sigma = c0*SigmaProp, lb = rep(-g0,length(Thet$Gam)), ub = rep(g0,length(Thet$Gam)), log=T)
    #LaplacesDemon::dmvnp(x = Gam_newk, mu = rep(0, length(Thet$Gamma)), Omega = K0*(1/Thet$siggam), log = T)
    
    #  (sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
    # - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
    
    logLknew <- sum(log(mapply(fGal.p0, e = ei_new, sig2 = sqrt(Thet$sig2k.e[idk]), gam = Thet$gamk[idk], p =Thet$p[idk], tau0 =Thet$tau0) + .Machine$double.eps))
    - .5*quad.form(K0*(1/Thet$siggam), Thet$Gamma) - TruncatedNormal::dtmvnorm(x = Thet$Gamma , mu = Gam_newk, sigma = c0*SigmaProp, lb = rep(-g0,length(Thet$Gam)), ub = rep(g0,length(Thet$Gam)), log=T)   
    #+ LaplacesDemon::dmvnp(x = Thet$Gamma, mu = rep(0, length(Thet$Gamma)), Omega = K0*(1/Thet$siggam), log = T)
    #+ dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
    #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
    
    uv =  logLknew - logLkold 
    jump <- runif(n=1)
    if(log(jump) < uv){
      res <- c(1, Gam_newk)
    }else{
      res <- c(0, Thet$Gamma)
    }
    res
  }
  updateGamma <- cmpfun(updateGamma)
  
  
  
  #siggam ~ IG(al0, bet0)
  al0 = 0.001
  bet0 = .005
  al0 = 2.5978252 #1
  bet0 = 0.4842963 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    val <- nimble::rinvgamma(n = 1, shape = al, scale = bet)
    ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e 
  #alpha.e = 1.0
  updatePik.e <- function(Thet, alpha.e){
    
    nk <- numeric(length(Thet$Cik.e))
    for(i in 1:length(Thet$Cik.e)){ nk[i] = sum(Thet$Cik.e == i)}
    
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$Cik.e) + nk
    
    rdirichlet(n=1, alpha.enew)
    
  }
  
  #-------- Update Cik.e  
  
  #ei <- Y - Z%*%Thet$Betaz - Thet$X%*%Thet$Gamma
  updateCik.e <- function(i, Thet,ei){
    #Y - Thet$C*abs(Thet$gam)*Thet$si - Thet$A*Thet$nui
    
    #  Yi.tld <- Y[i] - c(crossprod(Thet$Gamma, Thet$X[i,]))
    #pi.k  <- Thet$Pik.e*mapply(dnorm, x = Yi.tld, mean=Thet$muk.e, sd = sqrt(Thet$sig2k.e)) + .Machine$double.eps
    pik <- Thet$Pik.e*mapply(fGal.p0, e = ei[i], sig2 = sqrt(Thet$sig2k.e), gam = Thet$gamk, p = Thet$p, tau0 = Thet$tau0) #+ .Machine$double.eps
    which(rmultinom(n=1, size=1, prob = pik) == 1)
  }
  updateCik.e <- Vectorize(updateCik.e,"i") 
  
  #--- update Si
  #-- calss id for individual i (Matrix with columns of nui and si)
  updatenui <- function(Thet, Y, Z){
    
    ki = Thet$Cik.e
    bi <- ( ((Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ck[ki]*abs(Thet$gamk[ki])*Thet$si)^2)/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])))
    ai = ((Thet$Ak[ki]*Thet$Ak[ki])/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])) + 2/sqrt(Thet$sig2k.e[ki]))
    
    #mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) 
    #val = mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    #ifelse(val < .001, rexp(n = length(muui), rate = 1/sqrt(Thet$sig2k.e[Thet$Cik.e] )), val)
  }
  
  updatesi <- function(Thet, Y, Z){   
    #-- update si
    ki = Thet$Cik.e
    sig2si <- 1 /( 1/Thet$sig2k.e[ki] +  ((Thet$Ck[ki]*Thet$gamk[ki])^2)/ (Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui))
    muui <- sig2si*(Thet$Ck[ki]*abs(Thet$gamk[ki])*(Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ak[ki]*Thet$nui))/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui)
    
    #mapply(truncnorm::rtruncnorm, n=1, a = 0.01, mean = muui, sd = sqrt(sig2si))
    mapply(truncnorm::rtruncnorm, n=1, a = 0.001, mean = muui, sd = sqrt(sig2si))
    
    
  }
  
  
  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  #a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig=5/2; b.sig = 8/2; sdsig.prop = 0.2; sdgam.prop = .3  #2.153640 1.167331
  a.sig=2.153640; b.sig = 1.167331; sdsig.prop = 0.2; sdgam.prop = .3 # 7.133445 4.303263
  a.sig=7.133445; b.sig = 4.303263; sdsig.prop = 0.2; sdgam.prop = .3 #
  updatesig_gam <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop){
    
    a0 = 0.001
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      sig_newk <- truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gam <- Vectorize(updatesig_gam,"k")
  
  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10,10,10)*1
  varxi <- c(10,10,10)*.25
  #varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi){
    
    a0 = 0.1
    ab = 5
    #varxi = 5
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]),Thet$gamk[k]),sigma = Sdsig.prop*varxi[k], lb = c(a0,0.95*Thet$gamL), ub = c(ab, .95*Thet$gamU)  )
      
      
      sig_newk <- Val[1] #truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] #truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu=c(sig_newk, gam_newk), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE) )
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes,"k")
  
  
  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices
  InvCov <- function(Mat){
    K <- nrow(Mat)
    J <- round(sqrt(ncol(Mat)))
    Res0 = matrix(0, nrow = K, ncol=ncol(Mat))
    for(k in 1:K){ Res0[k,] <- c(as.inverse(as.symmetric.matrix(matrix(Mat[k,],ncol=J, byrow=F)))) }
    Res0
  }
  
  
  #----------------------------------------------
  # Update things related to Wi 
  #--------------------------------------------
  #Piw
  #Cik.w
  #--- Update Pik.x  
  #alpha.W = alpha.w = 1.0
  updatePik.w <- function(Thet, alpha.W){
    Ku = nrow(Thet$Muk.w)
    nk <- numeric(Ku)
    for(k in 1:Ku){ nk[k] = sum(Thet$Cik.w == k)}
    
    #update the Pis
    alpha.wnew <- alpha.W/length(Thet$muk.w) + nk
    
    rdirichlet(n=1, alpha.wnew)
    
  }
  
  #--- Update Ci.k mixture of normals 
  dmvnormVec <- function(x, Mean, SigmaVc){
    Ku <- nrow(Mean)
    Val <- numeric(Ku)
    for(k in 1:Ku){ Val[k] <- mvtnorm::dmvnorm(x = x, mean = c(Mean[k,]), sigma = as.symmetric.matrix(matrix(SigmaVc[k,],ncol=length(x),byrow=F))) }
    Val
  }
  #dmvnormVec <- Vectorize(dmvnormVec,"k")
  #dmvnormVec(x = rep(0,6), Mn=Thet$Muk.w,SigmaVc = Thet$Sigk.w)
  
  updateCik.w <- function(i, Thet, W){ # mixture of normals 
    #Sig <- matrix(Thet$Sigk.w[id,], ncol = ncol(W), byrow = F)
    #kw = nrow(Thet$Muk.w)
    xi =  c(W[i,] - Thet$X[i,])
    #pi.k  = Thet$Pik.w*mapply(dmvnormVec, k = 1:Kw, MoreArgs = list(x = xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w))
    pi.k  = Thet$Pik.w*dmvnormVec(x=xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  
  #updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  #Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol=pn, nrow=pn));inSig0.w = as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.w <- function(Thet, W, Mu0.w, inSig0.w, Sig0.w){
    Ku = length(Thet$Pik.w)
    J = ncol(Thet$X)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol=J, nrow=J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = W - Thet$X 
    #Muk <- matrix(0, ncol=J,nrow=Ku)
    SigK <- list()
    for(k in 1:Ku){
      idk <- which(Thet$Cik.w == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow= nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.w)) ## sum of matrices inverse
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi.tld[idk,])) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
        Mu.k[k,] <- c(SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.w%*%Mu0.w))      ## some of mat
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(apply(t(Yi.tld[idk,]), 2, sum)) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
      }else{
        SigK[[k]] <- Sig0.w
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.w
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL; for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.w[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.w) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- 1.0*as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], sigma = SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.w[-Ku]) %*% SimMU.k)/Thet$Pik.w[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.w <- cmpfun(updateMuk.w)
  
  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w = pn + 2; Psi0.w = 0.5*(nu0.w-pn-1)*cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  updateSigk.w <- function(Thet, W, nu0.w, Psi0.w){
    
    Ku = nrow(Thet$Muk.w)
    m = ncol(W)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Wtl <- W - Thet$X - Thet$Muk.w[Thet$Cik.w,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.w == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.w  + nk
        Yi0 = matrix(c(Wtl[id,]), nrow= nk, byrow=F)
        Psin.w <- as.positive.definite(Psi0.w + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.w))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.w, Psi0.w))
      }
    }
    Res
  }
  updateSigk.w <- cmpfun(updateSigk.w)
  
  #----------------------------------------------
  # Update things related to M 
  #--------------------------------------------  
  
  #--- update Pik.m for epsilon_i
  #alpha.m = 1.0
  updatePik.m <- function(Thet, alpha.m){
    nk <- numeric(nrow(Thet$Muk.m))
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    
    #update the Pis
    alpha.mnew <- alpha.m/nrow(Thet$Muk.m) + nk
    
    c(rdirichlet(n=1, alpha.mnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  
  updateCik.m <- function(i, Thet, M){ # mixture of normals 
    #id <- Thet$Cik.m[i]
    #Sig <- matrix(Thet$Sigk.m[id,], ncol = ncol(M), byrow = F)
    Km = nrow(Thet$Muk.m)
    xi = M[i,] - Thet$Delta*Thet$X[i,]
    #pi.k  = Thet$Pik.m*mapply(dmvnormVec, k = 1:Km, MoreArgs = list(x = xi , Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m))
    pi.k  = Thet$Pik.m*dmvnormVec(x = xi, Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m = rep(0, pn);  Sig0.m = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.m = as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.m <- function(Thet, M, Mu0.m, inSig0.m, Sig0.m){
    #cat(dim(inSig0.m),"(--1)\n")
    Ku = length(Thet$Pik.m)
    J = ncol(M)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol = J, nrow = J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = M - Thet$X%*%diag(Thet$Betadel) 
    Muk <- matrix(0, ncol=J, nrow=Ku)
    for(k in 1:Ku){
      idk <- which(Thet$Cik.m == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow = nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.m)) ## sum of matrices inverse
        #cat(dim(SigK[[k]]),"(--2 \n")
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.m%*%Mu0.m)      ## some of mat
        #cat(dim(SigKWg),"(--3 \n")
      }else{
        SigK[[k]] <- Sig0.m
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.m
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL;  for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.m[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.m) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.m[-Ku]) %*% SimMU.k)/Thet$Pik.m[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.m <- cmpfun(updateMuk.m)
  
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m = pn + 2; Psi0.m = .5*(nu0.m-pn-1)*as.symmetric.matrix(cov(Wn))  #diag(pn)
  
  updateSigk.m <- function(Thet, M, nu0.m, Psi0.m){
    
    Ku = nrow(Thet$Muk.m)
    m = ncol(M)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Mtl <- M - Thet$X%*%diag(Thet$Betadel) - Thet$Muk.m[Thet$Cik.m,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.m == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.m  + nk
        Yi0 <- matrix(c(Mtl[id,]), nrow=nk, byrow=F)
        Psin.m <- as.positive.definite(Psi0.m + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.m))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.m, Psi0.m))
      }
    }
    Res
  }
  updateSigk.m <- cmpfun(updateSigk.m)
  
  #---- Update things related to delta
  quadFormVec <- function(i,X,sigma,Val){
    id = Val[i]
    as.symmetric.matrix(quad.form(as.inverse(as.symmetric.matrix(matrix(sigma[id,], ncol = length(X[i,]),byrow=F))), diag(c(X[i,])))) 
  }
  quadFormVec <- Vectorize(quadFormVec,"i",SIMPLIFY = F) ## Resulting in a list if precision matrices
  quadFormVec <- cmpfun(quadFormVec)
  
  quadFormVec3 <- function(i,Mi,X,sigma,Val){
    id = Val[i]
    c(quad.3form.inv(matrix(sigma[id,], ncol = length(X[i,]),byrow=F), diag(X[i,]),Mi[i,]))
  }
  quadFormVec3 <- Vectorize(quadFormVec3,"i", SIMPLIFY = F)
  quadFormVec3 <- cmpfun(quadFormVec3)
  
  #SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),   
  # prior values
  mu0.del = 1.0; #sig02.del = 100
  updateDelta <- function(Thet, M, mu0.del, sig02.del){
    n = nrow(M)
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X), Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    sig20 <- 1/( 1/sig02.del + sum(Res))
    mu <- sig20*(sum(Res3) + mu0.del/sig02.del)
    rtruncnorm(n=1, a=0, b=10, mean = mu, sd = sqrt(sig20))
    
    #c(sum(Res3), sum(Res),mu, sqrt(sig20), rtruncnorm(n=1, a=0, b=Inf, mean = mu, sd = sqrt(sig20)))
    
  }
  updateDelta <- cmpfun(updateDelta)
  
  
  # prior values
  mu0.del = rep(0,length(Thet$Betadel)); #sig02.del = 20
  Sig0.del = P.mat(length(Thet$Betadel))*.1   #Thet$siggam) # Inverse or precision covariance matrix
  
  updateFunDelta <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDelta <- cmpfun(updateFunDelta)
  
  
  SimDElVec <- function(i, Mi, X, Sigma, Val){
    id <- Val[i]
    Ot = mvtnorm::rmvnorm(n = 1,mean = Mi[i,],sigma = as.symmetric.matrix(matrix(Sigma[id,], ncol = length(X[i,]),byrow=F)))
  }
  SimDElVec <- Vectorize(SimDElVec, "i")
  
  #-- 
  DeltComp <- function(k,Xst, X){
    abs(sum(Xst[,k]*X[,k])/(sum(X[,k]^2)))
    #abs(diag(as.inverse(crossprod(X))%*%crossprod(X,Xst)))
  }
  DeltComp <- Vectorize(DeltComp,"k")
  
  updateFunDeltaSt <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Xst <- base::t(SimDElVec(1:nrow(M), Mitl, Thet$X, Thet$Sigk.m, Thet$Cik.m)) ## K by n matrix and now I get a n by K
    c(DeltComp(1:ncol(Thet$X), Xst, Thet$X))
    #c(DeltComp(Xst, Thet$X))
    
    # Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    # Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    # Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    # Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    # c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDeltaSt  <- cmpfun(updateFunDeltaSt )
  
  
  #----------------------------------------------
  # Update things related to X 
  #--------------------------------------------------  
  #--- update Pik.m for epsilon_i
  #alpha.x <- 1.0
  updatePik.x <- function(Thet, alpha.x){
    nk <- numeric(nrow(Thet$Muk.x))
    for(k in 1:nrow(Thet$Muk.x)){ nk[k] = sum(Thet$Cik.x == k)}
    
    #update the Pis
    alpha.xnew <- alpha.x/nrow(Thet$Muk.x) + nk
    
    c(rdirichlet(n=1, alpha.xnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  updateCik.x <- function(i, Thet){ # mixture of normals 
    #Kx = ncol(Thet$X) # J x J
    #id <- Thet$Cik.x[i]
    #Sig <- matrix(Thet$Sigk.x[id,], ncol = length(Thet$Gamma), byrow = F)
    Xi = Thet$X[i,]
    #pi.k  = Thet$Pik.x*mapply(dmvnormVec, k = 1:Kx, x = Xi,  Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x)
    pi.k  = Thet$Pik.x*dmvnormVec(x=Xi,Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x = numeric(pn); Sig0.x = .5*diag(pn); inSig0.x = 2*diag(pn)#as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x = rep(0, pn);  Sig0.x = .25*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w)+colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.x = as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.x <- function(Thet, Mu0.x, inSig0.x, Sig0.x){
    K = length(Thet$Pik.x)
    J = ncol(Thet$X)
    Res0 <- matrix(0, nrow = K, ncol = J)
    Mu.k <- matrix(0, nrow = K, ncol = J) # K x J
    #SigKWg <- matrix(0, ncol = ncol(Y),nrow=ncol(Y)) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = Thet$X 
    Muk <- matrix(0, ncol = J, nrow = K)
    for(k in 1:K){
      idk <- which(Thet$Cik.x == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 <- matrix(c(Yi.tld[idk,]), nrow=nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[k,], ncol=J, byrow=F)))
        SigK <- as.inverse(as.symmetric.matrix((nk*TpSi + inSig0.x))) ## sum of matrices inverse
        Muk <- SigK%*%(crossprod(TpSi, c(colSums(Yi0))) + inSig0.x%*%Mu0.x)      ## some of mat
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Muk, sigma = SigK))
      }else{
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Mu0.x, sigma = Sig0.x ))
      }
    }
    #--- we impose some constrainsts on the mean
    Mu.k
    
  }
  updateMuk.x <- cmpfun(updateMuk.x)
  
  nu0.x <- pn + 2 ; Psi0.x <- .5*diag(pn) #as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2 ; Psi0.x <- .5*(nu0.x-pn-1)*cov(.5*(Wn+Mn)) #as.symmetric.matrix(.25*cov(Wn))
  updateSigk.x <- function(Thet, nu0.x, Psi0.x){
    
    Ku = nrow(Thet$Muk.x)
    m = ncol(Thet$Muk.x)
    Xtl <- Thet$X - Thet$Muk.x[Thet$Cik.x,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.x == i)
      nk = length(id)
      if(nk > 0){
        nux <- nu0.x  + nk
        Yi0 <- matrix(c(Xtl[id,]), nrow=nk, byrow=F)
        Psin.x <- Psi0.x + crossprod(Yi0)
        Res[i, ] <- c(rinvwishart(nux, Psin.x))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.x, Psi0.x))
      }
    }
    Res
  }
  updateSigk.x <- cmpfun(updateSigk.x)
  # 
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn))) 
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn))) 
  
  A <- min(c(Wn, Mn)) - .1*diff(range(c(Wn, Mn))) 
  B <- max(c(Wn, Mn)) + .1*diff(range(c(Wn, Mn))) 
  
  udapateXi <- function(i, Thet, Y, W, M, Z, A=A, B=B){
    ie =Thet$Cik.e[i]
    iw = Thet$Cik.w[i]
    im = Thet$Cik.m[i]
    ix = Thet$Cik.x[i]
    J = ncol(Thet$X)
    
    Zmod <- Z #cbind(1,Z)
    Ytl = (Y - Zmod%*%Thet$BetaZ - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Wtl = W - Thet$Muk.w[Thet$Cik.w,]
    Mtl = (M - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
    Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
    Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
    
    #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
    Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
    
    Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))) 
    
    #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi <- cmpfun(udapateXi)
  
  
  udapateXi2A <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2A <- cmpfun(udapateXi2A)
  
  ## This approach of simulating Xi only uses W, M, and X
  
  udapateXi2 <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #Mu <- c(Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2 <- cmpfun(udapateXi2)
  
  
  udapateXi2V <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    #Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    #GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- (Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi2V <- Vectorize(udapateXi2V,"i")
  udapateXi2V <- cmpfun(udapateXi2V)
  
  #--- Write Rcpp function to speed things up
  {
    #
    
    
  }
  
  
  
  #Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)
  
  ##----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  #Nsim <- 5000
  #Nsim = 5000 
  MCMCSample <- list()
  
  #-- for X
  MCMCSample$X <- matrix(0,nrow = Nsim,ncol=length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined) 
  MCMCSample$Muk.x <- matrix(0,nrow = Nsim, ncol = Kx*pn)
  MCMCSample$Sigk.x <- matrix(0,nrow = Nsim,ncol=Kx*pn*pn)
  MCMCSample$Pik.x <- matrix(0,nrow = Nsim,ncol=Kx)
  MCMCSample$Cik.x <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for W
  MCMCSample$Muk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn)
  MCMCSample$Sigk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn*pn)
  MCMCSample$Pik.w <- matrix(0,nrow = Nsim,ncol=Kw)
  MCMCSample$Cik.w <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for M
  MCMCSample$Muk.m <- matrix(0,nrow = Nsim,ncol=Km*pn)
  MCMCSample$Sigk.m <- matrix(0,nrow = Nsim,ncol=Km*pn*pn)
  MCMCSample$Pik.m <- matrix(0,nrow = Nsim,ncol=Km)
  MCMCSample$Cik.m <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow=Nsim, ncol=length(Thet$Betadel))
  
  # For Y
  MCMCSample$gamk <- matrix(0,nrow = Nsim, ncol=Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Pik.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Cik.e <- matrix(0,nrow = Nsim, ncol=nrow(Wn))
  
  MCMCSample$Gamma <- matrix(0,nrow = Nsim, ncol=pn) #length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)
  MCMCSample$AcceptGam <- rep(NA, Nsim)
  
  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  # 
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  # 
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  # 
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  #Accept <- 
  # 
  MCMCSample$AcceptX <- matrix(0,nrow = Nsim, ncol=Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  #stop()
  #------------------------------------------------------------------
  #Nsim = 5000 
  #Nsim = 1000
  #Nsim = 3000
  #library(profvis)
  #profvis(
  #system.time(
  #cat("\n #--- MCMC is starting ----> \n")
  #Thet$X <- 
  Thet$X <- (as.matrix(FUI(model="gaussian",smooth=TRUE, W,silent = FALSE)) %*%bs2)/length(a)
  for(iter in 1:Nsim){
    if(iter %% 500 == 0){cat("iter=",iter,"\n")}
    #----------------------------------------------
    # Update things related to X 
    #--------------------------------------------  
    # #--- update Pik.m for epsilon_i
    # Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    # 
    # #--- update Cik.m for epsilon_i based on mixture of normals
    # Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    # #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x)) 
    # 
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    # 
    # Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    # Thet$InvSigk.x <- InvCov(Thet$Sigk.x)
    # 
    #Ytd = Y - Zmod%*%Thet$BetaZ
    #if(iter%%10 == 0){
    #Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B) # Xn #
    #Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    #Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    #Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w, 
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)
    
    #Thet$X <- Xn#UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,]) 
    
    #}
    #dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))
    
    #Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    #Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    #Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    #Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w, 
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J))) 
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    # #--- Update Pik.w  
    # #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Kw)
    # for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    # #update the Pis
    # alpha.wnew <- alpha.W/Kw + nk
    # Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    # 
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    # Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    # Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    # Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w  
    # #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Km)
    # for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    # #update the Pis
    # alpha.mnew <- alpha.m/Kw + nk
    # Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    # 
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    # Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m)) 
    # #Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))  
    # #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    # Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    # Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # #if(iter%%50 == 0){
    # Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    # MCMCSample$Betdel[iter,] <- Thet$Betadel
    # #}
    #---------------------------------------------
    #cat("#--- update things related to Y or epsilon ")
    #--------------------------------------------- 
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for(k in 1:length(Thet$sig2k.e)){ nk[k] = sum(Thet$Cik.e == k)}
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n=1, alpha.enew))
    
    #update the Cik.e
    ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
    #Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) #rep(1,n) #
    #Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    #update nui  and si
    #Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    #nuisi <- updatenuisi(Thet, Y, Zmod)
    #Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    #update Sigk.e
    #Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    #update sigk and gamk
    #--- update jointly sig2ke and gamke
    #Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi)
    Thet$sig2k.e <- Temp[1,];  Thet$gamk <- Temp[2,]; Thet$p <- Temp[3,]; MCMCSample$AcceptX[iter,] <- Temp[4,]; Thet$Ak <- Temp[5,]; Thet$Bk <- Temp[6,]; Thet$Ck <- Temp[7,];
    
    #update Gamma
    #Thet$siggam <- sig02.del
    #Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0)
    # Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    # Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    #Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    
    Thet$BetaZ <- updateGamBeta(Thet, Y, Zmod, InvSig0.bet0, Mu0.bet0)#Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Tep <- updateGamma(Thet, Y, Zmod, K0,SigmaProp, c0, g0)
    #print(Tep[length(Tep)])
    GamAccep <- Tep[1]
    Thet$Gamma <- Tep[-1]  #updateGamma(Thet, Y, Zmod, K0,SigmaProp, c0)
    #updateGamma(Thet, Y, Zmod, InvSig0.gam0, Mu0.gam0)  # Tp0[-c(1:ncol(Zmod))]
    
    Thet$siggam <-  updateSigGam(Thet, al0, bet0, order = 2)# sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#
    
    #Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    # 
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    # 
    # #--- upodate beta_{z,m}
    # #--  Prior 
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    # 
    # #--- update X_i 
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX) 
    # 
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]); 
    # 
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb 
    
    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x))  ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise
    
    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w
    
    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    #MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e
    MCMCSample$AcceptGam[iter] <- GamAccep 
    
    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  } 
  
  #)
  #)
  
  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample$XFui <- Thet$X
  MCMCSample
}



#---- Simulated tau
#-- Using the rtmvnorm approach for the truncated multivariate normal
#--- Update Beta(t) and beta_z together ()
#-- Fix X
BQBayes.ME_SoFRFuncSimFx <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0 = 0.5,X, g0,tunprop){
  
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
  met <- c("f1","f2","f3","f4","f5")
  
  
  func <- function(Pik, n=1, size=1){
    which.max(rmultinom(n=1,size=1,prob = Pik))
  }
  funcCmp <- cmpfun(func)
  
  F2 <- function(Pik, n=1, size=1){
    
    apply(Pik,1,funcCmp)
  }
  F2Cmp <- cmpfun(F2)
  
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
  
  
  
  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  #pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  #pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  n = length(Y) 
  Zmod = cbind(1,Z)
  pn = c(5, 6, 20, 20)[which(n == c(100, 200,500,1000))]
  alpha.e = alpx
  alpha.w = alpha.W = alpx
  alpha.x = alpha.X = alpx
  alpha.m = alpx
  pn = pn #min(15, ceiling(length(Y)^{1/3.8})+4)
  
  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt #c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) # 
  Mn_Mod <- M%*%diag(1/Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20,"knots"), 1) ))
  Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a) 
  Wn <- (W%*%bs2)/length(a)
  Xn <- (X%*%bs2)/length(a)
  
  cat("\n #--- Initial values...\n")  
  P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D #+ diag(rep(1e-3, K))
    return(P)
  }
  
  
  InitialValues <- function(Y, W, M, Z, df0=3, pn, Gx, a){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- Z #cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    lmFit <- lm(Y ~ -1 + Zmod + W_i) 
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- residuals.lm(lmFit)
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  InitialValuesQR <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z+Wn)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    #plot(a, bs2%*%lmFit$coefficients[-c(1:3)], type="l")
    
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    Tp0 = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd)), upper=c(30, 0.95*max(bnd) ))
    Res$Hess <- Tp0$hessian
    
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  #Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, W, M, Z, df0=3, pn, Gx=1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  #Kvalues
  #Kc <- 5
  
  Kx = Thet0$Kx               ## number of X  cluster
  Ke = Thet0$Ke               ## number of ei cluster
  Km = Thet0$Km               ## number of ei cluster
  Kw = Thet0$Kw              ## number of omega_i cluster 
  
  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta   #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n=pn) #as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ #rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del#0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  #Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1]); Thet$gamU <- max(bnd[2])
  
  #bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) #c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
  
  Thet$Cik.x =  sample.int(Kx,size=n, replace = T) #Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e =  sample.int(Ke,size=n, replace = T) #Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w = sample.int(Kw,size=n, replace = T) #Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m = sample.int(Km,size=n, replace = T) #Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km
  
  
  Thet$Pik.x = rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)#Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e = rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)#Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w = rdirch(n=1,alpha = rep(alpha.e, Kw)/Kw) #Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m = rdirch(n=1,alpha = rep(alpha.m, Km)/Km)#Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw
  
  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw
  
  Thet$sig2k.e  = nimble::rinvgamma(n=Ke, shape=5/2,scale=8/2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n=Ke, min = .95*Thet$gamL, max=.95*Thet$gamU)
  Thet$p <- 1*(Thet$gamk < 0) + (Thet$tau0 - 1*(Thet$gamk < 0))/(2*pnorm(-abs(Thet$gamk))*exp(0.5*Thet$gamk*Thet$gamk))
  Thet$Ak <- (1-2*Thet$p)/(Thet$p*(1-Thet$p)); Thet$Bk <- 2/(Thet$p*(1-Thet$p)); Thet$Ck <- 1/( 1*(Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean=0, sd=sqrt(Thet$sig2k.e[1])) ; Thet$nui <- rexp(n=n, rate = 1/Thet$sig2k.e[1]) 
  #Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix
  
  Thet$Sigk.w <- repmat(c(.5*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw*pn), nrow = Kw)          #- vectors of length Kw and Kw respectively
  
  Thet$Sigk.x <- repmat(c(.5*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$Muk.x  <- matrix(rnorm(n=Kx*pn), nrow=Kx)   #- vectors of length Kx and Kx respectively
  
  Thet$Sigk.m <- repmat(c(.5*diag(pn)), Km, 1)     #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2*diag(pn)), Km, 1)
  Thet$Muk.m  <- matrix(rnorm(n=Km*pn), nrow = Km)   #- vectors of length Kx and Kx respectively
  
  A <- min(c(c(Wn), c(Mn))) - .5*diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5*diff(range(c(Wn, Mn)))
  #Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  #matrix(rnorm(n = n*pn), ncol=pn)
  #Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5*(Mn+Wn) #Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))
  
  
  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
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
  
  
  #method = "L-BFGS-B"
  #method = "BFGS"
  
  #Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  #Res
  
  #Res$hessian
  #Res$hessian/n
  #solve(Res$hessian/n)
  
  #LogPlostGalLik(c(.05,), c(ei), Thet)
  
  #-------- Update Beta (vector of length pn) 
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #   
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma 
  #   
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #   
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #   
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #   
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)
  
  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together) 
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 = ncol(Z) +1
  Pgam <- P.mat(pn)*tauval
  Sig0.gam0 = diag(pn0); Mu0.gam0 = numeric(pn0+pn);
  #InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))
  
  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.gam0%*%Mu0.gam0))
    
    #c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb =c(rep(-Inf, ncol(Zmod)),rep(-g0,length(Mu.gam)-ncol(Zmod))),
                                ub = c(rep(Inf, ncol(Zmod)),rep(g0,length(Mu.gam)-ncol(Zmod)))   ))
    
  }
  updateGam <- cmpfun(updateGam)
  
  #siggam ~ IG(al0, bet0)
  al0 = 0.001
  bet0 = .005
  al0 = 2.5978252 #1
  bet0 = 0.4842963 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    val <- nimble::rinvgamma(n = 1, shape = al, scale = bet)
    ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  
  al0 = 0.001 #1 #1
  bet0 = 0.001 #0.005 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    #rgamma(n = 1, shape = al, scale = bet)
    nimble::rinvgamma(n = 1, shape = al, scale = bet)
    #ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e 
  #alpha.e = 1.0
  updatePik.e <- function(Thet, alpha.e){
    
    nk <- numeric(length(Thet$Cik.e))
    for(i in 1:length(Thet$Cik.e)){ nk[i] = sum(Thet$Cik.e == i)}
    
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$Cik.e) + nk
    
    rdirichlet(n=1, alpha.enew)
    
  }
  
  #-------- Update Cik.e  
  
  #ei <- Y - Z%*%Thet$Betaz - Thet$X%*%Thet$Gamma
  updateCik.e <- function(i, Thet,ei){
    #Y - Thet$C*abs(Thet$gam)*Thet$si - Thet$A*Thet$nui
    
    #  Yi.tld <- Y[i] - c(crossprod(Thet$Gamma, Thet$X[i,]))
    #pi.k  <- Thet$Pik.e*mapply(dnorm, x = Yi.tld, mean=Thet$muk.e, sd = sqrt(Thet$sig2k.e)) + .Machine$double.eps
    pik <- Thet$Pik.e*mapply(fGal.p0, e = ei[i], sig2 = sqrt(Thet$sig2k.e), gam = Thet$gamk, p = Thet$p, tau0 = Thet$tau0) #+ .Machine$double.eps
    which(rmultinom(n=1, size=1, prob = pik) == 1)
  }
  updateCik.e <- Vectorize(updateCik.e,"i") 
  
  #--- update Si
  #-- calss id for individual i (Matrix with columns of nui and si)
  updatenui <- function(Thet, Y, Z){
    
    ki = Thet$Cik.e
    bi <- ( ((Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ck[ki]*abs(Thet$gamk[ki])*Thet$si)^2)/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])))
    ai = ((Thet$Ak[ki]*Thet$Ak[ki])/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])) + 2/sqrt(Thet$sig2k.e[ki]))
    
    #mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) 
    #val = mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    #ifelse(val < .001, rexp(n = length(muui), rate = 1/sqrt(Thet$sig2k.e[Thet$Cik.e] )), val)
  }
  
  updatesi <- function(Thet, Y, Z){   
    #-- update si
    ki = Thet$Cik.e
    sig2si <- 1 /( 1/Thet$sig2k.e[ki] +  ((Thet$Ck[ki]*Thet$gamk[ki])^2)/ (Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui))
    muui <- sig2si*(Thet$Ck[ki]*abs(Thet$gamk[ki])*(Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ak[ki]*Thet$nui))/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui)
    
    #mapply(truncnorm::rtruncnorm, n=1, a = 0.01, mean = muui, sd = sqrt(sig2si))
    mapply(truncnorm::rtruncnorm, n=1, a = 0.001, mean = muui, sd = sqrt(sig2si))
    
    
  }
  
  
  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  #a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig=5/2; b.sig = 8/2; sdsig.prop = 0.2; sdgam.prop = .3  #2.153640 1.167331
  a.sig=2.153640; b.sig = 1.167331; sdsig.prop = 0.2; sdgam.prop = .3 # 7.133445 4.303263
  a.sig=7.133445; b.sig = 4.303263; sdsig.prop = 0.2; sdgam.prop = .3 #
  updatesig_gam <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop){
    
    a0 = 0.001
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      sig_newk <- truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gam <- Vectorize(updatesig_gam,"k")
  
  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10,10,10)*1
  #varxi <- c(10,10,10)*.25
  #varxi <- c(10,10,10)*.15 ## high acceptance rate
  varxi <- c(10,10,10)*.35 #-- high acceptance rate
  varxi <- c(10,10,10)*tunprop ##  tunprop
  #varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi){
    
    a0 = 0.1
    ab = 5 ## This works well
    ab = 10
    #varxi = 5
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]),Thet$gamk[k]),sigma = Sdsig.prop*varxi[k], lb = c(a0,0.95*Thet$gamL), ub = c(ab, .95*Thet$gamU)  )
      
      
      sig_newk <- Val[1] #truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] #truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu=c(sig_newk, gam_newk), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE) )
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes,"k")
  
  
  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices
  InvCov <- function(Mat){
    K <- nrow(Mat)
    J <- round(sqrt(ncol(Mat)))
    Res0 = matrix(0, nrow = K, ncol=ncol(Mat))
    for(k in 1:K){ Res0[k,] <- c(as.inverse(as.symmetric.matrix(matrix(Mat[k,],ncol=J, byrow=F)))) }
    Res0
  }
  
  
  #----------------------------------------------
  # Update things related to Wi 
  #--------------------------------------------
  #Piw
  #Cik.w
  #--- Update Pik.x  
  #alpha.W = alpha.w = 1.0
  updatePik.w <- function(Thet, alpha.W){
    Ku = nrow(Thet$Muk.w)
    nk <- numeric(Ku)
    for(k in 1:Ku){ nk[k] = sum(Thet$Cik.w == k)}
    
    #update the Pis
    alpha.wnew <- alpha.W/length(Thet$muk.w) + nk
    
    rdirichlet(n=1, alpha.wnew)
    
  }
  
  #--- Update Ci.k mixture of normals 
  dmvnormVec <- function(x, Mean, SigmaVc){
    Ku <- nrow(Mean)
    Val <- numeric(Ku)
    for(k in 1:Ku){ Val[k] <- mvtnorm::dmvnorm(x = x, mean = c(Mean[k,]), sigma = as.symmetric.matrix(matrix(SigmaVc[k,],ncol=length(x),byrow=F))) }
    Val
  }
  #dmvnormVec <- Vectorize(dmvnormVec,"k")
  #dmvnormVec(x = rep(0,6), Mn=Thet$Muk.w,SigmaVc = Thet$Sigk.w)
  
  updateCik.w <- function(i, Thet, W){ # mixture of normals 
    #Sig <- matrix(Thet$Sigk.w[id,], ncol = ncol(W), byrow = F)
    #kw = nrow(Thet$Muk.w)
    xi =  c(W[i,] - Thet$X[i,])
    #pi.k  = Thet$Pik.w*mapply(dmvnormVec, k = 1:Kw, MoreArgs = list(x = xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w))
    pi.k  = Thet$Pik.w*dmvnormVec(x=xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  
  #updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  #Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol=pn, nrow=pn));inSig0.w = as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.w <- function(Thet, W, Mu0.w, inSig0.w, Sig0.w){
    Ku = length(Thet$Pik.w)
    J = ncol(Thet$X)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol=J, nrow=J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = W - Thet$X 
    #Muk <- matrix(0, ncol=J,nrow=Ku)
    SigK <- list()
    for(k in 1:Ku){
      idk <- which(Thet$Cik.w == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow= nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.w)) ## sum of matrices inverse
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi.tld[idk,])) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
        Mu.k[k,] <- c(SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.w%*%Mu0.w))      ## some of mat
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(apply(t(Yi.tld[idk,]), 2, sum)) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
      }else{
        SigK[[k]] <- Sig0.w
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.w
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL; for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.w[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.w) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- 1.0*as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], sigma = SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.w[-Ku]) %*% SimMU.k)/Thet$Pik.w[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.w <- cmpfun(updateMuk.w)
  
  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w = pn + 2; Psi0.w = 0.5*(nu0.w-pn-1)*cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  updateSigk.w <- function(Thet, W, nu0.w, Psi0.w){
    
    Ku = nrow(Thet$Muk.w)
    m = ncol(W)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Wtl <- W - Thet$X - Thet$Muk.w[Thet$Cik.w,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.w == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.w  + nk
        Yi0 = matrix(c(Wtl[id,]), nrow= nk, byrow=F)
        Psin.w <- as.positive.definite(Psi0.w + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.w))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.w, Psi0.w))
      }
    }
    Res
  }
  updateSigk.w <- cmpfun(updateSigk.w)
  
  #----------------------------------------------
  # Update things related to M 
  #--------------------------------------------  
  
  #--- update Pik.m for epsilon_i
  #alpha.m = 1.0
  updatePik.m <- function(Thet, alpha.m){
    nk <- numeric(nrow(Thet$Muk.m))
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    
    #update the Pis
    alpha.mnew <- alpha.m/nrow(Thet$Muk.m) + nk
    
    c(rdirichlet(n=1, alpha.mnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  
  updateCik.m <- function(i, Thet, M){ # mixture of normals 
    #id <- Thet$Cik.m[i]
    #Sig <- matrix(Thet$Sigk.m[id,], ncol = ncol(M), byrow = F)
    Km = nrow(Thet$Muk.m)
    xi = M[i,] - Thet$Delta*Thet$X[i,]
    #pi.k  = Thet$Pik.m*mapply(dmvnormVec, k = 1:Km, MoreArgs = list(x = xi , Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m))
    pi.k  = Thet$Pik.m*dmvnormVec(x = xi, Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m = rep(0, pn);  Sig0.m = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.m = as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.m <- function(Thet, M, Mu0.m, inSig0.m, Sig0.m){
    #cat(dim(inSig0.m),"(--1)\n")
    Ku = length(Thet$Pik.m)
    J = ncol(M)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol = J, nrow = J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = M - Thet$X%*%diag(Thet$Betadel) 
    Muk <- matrix(0, ncol=J, nrow=Ku)
    for(k in 1:Ku){
      idk <- which(Thet$Cik.m == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow = nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.m)) ## sum of matrices inverse
        #cat(dim(SigK[[k]]),"(--2 \n")
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.m%*%Mu0.m)      ## some of mat
        #cat(dim(SigKWg),"(--3 \n")
      }else{
        SigK[[k]] <- Sig0.m
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.m
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL;  for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.m[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.m) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.m[-Ku]) %*% SimMU.k)/Thet$Pik.m[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.m <- cmpfun(updateMuk.m)
  
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m = pn + 2; Psi0.m = .5*(nu0.m-pn-1)*as.symmetric.matrix(cov(Wn))  #diag(pn)
  
  updateSigk.m <- function(Thet, M, nu0.m, Psi0.m){
    
    Ku = nrow(Thet$Muk.m)
    m = ncol(M)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Mtl <- M - Thet$X%*%diag(Thet$Betadel) - Thet$Muk.m[Thet$Cik.m,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.m == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.m  + nk
        Yi0 <- matrix(c(Mtl[id,]), nrow=nk, byrow=F)
        Psin.m <- as.positive.definite(Psi0.m + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.m))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.m, Psi0.m))
      }
    }
    Res
  }
  updateSigk.m <- cmpfun(updateSigk.m)
  
  #---- Update things related to delta
  quadFormVec <- function(i,X,sigma,Val){
    id = Val[i]
    as.symmetric.matrix(quad.form(as.inverse(as.symmetric.matrix(matrix(sigma[id,], ncol = length(X[i,]),byrow=F))), diag(c(X[i,])))) 
  }
  quadFormVec <- Vectorize(quadFormVec,"i",SIMPLIFY = F) ## Resulting in a list if precision matrices
  quadFormVec <- cmpfun(quadFormVec)
  
  quadFormVec3 <- function(i,Mi,X,sigma,Val){
    id = Val[i]
    c(quad.3form.inv(matrix(sigma[id,], ncol = length(X[i,]),byrow=F), diag(X[i,]),Mi[i,]))
  }
  quadFormVec3 <- Vectorize(quadFormVec3,"i", SIMPLIFY = F)
  quadFormVec3 <- cmpfun(quadFormVec3)
  
  #SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),   
  # prior values
  mu0.del = 1.0; #sig02.del = 100
  updateDelta <- function(Thet, M, mu0.del, sig02.del){
    n = nrow(M)
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X), Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    sig20 <- 1/( 1/sig02.del + sum(Res))
    mu <- sig20*(sum(Res3) + mu0.del/sig02.del)
    rtruncnorm(n=1, a=0, b=10, mean = mu, sd = sqrt(sig20))
    
    #c(sum(Res3), sum(Res),mu, sqrt(sig20), rtruncnorm(n=1, a=0, b=Inf, mean = mu, sd = sqrt(sig20)))
    
  }
  updateDelta <- cmpfun(updateDelta)
  
  
  # prior values
  mu0.del = rep(0,length(Thet$Betadel)); #sig02.del = 20
  Sig0.del = P.mat(length(Thet$Betadel))*.1   #Thet$siggam) # Inverse or precision covariance matrix
  
  updateFunDelta <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDelta <- cmpfun(updateFunDelta)
  
  
  SimDElVec <- function(i, Mi, X, Sigma, Val){
    id <- Val[i]
    Ot = mvtnorm::rmvnorm(n = 1,mean = Mi[i,],sigma = as.symmetric.matrix(matrix(Sigma[id,], ncol = length(X[i,]),byrow=F)))
  }
  SimDElVec <- Vectorize(SimDElVec, "i")
  
  #-- 
  DeltComp <- function(k,Xst, X){
    abs(sum(Xst[,k]*X[,k])/(sum(X[,k]^2)))
    #abs(diag(as.inverse(crossprod(X))%*%crossprod(X,Xst)))
  }
  DeltComp <- Vectorize(DeltComp,"k")
  
  updateFunDeltaSt <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Xst <- base::t(SimDElVec(1:nrow(M), Mitl, Thet$X, Thet$Sigk.m, Thet$Cik.m)) ## K by n matrix and now I get a n by K
    c(DeltComp(1:ncol(Thet$X), Xst, Thet$X))
    #c(DeltComp(Xst, Thet$X))
    
    # Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    # Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    # Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    # Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    # c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDeltaSt  <- cmpfun(updateFunDeltaSt )
  
  
  #----------------------------------------------
  # Update things related to X 
  #--------------------------------------------------  
  #--- update Pik.m for epsilon_i
  #alpha.x <- 1.0
  updatePik.x <- function(Thet, alpha.x){
    nk <- numeric(nrow(Thet$Muk.x))
    for(k in 1:nrow(Thet$Muk.x)){ nk[k] = sum(Thet$Cik.x == k)}
    
    #update the Pis
    alpha.xnew <- alpha.x/nrow(Thet$Muk.x) + nk
    
    c(rdirichlet(n=1, alpha.xnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  updateCik.x <- function(i, Thet){ # mixture of normals 
    #Kx = ncol(Thet$X) # J x J
    #id <- Thet$Cik.x[i]
    #Sig <- matrix(Thet$Sigk.x[id,], ncol = length(Thet$Gamma), byrow = F)
    Xi = Thet$X[i,]
    #pi.k  = Thet$Pik.x*mapply(dmvnormVec, k = 1:Kx, x = Xi,  Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x)
    pi.k  = Thet$Pik.x*dmvnormVec(x=Xi,Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x = numeric(pn); Sig0.x = .5*diag(pn); inSig0.x = 2*diag(pn)#as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x = rep(0, pn);  Sig0.x = .25*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w)+colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.x = as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.x <- function(Thet, Mu0.x, inSig0.x, Sig0.x){
    K = length(Thet$Pik.x)
    J = ncol(Thet$X)
    Res0 <- matrix(0, nrow = K, ncol = J)
    Mu.k <- matrix(0, nrow = K, ncol = J) # K x J
    #SigKWg <- matrix(0, ncol = ncol(Y),nrow=ncol(Y)) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = Thet$X 
    Muk <- matrix(0, ncol = J, nrow = K)
    for(k in 1:K){
      idk <- which(Thet$Cik.x == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 <- matrix(c(Yi.tld[idk,]), nrow=nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[k,], ncol=J, byrow=F)))
        SigK <- as.inverse(as.symmetric.matrix((nk*TpSi + inSig0.x))) ## sum of matrices inverse
        Muk <- SigK%*%(crossprod(TpSi, c(colSums(Yi0))) + inSig0.x%*%Mu0.x)      ## some of mat
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Muk, sigma = SigK))
      }else{
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Mu0.x, sigma = Sig0.x ))
      }
    }
    #--- we impose some constrainsts on the mean
    Mu.k
    
  }
  updateMuk.x <- cmpfun(updateMuk.x)
  
  nu0.x <- pn + 2 ; Psi0.x <- .5*diag(pn) #as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2 ; Psi0.x <- .5*(nu0.x-pn-1)*cov(.5*(Wn+Mn)) #as.symmetric.matrix(.25*cov(Wn))
  updateSigk.x <- function(Thet, nu0.x, Psi0.x){
    
    Ku = nrow(Thet$Muk.x)
    m = ncol(Thet$Muk.x)
    Xtl <- Thet$X - Thet$Muk.x[Thet$Cik.x,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.x == i)
      nk = length(id)
      if(nk > 0){
        nux <- nu0.x  + nk
        Yi0 <- matrix(c(Xtl[id,]), nrow=nk, byrow=F)
        Psin.x <- Psi0.x + crossprod(Yi0)
        Res[i, ] <- c(rinvwishart(nux, Psin.x))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.x, Psi0.x))
      }
    }
    Res
  }
  updateSigk.x <- cmpfun(updateSigk.x)
  # 
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn))) 
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn))) 
  
  A <- min(c(Wn, Mn)) - .1*diff(range(c(Wn, Mn))) 
  B <- max(c(Wn, Mn)) + .1*diff(range(c(Wn, Mn))) 
  
  udapateXi <- function(i, Thet, Y, W, M, Z, A=A, B=B){
    ie =Thet$Cik.e[i]
    iw = Thet$Cik.w[i]
    im = Thet$Cik.m[i]
    ix = Thet$Cik.x[i]
    J = ncol(Thet$X)
    
    Zmod <- Z #cbind(1,Z)
    Ytl = (Y - Zmod%*%Thet$BetaZ - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Wtl = W - Thet$Muk.w[Thet$Cik.w,]
    Mtl = (M - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
    Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
    Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
    
    #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
    Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
    
    Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))) 
    
    #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi <- cmpfun(udapateXi)
  
  
  udapateXi2A <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2A <- cmpfun(udapateXi2A)
  
  ## This approach of simulating Xi only uses W, M, and X
  
  udapateXi2 <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #Mu <- c(Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2 <- cmpfun(udapateXi2)
  
  
  udapateXi2V <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    #Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    #GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- (Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi2V <- Vectorize(udapateXi2V,"i")
  udapateXi2V <- cmpfun(udapateXi2V)
  
  #--- Write Rcpp function to speed things up
  {
    #
    
    
  }
  
  
  
  #Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)
  
  ##----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  #Nsim <- 5000
  #Nsim = 5000 
  MCMCSample <- list()
  
  #-- for X
  MCMCSample$X <- matrix(0,nrow = Nsim,ncol=length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined) 
  MCMCSample$Muk.x <- matrix(0,nrow = Nsim, ncol = Kx*pn)
  MCMCSample$Sigk.x <- matrix(0,nrow = Nsim,ncol=Kx*pn*pn)
  MCMCSample$Pik.x <- matrix(0,nrow = Nsim,ncol=Kx)
  MCMCSample$Cik.x <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for W
  MCMCSample$Muk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn)
  MCMCSample$Sigk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn*pn)
  MCMCSample$Pik.w <- matrix(0,nrow = Nsim,ncol=Kw)
  MCMCSample$Cik.w <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for M
  MCMCSample$Muk.m <- matrix(0,nrow = Nsim,ncol=Km*pn)
  MCMCSample$Sigk.m <- matrix(0,nrow = Nsim,ncol=Km*pn*pn)
  MCMCSample$Pik.m <- matrix(0,nrow = Nsim,ncol=Km)
  MCMCSample$Cik.m <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow=Nsim, ncol=length(Thet$Betadel))
  
  # For Y
  MCMCSample$gamk <- matrix(0,nrow = Nsim, ncol=Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Pik.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Cik.e <- matrix(0,nrow = Nsim, ncol=nrow(Wn))
  
  MCMCSample$Gamma <- matrix(0,nrow = Nsim, ncol=length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)
  
  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  # 
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  # 
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  # 
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  #Accept <- 
  # 
  MCMCSample$AcceptX <- matrix(0,nrow = Nsim, ncol=Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  #stop()
  #------------------------------------------------------------------
  #Nsim = 5000 
  #Nsim = 1000
  #Nsim = 3000
  #library(profvis)
  #profvis(
  #system.time(
  #cat("\n #--- MCMC is starting ----> \n")
  Thet$X <- Xn
  for(iter in 1:Nsim){
    if(iter %% 500 == 0){cat("iter=",iter,"\n")}
    #----------------------------------------------
    # Update things related to X 
    #--------------------------------------------  
    #--- update Pik.m for epsilon_i
    Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    
    #--- update Cik.m for epsilon_i based on mixture of normals
    Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x)) 
    
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    
    Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    Thet$InvSigk.x <- InvCov(Thet$Sigk.x)
    
    #Ytd = Y - Zmod%*%Thet$BetaZ
    #if(iter%%10 == 0){
    #Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B) # Xn #
    #Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    #Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    #Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w, 
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)
    
    #Thet$X <- Xn #UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,]) 
    
    #}
    #dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))
    
    #Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    #Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    #Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    #Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w, 
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J))) 
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Kw)
    for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    #update the Pis
    alpha.wnew <- alpha.W/Kw + nk
    Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    
    #--- Update Ci.k mixture of normals
    #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Km)
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    #update the Pis
    alpha.mnew <- alpha.m/Kw + nk
    Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    
    #--- Update Ci.k mixture of normals
    #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    #Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))  
    #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    #if(iter%%50 == 0){
    Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    MCMCSample$Betdel[iter,] <- Thet$Betadel
    #}
    #---------------------------------------------
    #cat("#--- update things related to Y or epsilon ")
    #--------------------------------------------- 
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for(k in 1:length(Thet$sig2k.e)){ nk[k] = sum(Thet$Cik.e == k)}
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n=1, alpha.enew))
    
    #update the Cik.e
    ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
    #Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) #rep(1,n) #
    #Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    #update nui  and si
    #Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    #nuisi <- updatenuisi(Thet, Y, Zmod)
    #Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    #update Sigk.e
    #Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    #update sigk and gamk
    #--- update jointly sig2ke and gamke
    #Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi)
    Thet$sig2k.e <- Temp[1,];  Thet$gamk <- Temp[2,]; Thet$p <- Temp[3,]; MCMCSample$AcceptX[iter,] <- Temp[4,]; Thet$Ak <- Temp[5,]; Thet$Bk <- Temp[6,]; Thet$Ck <- Temp[7,];
    
    #update Gamma
    #Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    #Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <-  updateSigGam(Thet, al0, bet0, order = 2)# sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#
    
    #Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    # 
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    # 
    # #--- upodate beta_{z,m}
    # #--  Prior 
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    # 
    # #--- update X_i 
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX) 
    # 
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]); 
    # 
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb 
    
    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x))  ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise
    
    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w
    
    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    #MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e
    
    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  } 
  
  #)
  #)
  
  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample
}


#--  using W (average over days) and M
BQBayes.ME_SoFRFuncSim <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0 = 0.5, X, g0,tunprop){
  
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
  met <- c("f1","f2","f3","f4","f5")
  
  
  func <- function(Pik, n=1, size=1){
    which.max(rmultinom(n=1,size=1,prob = Pik))
  }
  funcCmp <- cmpfun(func)
  
  F2 <- function(Pik, n=1, size=1){
    
    apply(Pik,1,funcCmp)
  }
  F2Cmp <- cmpfun(F2)
  
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
  
  
  
  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  #pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  #pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  n = length(Y) 
  Zmod = cbind(1,Z)
  pn = c(5, 6, 20, 20)[which(n == c(100, 200,500,1000))]
  alpha.e = alpx
  alpha.w = alpha.W = alpx
  alpha.x = alpha.X = alpx
  alpha.m = alpx
  pn = pn #min(15, ceiling(length(Y)^{1/3.8})+4)
  
  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt #c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) # 
  Mn_Mod <- M%*%diag(1/Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20,"knots"), 1) ))
  Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a) 
  Wn <- (W%*%bs2)/length(a)
  Xn <- (X%*%bs2)/length(a)
  
  cat("\n #--- Initial values...\n")  
  P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D #+ diag(rep(1e-6, K))
    return(P)
  }
  
  
  InitialValues <- function(Y, W, M, Z, df0=3, pn, Gx, a){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- Z #cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    lmFit <- lm(Y ~ -1 + Zmod + W_i) 
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- residuals.lm(lmFit)
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  InitialValuesQR <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z+Wn)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    #plot(a, bs2%*%lmFit$coefficients[-c(1:3)], type="l")
    
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    Tp0 = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd)), upper=c(30, 0.95*max(bnd) ))
    Res$Hess <- Tp0$hessian
    
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  #Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, W, M, Z, df0=3, pn, Gx=1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  #Kvalues
  #Kc <- 5
  
  Kx = Thet0$Kx               ## number of X  cluster
  Ke = Thet0$Ke               ## number of ei cluster
  Km = Thet0$Km               ## number of ei cluster
  Kw = Thet0$Kw              ## number of omega_i cluster 
  
  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta   #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n=pn) #as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ #rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del#0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  #Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1]); Thet$gamU <- max(bnd[2])
  
  #bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) #c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
  
  Thet$Cik.x =  sample.int(Kx,size=n, replace = T) #Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e =  sample.int(Ke,size=n, replace = T) #Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w = sample.int(Kw,size=n, replace = T) #Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m = sample.int(Km,size=n, replace = T) #Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km
  
  
  Thet$Pik.x = rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)#Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e = rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)#Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w = rdirch(n=1,alpha = rep(alpha.e, Kw)/Kw) #Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m = rdirch(n=1,alpha = rep(alpha.m, Km)/Km)#Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw
  
  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw
  
  Thet$sig2k.e  = nimble::rinvgamma(n=Ke, shape=5/2,scale=8/2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n=Ke, min = .95*Thet$gamL, max=.95*Thet$gamU)
  Thet$p <- 1*(Thet$gamk < 0) + (Thet$tau0 - 1*(Thet$gamk < 0))/(2*pnorm(-abs(Thet$gamk))*exp(0.5*Thet$gamk*Thet$gamk))
  Thet$Ak <- (1-2*Thet$p)/(Thet$p*(1-Thet$p)); Thet$Bk <- 2/(Thet$p*(1-Thet$p)); Thet$Ck <- 1/( 1*(Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean=0, sd=sqrt(Thet$sig2k.e[1])) ; Thet$nui <- rexp(n=n, rate = 1/Thet$sig2k.e[1]) 
  #Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix
  
  Thet$Sigk.w <- repmat(c(.5*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw*pn), nrow = Kw)          #- vectors of length Kw and Kw respectively
  
  Thet$Sigk.x <- repmat(c(.5*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$Muk.x  <- matrix(rnorm(n=Kx*pn), nrow=Kx)   #- vectors of length Kx and Kx respectively
  
  Thet$Sigk.m <- repmat(c(.5*diag(pn)), Km, 1)     #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2*diag(pn)), Km, 1)
  Thet$Muk.m  <- matrix(rnorm(n=Km*pn), nrow = Km)   #- vectors of length Kx and Kx respectively
  
  A <- min(c(c(Wn), c(Mn))) - .5*diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5*diff(range(c(Wn, Mn)))
  #Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  #matrix(rnorm(n = n*pn), ncol=pn)
  #Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5*(Mn+Wn) #Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))
  
  
  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
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
  
  
  #method = "L-BFGS-B"
  #method = "BFGS"
  
  #Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  #Res
  
  #Res$hessian
  #Res$hessian/n
  #solve(Res$hessian/n)
  
  #LogPlostGalLik(c(.05,), c(ei), Thet)
  
  #-------- Update Beta (vector of length pn) 
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #   
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma 
  #   
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #   
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #   
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #   
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)
  
  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together) 
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 = ncol(Z) +1
  Pgam <- P.mat(pn)*tauval
  Sig0.gam0 = diag(pn0); Mu0.gam0 = numeric(pn0+pn);
  #InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))
  
  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.gam0%*%Mu0.gam0))
    
    #c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    expr1 <- quote(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb =c(rep(-Inf, ncol(Zmod)),rep(-g0,length(Mu.gam)-ncol(Zmod))),
                                                                                               ub = c(rep(Inf, ncol(Zmod)),rep(g0,length(Mu.gam)-ncol(Zmod))) ))
    res <- R.utils::withTimeout(expr1, substitute = F, timeout = 30, onTimeout = "silent")
    if(is.null(res)){
      #res = c(c(mvtnorm::rmvnorm(n=1, mean = Mu.gam[1:pn0], sigma = as.positive.definite(Sig.gam)[1:pn0,1:pn0])), 
      #        c( TruncatedNormal::rtmvnorm(n=1, mu = Mu0.gam0[-c(1:pn0)], sigma = as.positive.definite(Sig.gam)[-c(1:pn0),-c(1:pn0)], lb =c(rep(-g0,length(Mu.gam)-ncol(Zmod))),
      #                                     ub = c(rep(g0,length(Mu.gam)-ncol(Zmod))) ) ) )
      res = c(tmvtnorm::rtmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam), lower =c(rep(-Inf, ncol(Zmod)),rep(-g0,length(Mu.gam)-ncol(Zmod))),
                                 upper = c(rep(Inf, ncol(Zmod)),rep(g0,length(Mu.gam)-ncol(Zmod))), algorithm="gibbs", burn.in.samples=100)) 
      }
    
   # expr2 <- quotec(tmvtnorm::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb =c(rep(-Inf, ncol(Zmod)),rep(-g0,length(Mu.gam)-ncol(Zmod))),
   #                             ub = c(rep(Inf, ncol(Zmod)),rep(g0,length(Mu.gam)-ncol(Zmod)))   ))
    
    res
    
    
  }
  updateGam <- cmpfun(updateGam)
  
  #siggam ~ IG(al0, bet0)
  al0 = 0.001
  bet0 = .005
  al0 = 2.5978252 #1
  bet0 = 0.4842963 #.005
  updateSigGam0 <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    val <- nimble::rinvgamma(n = 1, shape = al, scale = bet)
    ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  
  al0 = 0.001#1 #1
  bet0 = 0.001 #0.005 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    #val <- 
      #rgamma(n = 1, shape = al, scale = bet)
      nimble::rinvgamma(n = 1, shape = al, scale = bet)
    #ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  
  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e 
  #alpha.e = 1.0
  updatePik.e <- function(Thet, alpha.e){
    
    nk <- numeric(length(Thet$Cik.e))
    for(i in 1:length(Thet$Cik.e)){ nk[i] = sum(Thet$Cik.e == i)}
    
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$Cik.e) + nk
    
    rdirichlet(n=1, alpha.enew)
    
  }
  
  #-------- Update Cik.e  
  
  #ei <- Y - Z%*%Thet$Betaz - Thet$X%*%Thet$Gamma
  updateCik.e <- function(i, Thet,ei){
    #Y - Thet$C*abs(Thet$gam)*Thet$si - Thet$A*Thet$nui
    
    #  Yi.tld <- Y[i] - c(crossprod(Thet$Gamma, Thet$X[i,]))
    #pi.k  <- Thet$Pik.e*mapply(dnorm, x = Yi.tld, mean=Thet$muk.e, sd = sqrt(Thet$sig2k.e)) + .Machine$double.eps
    pik <- Thet$Pik.e*mapply(fGal.p0, e = ei[i], sig2 = sqrt(Thet$sig2k.e), gam = Thet$gamk, p = Thet$p, tau0 = Thet$tau0) #+ .Machine$double.eps
    which(rmultinom(n=1, size=1, prob = pik) == 1)
  }
  updateCik.e <- Vectorize(updateCik.e,"i") 
  
  #--- update Si
  #-- calss id for individual i (Matrix with columns of nui and si)
  updatenui <- function(Thet, Y, Z){
    
    ki = Thet$Cik.e
    bi <- ( ((Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ck[ki]*abs(Thet$gamk[ki])*Thet$si)^2)/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])))
    ai = ((Thet$Ak[ki]*Thet$Ak[ki])/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])) + 2/sqrt(Thet$sig2k.e[ki]))
    
    #mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) 
    #val = mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    #ifelse(val < .001, rexp(n = length(muui), rate = 1/sqrt(Thet$sig2k.e[Thet$Cik.e] )), val)
  }
  
  updatesi <- function(Thet, Y, Z){   
    #-- update si
    ki = Thet$Cik.e
    sig2si <- 1 /( 1/Thet$sig2k.e[ki] +  ((Thet$Ck[ki]*Thet$gamk[ki])^2)/ (Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui))
    muui <- sig2si*(Thet$Ck[ki]*abs(Thet$gamk[ki])*(Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ak[ki]*Thet$nui))/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui)
    
    #mapply(truncnorm::rtruncnorm, n=1, a = 0.01, mean = muui, sd = sqrt(sig2si))
    mapply(truncnorm::rtruncnorm, n=1, a = 0.001, mean = muui, sd = sqrt(sig2si))
    
    
  }
  
  
  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  #a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig=5/2; b.sig = 8/2; sdsig.prop = 0.2; sdgam.prop = .3  #2.153640 1.167331
  a.sig=2.153640; b.sig = 1.167331; sdsig.prop = 0.2; sdgam.prop = .3 # 7.133445 4.303263
  a.sig=7.133445; b.sig = 4.303263; sdsig.prop = 0.2; sdgam.prop = .3 #
  updatesig_gam <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop){
    
    a0 = 0.001
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      sig_newk <- truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gam <- Vectorize(updatesig_gam,"k")
  
  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10,10,10)*1
  varxi <- c(10,10,10)*.25
  varxi <- c(10,10,10)*.45 ## high aceeptance tunprop
  varxi <- c(10,10,10)*tunprop ##  tunprop
  #varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi){
    
    a0 = 0.1
    ab = 5 # THis works well
    ab = 10 # THis works well
    #varxi = 5
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]),Thet$gamk[k]),sigma = Sdsig.prop*varxi[k], lb = c(a0,0.95*Thet$gamL), ub = c(ab, .95*Thet$gamU)  )
      
      
      sig_newk <- Val[1] #truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] #truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu=c(sig_newk, gam_newk), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE) )
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes,"k")
  
  
  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices
  InvCov <- function(Mat){
    K <- nrow(Mat)
    J <- round(sqrt(ncol(Mat)))
    Res0 = matrix(0, nrow = K, ncol=ncol(Mat))
    for(k in 1:K){ Res0[k,] <- c(as.inverse(as.symmetric.matrix(matrix(Mat[k,],ncol=J, byrow=F)))) }
    Res0
  }
  
  
  #----------------------------------------------
  # Update things related to Wi 
  #--------------------------------------------
  #Piw
  #Cik.w
  #--- Update Pik.x  
  #alpha.W = alpha.w = 1.0
  updatePik.w <- function(Thet, alpha.W){
    Ku = nrow(Thet$Muk.w)
    nk <- numeric(Ku)
    for(k in 1:Ku){ nk[k] = sum(Thet$Cik.w == k)}
    
    #update the Pis
    alpha.wnew <- alpha.W/length(Thet$muk.w) + nk
    
    rdirichlet(n=1, alpha.wnew)
    
  }
  
  #--- Update Ci.k mixture of normals 
  dmvnormVec <- function(x, Mean, SigmaVc){
    Ku <- nrow(Mean)
    Val <- numeric(Ku)
    for(k in 1:Ku){ Val[k] <- mvtnorm::dmvnorm(x = x, mean = c(Mean[k,]), sigma = as.symmetric.matrix(matrix(SigmaVc[k,],ncol=length(x),byrow=F))) }
    Val
  }
  #dmvnormVec <- Vectorize(dmvnormVec,"k")
  #dmvnormVec(x = rep(0,6), Mn=Thet$Muk.w,SigmaVc = Thet$Sigk.w)
  
  updateCik.w <- function(i, Thet, W){ # mixture of normals 
    #Sig <- matrix(Thet$Sigk.w[id,], ncol = ncol(W), byrow = F)
    #kw = nrow(Thet$Muk.w)
    xi =  c(W[i,] - Thet$X[i,])
    #pi.k  = Thet$Pik.w*mapply(dmvnormVec, k = 1:Kw, MoreArgs = list(x = xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w))
    pi.k  = Thet$Pik.w*dmvnormVec(x=xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  
  #updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  #Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol=pn, nrow=pn));inSig0.w = as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.w <- function(Thet, W, Mu0.w, inSig0.w, Sig0.w){
    Ku = length(Thet$Pik.w)
    J = ncol(Thet$X)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol=J, nrow=J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = W - Thet$X 
    #Muk <- matrix(0, ncol=J,nrow=Ku)
    SigK <- list()
    for(k in 1:Ku){
      idk <- which(Thet$Cik.w == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow= nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.w)) ## sum of matrices inverse
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi.tld[idk,])) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
        Mu.k[k,] <- c(SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.w%*%Mu0.w))      ## some of mat
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(apply(t(Yi.tld[idk,]), 2, sum)) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
      }else{
        SigK[[k]] <- Sig0.w
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.w
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL; for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.w[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.w) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- 1.0*as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], sigma = SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.w[-Ku]) %*% SimMU.k)/Thet$Pik.w[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.w <- cmpfun(updateMuk.w)
  
  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w = pn + 2; Psi0.w = 0.5*(nu0.w-pn-1)*cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  updateSigk.w <- function(Thet, W, nu0.w, Psi0.w){
    
    Ku = nrow(Thet$Muk.w)
    m = ncol(W)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Wtl <- W - Thet$X - Thet$Muk.w[Thet$Cik.w,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.w == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.w  + nk
        Yi0 = matrix(c(Wtl[id,]), nrow= nk, byrow=F)
        Psin.w <- as.positive.definite(Psi0.w + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.w))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.w, Psi0.w))
      }
    }
    Res
  }
  updateSigk.w <- cmpfun(updateSigk.w)
  
  #----------------------------------------------
  # Update things related to M 
  #--------------------------------------------  
  
  #--- update Pik.m for epsilon_i
  #alpha.m = 1.0
  updatePik.m <- function(Thet, alpha.m){
    nk <- numeric(nrow(Thet$Muk.m))
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    
    #update the Pis
    alpha.mnew <- alpha.m/nrow(Thet$Muk.m) + nk
    
    c(rdirichlet(n=1, alpha.mnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  
  updateCik.m <- function(i, Thet, M){ # mixture of normals 
    #id <- Thet$Cik.m[i]
    #Sig <- matrix(Thet$Sigk.m[id,], ncol = ncol(M), byrow = F)
    Km = nrow(Thet$Muk.m)
    xi = M[i,] - Thet$Delta*Thet$X[i,]
    #pi.k  = Thet$Pik.m*mapply(dmvnormVec, k = 1:Km, MoreArgs = list(x = xi , Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m))
    pi.k  = Thet$Pik.m*dmvnormVec(x = xi, Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m = rep(0, pn);  Sig0.m = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.m = as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.m <- function(Thet, M, Mu0.m, inSig0.m, Sig0.m){
    #cat(dim(inSig0.m),"(--1)\n")
    Ku = length(Thet$Pik.m)
    J = ncol(M)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol = J, nrow = J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = M - Thet$X%*%diag(Thet$Betadel) 
    Muk <- matrix(0, ncol=J, nrow=Ku)
    for(k in 1:Ku){
      idk <- which(Thet$Cik.m == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow = nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.m)) ## sum of matrices inverse
        #cat(dim(SigK[[k]]),"(--2 \n")
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.m%*%Mu0.m)      ## some of mat
        #cat(dim(SigKWg),"(--3 \n")
      }else{
        SigK[[k]] <- Sig0.m
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.m
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL;  for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.m[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.m) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.m[-Ku]) %*% SimMU.k)/Thet$Pik.m[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.m <- cmpfun(updateMuk.m)
  
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m = pn + 2; Psi0.m = .5*(nu0.m-pn-1)*as.symmetric.matrix(cov(Wn))  #diag(pn)
  
  updateSigk.m <- function(Thet, M, nu0.m, Psi0.m){
    
    Ku = nrow(Thet$Muk.m)
    m = ncol(M)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Mtl <- M - Thet$X%*%diag(Thet$Betadel) - Thet$Muk.m[Thet$Cik.m,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.m == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.m  + nk
        Yi0 <- matrix(c(Mtl[id,]), nrow=nk, byrow=F)
        Psin.m <- as.positive.definite(Psi0.m + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.m))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.m, Psi0.m))
      }
    }
    Res
  }
  updateSigk.m <- cmpfun(updateSigk.m)
  
  #---- Update things related to delta
  quadFormVec <- function(i,X,sigma,Val){
    id = Val[i]
    as.symmetric.matrix(quad.form(as.inverse(as.symmetric.matrix(matrix(sigma[id,], ncol = length(X[i,]),byrow=F))), diag(c(X[i,])))) 
  }
  quadFormVec <- Vectorize(quadFormVec,"i",SIMPLIFY = F) ## Resulting in a list if precision matrices
  quadFormVec <- cmpfun(quadFormVec)
  
  quadFormVec3 <- function(i,Mi,X,sigma,Val){
    id = Val[i]
    c(quad.3form.inv(matrix(sigma[id,], ncol = length(X[i,]),byrow=F), diag(X[i,]),Mi[i,]))
  }
  quadFormVec3 <- Vectorize(quadFormVec3,"i", SIMPLIFY = F)
  quadFormVec3 <- cmpfun(quadFormVec3)
  
  #SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),   
  # prior values
  mu0.del = 1.0; #sig02.del = 100
  updateDelta <- function(Thet, M, mu0.del, sig02.del){
    n = nrow(M)
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X), Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    sig20 <- 1/( 1/sig02.del + sum(Res))
    mu <- sig20*(sum(Res3) + mu0.del/sig02.del)
    rtruncnorm(n=1, a=0, b=10, mean = mu, sd = sqrt(sig20))
    
    #c(sum(Res3), sum(Res),mu, sqrt(sig20), rtruncnorm(n=1, a=0, b=Inf, mean = mu, sd = sqrt(sig20)))
    
  }
  updateDelta <- cmpfun(updateDelta)
  
  
  # prior values
  mu0.del = rep(0,length(Thet$Betadel)); #sig02.del = 20
  Sig0.del = P.mat(length(Thet$Betadel))*.1   #Thet$siggam) # Inverse or precision covariance matrix
  
  updateFunDelta <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDelta <- cmpfun(updateFunDelta)
  
  
  SimDElVec <- function(i, Mi, X, Sigma, Val){
    id <- Val[i]
    Ot = mvtnorm::rmvnorm(n = 1,mean = Mi[i,],sigma = as.symmetric.matrix(matrix(Sigma[id,], ncol = length(X[i,]),byrow=F)))
  }
  SimDElVec <- Vectorize(SimDElVec, "i")
  
  #-- 
  DeltComp <- function(k,Xst, X){
    abs(sum(Xst[,k]*X[,k])/(sum(X[,k]^2)))
    #abs(diag(as.inverse(crossprod(X))%*%crossprod(X,Xst)))
  }
  DeltComp <- Vectorize(DeltComp,"k")
  
  updateFunDeltaSt <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Xst <- base::t(SimDElVec(1:nrow(M), Mitl, Thet$X, Thet$Sigk.m, Thet$Cik.m)) ## K by n matrix and now I get a n by K
    c(DeltComp(1:ncol(Thet$X), Xst, Thet$X))
    #c(DeltComp(Xst, Thet$X))
    
    # Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    # Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    # Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    # Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    # c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDeltaSt  <- cmpfun(updateFunDeltaSt )
  
  
  #----------------------------------------------
  # Update things related to X 
  #--------------------------------------------------  
  #--- update Pik.m for epsilon_i
  #alpha.x <- 1.0
  updatePik.x <- function(Thet, alpha.x){
    nk <- numeric(nrow(Thet$Muk.x))
    for(k in 1:nrow(Thet$Muk.x)){ nk[k] = sum(Thet$Cik.x == k)}
    
    #update the Pis
    alpha.xnew <- alpha.x/nrow(Thet$Muk.x) + nk
    
    c(rdirichlet(n=1, alpha.xnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  updateCik.x <- function(i, Thet){ # mixture of normals 
    #Kx = ncol(Thet$X) # J x J
    #id <- Thet$Cik.x[i]
    #Sig <- matrix(Thet$Sigk.x[id,], ncol = length(Thet$Gamma), byrow = F)
    Xi = Thet$X[i,]
    #pi.k  = Thet$Pik.x*mapply(dmvnormVec, k = 1:Kx, x = Xi,  Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x)
    pi.k  = Thet$Pik.x*dmvnormVec(x=Xi,Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x = numeric(pn); Sig0.x = .5*diag(pn); inSig0.x = 2*diag(pn)#as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x = rep(0, pn);  Sig0.x = .25*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w)+colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.x = as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.x <- function(Thet, Mu0.x, inSig0.x, Sig0.x){
    K = length(Thet$Pik.x)
    J = ncol(Thet$X)
    Res0 <- matrix(0, nrow = K, ncol = J)
    Mu.k <- matrix(0, nrow = K, ncol = J) # K x J
    #SigKWg <- matrix(0, ncol = ncol(Y),nrow=ncol(Y)) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = Thet$X 
    Muk <- matrix(0, ncol = J, nrow = K)
    for(k in 1:K){
      idk <- which(Thet$Cik.x == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 <- matrix(c(Yi.tld[idk,]), nrow=nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[k,], ncol=J, byrow=F)))
        SigK <- as.inverse(as.symmetric.matrix((nk*TpSi + inSig0.x))) ## sum of matrices inverse
        Muk <- SigK%*%(crossprod(TpSi, c(colSums(Yi0))) + inSig0.x%*%Mu0.x)      ## some of mat
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Muk, sigma = SigK))
      }else{
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Mu0.x, sigma = Sig0.x ))
      }
    }
    #--- we impose some constrainsts on the mean
    Mu.k
    
  }
  updateMuk.x <- cmpfun(updateMuk.x)
  
  nu0.x <- pn + 2 ; Psi0.x <- .5*diag(pn) #as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2 ; Psi0.x <- .5*(nu0.x-pn-1)*cov(.5*(Wn+Mn)) #as.symmetric.matrix(.25*cov(Wn))
  updateSigk.x <- function(Thet, nu0.x, Psi0.x){
    
    Ku = nrow(Thet$Muk.x)
    m = ncol(Thet$Muk.x)
    Xtl <- Thet$X - Thet$Muk.x[Thet$Cik.x,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.x == i)
      nk = length(id)
      if(nk > 0){
        nux <- nu0.x  + nk
        Yi0 <- matrix(c(Xtl[id,]), nrow=nk, byrow=F)
        Psin.x <- Psi0.x + crossprod(Yi0)
        Res[i, ] <- c(rinvwishart(nux, Psin.x))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.x, Psi0.x))
      }
    }
    Res
  }
  updateSigk.x <- cmpfun(updateSigk.x)
  # 
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn))) 
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn))) 
  
  A <- min(c(Wn, Mn)) - .1*diff(range(c(Wn, Mn))) 
  B <- max(c(Wn, Mn)) + .1*diff(range(c(Wn, Mn))) 
  
  udapateXi <- function(i, Thet, Y, W, M, Z, A=A, B=B){
    ie =Thet$Cik.e[i]
    iw = Thet$Cik.w[i]
    im = Thet$Cik.m[i]
    ix = Thet$Cik.x[i]
    J = ncol(Thet$X)
    
    Zmod <- Z #cbind(1,Z)
    Ytl = (Y - Zmod%*%Thet$BetaZ - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Wtl = W - Thet$Muk.w[Thet$Cik.w,]
    Mtl = (M - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
    Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
    Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
    
    #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
    Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
    
    Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))) 
    
    #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi <- cmpfun(udapateXi)
  
  
  udapateXi2A <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2A <- cmpfun(udapateXi2A)
  
  ## This approach of simulating Xi only uses W, M, and X
  
  udapateXi2 <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #Mu <- c(Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2 <- cmpfun(udapateXi2)
  
  
  udapateXi2V <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    #Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    #GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- (Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi2V <- Vectorize(udapateXi2V,"i")
  udapateXi2V <- cmpfun(udapateXi2V)
  
  #--- Write Rcpp function to speed things up
  {
    #
    
    
  }
  
  
  
  #Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)
  
  ##----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  #Nsim <- 5000
  #Nsim = 5000 
  MCMCSample <- list()
  
  #-- for X
  MCMCSample$X <- matrix(0,nrow = Nsim,ncol=length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined) 
  MCMCSample$Muk.x <- matrix(0,nrow = Nsim, ncol = Kx*pn)
  MCMCSample$Sigk.x <- matrix(0,nrow = Nsim,ncol=Kx*pn*pn)
  MCMCSample$Pik.x <- matrix(0,nrow = Nsim,ncol=Kx)
  MCMCSample$Cik.x <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for W
  MCMCSample$Muk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn)
  MCMCSample$Sigk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn*pn)
  MCMCSample$Pik.w <- matrix(0,nrow = Nsim,ncol=Kw)
  MCMCSample$Cik.w <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for M
  MCMCSample$Muk.m <- matrix(0,nrow = Nsim,ncol=Km*pn)
  MCMCSample$Sigk.m <- matrix(0,nrow = Nsim,ncol=Km*pn*pn)
  MCMCSample$Pik.m <- matrix(0,nrow = Nsim,ncol=Km)
  MCMCSample$Cik.m <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow=Nsim, ncol=length(Thet$Betadel))
  
  # For Y
  MCMCSample$gamk <- matrix(0,nrow = Nsim, ncol=Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Pik.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Cik.e <- matrix(0,nrow = Nsim, ncol=nrow(Wn))
  
  MCMCSample$Gamma <- matrix(0,nrow = Nsim, ncol=length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)
  
  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  # 
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  # 
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  # 
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  #Accept <- 
  # 
  MCMCSample$AcceptX <- matrix(0,nrow = Nsim, ncol=Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  #stop()
  #------------------------------------------------------------------
  #Nsim = 5000 
  #Nsim = 1000
  #Nsim = 3000
  #library(profvis)
  #profvis(
  #system.time(
  #cat("\n #--- MCMC is starting ----> \n")
  
  for(iter in 1:Nsim){
    if(iter %% 500 == 0){cat("iter=",iter,"\n")}
    #----------------------------------------------
    # Update things related to X 
    #--------------------------------------------  
    #--- update Pik.m for epsilon_i
    Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    
    #--- update Cik.m for epsilon_i based on mixture of normals
    Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x)) 
    
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    
    Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    Thet$InvSigk.x <- InvCov(Thet$Sigk.x)
    
    #Ytd = Y - Zmod%*%Thet$BetaZ
    if(iter%%100 == 0){
    Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B)}
    # Xn #
    #Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    #Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    #Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w, 
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)
    
    #Thet$X <- Xn #UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,]) 
    
    #}
    #dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))
    
    #Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    #Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    #Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    #Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w, 
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J))) 
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Kw)
    for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    #update the Pis
    alpha.wnew <- alpha.W/Kw + nk
    Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    
    #--- Update Ci.k mixture of normals
    #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Km)
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    #update the Pis
    alpha.mnew <- alpha.m/Kw + nk
    Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    
    #--- Update Ci.k mixture of normals
    #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    #Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))  
    #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    #if(iter%%50 == 0){
    Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    MCMCSample$Betdel[iter,] <- Thet$Betadel
    #}
    #---------------------------------------------
    #cat("#--- update things related to Y or epsilon ")
    #--------------------------------------------- 
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for(k in 1:length(Thet$sig2k.e)){ nk[k] = sum(Thet$Cik.e == k)}
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n=1, alpha.enew))
    
    #update the Cik.e
    ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
    #Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) #rep(1,n) #
    #Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    #update nui  and si
    #Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    #nuisi <- updatenuisi(Thet, Y, Zmod)
    #Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    #update Sigk.e
    #Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    #update sigk and gamk
    #--- update jointly sig2ke and gamke
    #Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi)
    Thet$sig2k.e <- Temp[1,];  Thet$gamk <- Temp[2,]; Thet$p <- Temp[3,]; MCMCSample$AcceptX[iter,] <- Temp[4,]; Thet$Ak <- Temp[5,]; Thet$Bk <- Temp[6,]; Thet$Ck <- Temp[7,];
    
    #update Gamma
    #Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    #Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <-  updateSigGam(Thet, al0, bet0, order = 2)# sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#
    
    #Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    # 
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    # 
    # #--- upodate beta_{z,m}
    # #--  Prior 
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    # 
    # #--- update X_i 
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX) 
    # 
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]); 
    # 
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb 
    
    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x))  ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise
    
    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w
    
    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    #MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e
    
    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  } 
  
  #)
  #)
  
  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample
}


#-- Using replicated W not M
BQBayes.ME_SoFRFuncSimFxFUI <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0 = 0.5,X, g0,tunprop){
  
  #------------------------------------
  #------------
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
  met <- c("f1","f2","f3","f4","f5")
  
  
  func <- function(Pik, n=1, size=1){
    which.max(rmultinom(n=1,size=1,prob = Pik))
  }
  funcCmp <- cmpfun(func)
  
  F2 <- function(Pik, n=1, size=1){
    
    apply(Pik,1,funcCmp)
  }
  F2Cmp <- cmpfun(F2)
  
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

  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  #pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  #pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  n = length(Y) 
  Zmod = cbind(1,Z)
  pn = c(5, 6, 20, 20)[which(n == c(100, 200,500,1000))]
  alpha.e = alpx
  alpha.w = alpha.W = alpx
  alpha.x = alpha.X = alpx
  alpha.m = alpx
  pn = pn #min(15, ceiling(length(Y)^{1/3.8})+4)
  
  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt #c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) # 
  Mn_Mod <- M%*%diag(1/Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20,"knots"), 1) ))
  Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a) 
  Wn <- (W[,,1]%*%bs2)/length(a)
  Xn <- (X%*%bs2)/length(a)
  
  cat("\n #--- Initial values...\n")  
  P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D #+ diag(rep(1e-3, K))
    return(P)
  }
  
  
  InitialValues <- function(Y, W, M, Z, df0=3, pn, Gx, a){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- Z #cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    lmFit <- lm(Y ~ -1 + Zmod + W_i) 
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- residuals.lm(lmFit)
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  InitialValuesQR <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z+Wn)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    #plot(a, bs2%*%lmFit$coefficients[-c(1:(ncol(Z+1)))], type="l")
    
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    Tp0 = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd)), upper=c(30, 0.95*max(bnd) ))
    Res$Hess <- Tp0$hessian
    
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  #Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, apply(W,c(1,2), mean), M, Z, df0=3, pn, Gx=1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  #Kvalues
  #Kc <- 5
  Kx = Thet0$Kx               ## number of X  cluster
  Ke = Thet0$Ke               ## number of ei cluster
  Km = Thet0$Km               ## number of ei cluster
  Kw = Thet0$Kw              ## number of omega_i cluster 
  
  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta   #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n=pn) #as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ #rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del#0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  #Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1]); Thet$gamU <- max(bnd[2])
  
  #bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) #c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
  
  Thet$Cik.x =  sample.int(Kx,size=n, replace = T) #Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e =  sample.int(Ke,size=n, replace = T) #Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w = sample.int(Kw,size=n, replace = T) #Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m = sample.int(Km,size=n, replace = T) #Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km
  
  
  Thet$Pik.x = rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)#Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e = rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)#Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w = rdirch(n=1,alpha = rep(alpha.e, Kw)/Kw) #Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m = rdirch(n=1,alpha = rep(alpha.m, Km)/Km)#Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw
  
  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw
  
  Thet$sig2k.e  = nimble::rinvgamma(n=Ke, shape=5/2,scale=8/2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n=Ke, min = .95*Thet$gamL, max=.95*Thet$gamU)
  Thet$p <- 1*(Thet$gamk < 0) + (Thet$tau0 - 1*(Thet$gamk < 0))/(2*pnorm(-abs(Thet$gamk))*exp(0.5*Thet$gamk*Thet$gamk))
  Thet$Ak <- (1-2*Thet$p)/(Thet$p*(1-Thet$p)); Thet$Bk <- 2/(Thet$p*(1-Thet$p)); Thet$Ck <- 1/( 1*(Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean=0, sd=sqrt(Thet$sig2k.e[1])) ; Thet$nui <- rexp(n=n, rate = 1/Thet$sig2k.e[1]) 
  #Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix
  
  Thet$Sigk.w <- repmat(c(.5*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw*pn), nrow = Kw)          #- vectors of length Kw and Kw respectively
  
  Thet$Sigk.x <- repmat(c(.5*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$Muk.x  <- matrix(rnorm(n=Kx*pn), nrow=Kx)   #- vectors of length Kx and Kx respectively
  
  Thet$Sigk.m <- repmat(c(.5*diag(pn)), Km, 1)     #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2*diag(pn)), Km, 1)
  Thet$Muk.m  <- matrix(rnorm(n=Km*pn), nrow = Km)   #- vectors of length Kx and Kx respectively
  
  A <- min(c(c(Wn), c(Mn))) - .5*diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5*diff(range(c(Wn, Mn)))
  #Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  #matrix(rnorm(n = n*pn), ncol=pn)
  #Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5*(Mn+Wn) #Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))
  
  
  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
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
  
  
  #method = "L-BFGS-B"
  #method = "BFGS"
  
  #Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  #Res
  
  #Res$hessian
  #Res$hessian/n
  #solve(Res$hessian/n)
  
  #LogPlostGalLik(c(.05,), c(ei), Thet)
  
  #-------- Update Beta (vector of length pn) 
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #   
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma 
  #   
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #   
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #   
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #   
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)
  
  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together) 
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 = ncol(Z) +1
  Pgam <- P.mat(pn)*tauval
  Sig0.gam0 = diag(pn0); Mu0.gam0 = numeric(pn0+pn);
  #InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))
  
  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.gam0%*%Mu0.gam0))
    
    #c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb =c(rep(-Inf, ncol(Zmod)),rep(-g0,length(Mu.gam)-ncol(Zmod))),
                                ub = c(rep(Inf, ncol(Zmod)),rep(g0,length(Mu.gam)-ncol(Zmod)))   ))
    
  }
  updateGam <- cmpfun(updateGam)
  
  #siggam ~ IG(al0, bet0)
  al0 = 0.001
  bet0 = .005
  al0 = 2.5978252 #1
  bet0 = 0.4842963 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    val <- nimble::rinvgamma(n = 1, shape = al, scale = bet)
    ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  
  al0 = 0.001 #1 #1
  bet0 = 0.001 #0.005 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    #rgamma(n = 1, shape = al, scale = bet)
    nimble::rinvgamma(n = 1, shape = al, scale = bet)
    #ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e 
  #alpha.e = 1.0
  updatePik.e <- function(Thet, alpha.e){
    
    nk <- numeric(length(Thet$Cik.e))
    for(i in 1:length(Thet$Cik.e)){ nk[i] = sum(Thet$Cik.e == i)}
    
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$Cik.e) + nk
    
    rdirichlet(n=1, alpha.enew)
    
  }
  
  #-------- Update Cik.e  
  
  #ei <- Y - Z%*%Thet$Betaz - Thet$X%*%Thet$Gamma
  updateCik.e <- function(i, Thet,ei){
    #Y - Thet$C*abs(Thet$gam)*Thet$si - Thet$A*Thet$nui
    
    #  Yi.tld <- Y[i] - c(crossprod(Thet$Gamma, Thet$X[i,]))
    #pi.k  <- Thet$Pik.e*mapply(dnorm, x = Yi.tld, mean=Thet$muk.e, sd = sqrt(Thet$sig2k.e)) + .Machine$double.eps
    pik <- Thet$Pik.e*mapply(fGal.p0, e = ei[i], sig2 = sqrt(Thet$sig2k.e), gam = Thet$gamk, p = Thet$p, tau0 = Thet$tau0) #+ .Machine$double.eps
    which(rmultinom(n=1, size=1, prob = pik) == 1)
  }
  updateCik.e <- Vectorize(updateCik.e,"i") 
  
  #--- update Si
  #-- calss id for individual i (Matrix with columns of nui and si)
  updatenui <- function(Thet, Y, Z){
    
    ki = Thet$Cik.e
    bi <- ( ((Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ck[ki]*abs(Thet$gamk[ki])*Thet$si)^2)/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])))
    ai = ((Thet$Ak[ki]*Thet$Ak[ki])/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])) + 2/sqrt(Thet$sig2k.e[ki]))
    
    #mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) 
    #val = mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    #ifelse(val < .001, rexp(n = length(muui), rate = 1/sqrt(Thet$sig2k.e[Thet$Cik.e] )), val)
  }
  
  updatesi <- function(Thet, Y, Z){   
    #-- update si
    ki = Thet$Cik.e
    sig2si <- 1 /( 1/Thet$sig2k.e[ki] +  ((Thet$Ck[ki]*Thet$gamk[ki])^2)/ (Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui))
    muui <- sig2si*(Thet$Ck[ki]*abs(Thet$gamk[ki])*(Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ak[ki]*Thet$nui))/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui)
    
    #mapply(truncnorm::rtruncnorm, n=1, a = 0.01, mean = muui, sd = sqrt(sig2si))
    mapply(truncnorm::rtruncnorm, n=1, a = 0.001, mean = muui, sd = sqrt(sig2si))
    
    
  }
  
  
  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  #a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig=5/2; b.sig = 8/2; sdsig.prop = 0.2; sdgam.prop = .3  #2.153640 1.167331
  a.sig=2.153640; b.sig = 1.167331; sdsig.prop = 0.2; sdgam.prop = .3 # 7.133445 4.303263
  a.sig=7.133445; b.sig = 4.303263; sdsig.prop = 0.2; sdgam.prop = .3 #
  updatesig_gam <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop){
    
    a0 = 0.001
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      sig_newk <- truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gam <- Vectorize(updatesig_gam,"k")
  
  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10,10,10)*1
  #varxi <- c(10,10,10)*.25
  #varxi <- c(10,10,10)*.15 ## high acceptance rate
  varxi <- c(10,10,10)*.35 ##-- High acceptance rate
  varxi <- c(10,10,10)*tunprop ##-- 
  #varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi){
    
    a0 = 0.1
    ab = 5 ## This works well
    ab = 10
    #varxi = 5
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]),Thet$gamk[k]),sigma = Sdsig.prop*varxi[k], lb = c(a0,0.95*Thet$gamL), ub = c(ab, .95*Thet$gamU)  )
      
      
      sig_newk <- Val[1] #truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] #truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu=c(sig_newk, gam_newk), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE) )
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes,"k")
  
  
  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices
  InvCov <- function(Mat){
    K <- nrow(Mat)
    J <- round(sqrt(ncol(Mat)))
    Res0 = matrix(0, nrow = K, ncol=ncol(Mat))
    for(k in 1:K){ Res0[k,] <- c(as.inverse(as.symmetric.matrix(matrix(Mat[k,],ncol=J, byrow=F)))) }
    Res0
  }
  
  
  #----------------------------------------------
  # Update things related to Wi 
  #--------------------------------------------
  #Piw
  #Cik.w
  #--- Update Pik.x  
  #alpha.W = alpha.w = 1.0
  updatePik.w <- function(Thet, alpha.W){
    Ku = nrow(Thet$Muk.w)
    nk <- numeric(Ku)
    for(k in 1:Ku){ nk[k] = sum(Thet$Cik.w == k)}
    
    #update the Pis
    alpha.wnew <- alpha.W/length(Thet$muk.w) + nk
    
    rdirichlet(n=1, alpha.wnew)
    
  }
  
  #--- Update Ci.k mixture of normals 
  dmvnormVec <- function(x, Mean, SigmaVc){
    Ku <- nrow(Mean)
    Val <- numeric(Ku)
    for(k in 1:Ku){ Val[k] <- mvtnorm::dmvnorm(x = x, mean = c(Mean[k,]), sigma = as.symmetric.matrix(matrix(SigmaVc[k,],ncol=length(x),byrow=F))) }
    Val
  }
  #dmvnormVec <- Vectorize(dmvnormVec,"k")
  #dmvnormVec(x = rep(0,6), Mn=Thet$Muk.w,SigmaVc = Thet$Sigk.w)
  
  updateCik.w <- function(i, Thet, W){ # mixture of normals 
    #Sig <- matrix(Thet$Sigk.w[id,], ncol = ncol(W), byrow = F)
    #kw = nrow(Thet$Muk.w)
    xi =  c(W[i,] - Thet$X[i,])
    #pi.k  = Thet$Pik.w*mapply(dmvnormVec, k = 1:Kw, MoreArgs = list(x = xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w))
    pi.k  = Thet$Pik.w*dmvnormVec(x=xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  
  #updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  #Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol=pn, nrow=pn));inSig0.w = as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.w <- function(Thet, W, Mu0.w, inSig0.w, Sig0.w){
    Ku = length(Thet$Pik.w)
    J = ncol(Thet$X)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol=J, nrow=J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = W - Thet$X 
    #Muk <- matrix(0, ncol=J,nrow=Ku)
    SigK <- list()
    for(k in 1:Ku){
      idk <- which(Thet$Cik.w == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow= nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.w)) ## sum of matrices inverse
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi.tld[idk,])) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
        Mu.k[k,] <- c(SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.w%*%Mu0.w))      ## some of mat
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(apply(t(Yi.tld[idk,]), 2, sum)) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
      }else{
        SigK[[k]] <- Sig0.w
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.w
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL; for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.w[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.w) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- 1.0*as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], sigma = SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.w[-Ku]) %*% SimMU.k)/Thet$Pik.w[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.w <- cmpfun(updateMuk.w)
  
  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w = pn + 2; Psi0.w = 0.5*(nu0.w-pn-1)*cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  updateSigk.w <- function(Thet, W, nu0.w, Psi0.w){
    
    Ku = nrow(Thet$Muk.w)
    m = ncol(W)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Wtl <- W - Thet$X - Thet$Muk.w[Thet$Cik.w,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.w == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.w  + nk
        Yi0 = matrix(c(Wtl[id,]), nrow= nk, byrow=F)
        Psin.w <- as.positive.definite(Psi0.w + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.w))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.w, Psi0.w))
      }
    }
    Res
  }
  updateSigk.w <- cmpfun(updateSigk.w)
  
  #----------------------------------------------
  # Update things related to M 
  #--------------------------------------------  
  
  #--- update Pik.m for epsilon_i
  #alpha.m = 1.0
  updatePik.m <- function(Thet, alpha.m){
    nk <- numeric(nrow(Thet$Muk.m))
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    
    #update the Pis
    alpha.mnew <- alpha.m/nrow(Thet$Muk.m) + nk
    
    c(rdirichlet(n=1, alpha.mnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  
  updateCik.m <- function(i, Thet, M){ # mixture of normals 
    #id <- Thet$Cik.m[i]
    #Sig <- matrix(Thet$Sigk.m[id,], ncol = ncol(M), byrow = F)
    Km = nrow(Thet$Muk.m)
    xi = M[i,] - Thet$Delta*Thet$X[i,]
    #pi.k  = Thet$Pik.m*mapply(dmvnormVec, k = 1:Km, MoreArgs = list(x = xi , Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m))
    pi.k  = Thet$Pik.m*dmvnormVec(x = xi, Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m = rep(0, pn);  Sig0.m = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.m = as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.m <- function(Thet, M, Mu0.m, inSig0.m, Sig0.m){
    #cat(dim(inSig0.m),"(--1)\n")
    Ku = length(Thet$Pik.m)
    J = ncol(M)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol = J, nrow = J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = M - Thet$X%*%diag(Thet$Betadel) 
    Muk <- matrix(0, ncol=J, nrow=Ku)
    for(k in 1:Ku){
      idk <- which(Thet$Cik.m == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow = nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.m)) ## sum of matrices inverse
        #cat(dim(SigK[[k]]),"(--2 \n")
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.m%*%Mu0.m)      ## some of mat
        #cat(dim(SigKWg),"(--3 \n")
      }else{
        SigK[[k]] <- Sig0.m
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.m
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL;  for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.m[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.m) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.m[-Ku]) %*% SimMU.k)/Thet$Pik.m[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.m <- cmpfun(updateMuk.m)
  
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m = pn + 2; Psi0.m = .5*(nu0.m-pn-1)*as.symmetric.matrix(cov(Wn))  #diag(pn)
  
  updateSigk.m <- function(Thet, M, nu0.m, Psi0.m){
    
    Ku = nrow(Thet$Muk.m)
    m = ncol(M)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Mtl <- M - Thet$X%*%diag(Thet$Betadel) - Thet$Muk.m[Thet$Cik.m,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.m == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.m  + nk
        Yi0 <- matrix(c(Mtl[id,]), nrow=nk, byrow=F)
        Psin.m <- as.positive.definite(Psi0.m + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.m))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.m, Psi0.m))
      }
    }
    Res
  }
  updateSigk.m <- cmpfun(updateSigk.m)
  
  #---- Update things related to delta
  quadFormVec <- function(i,X,sigma,Val){
    id = Val[i]
    as.symmetric.matrix(quad.form(as.inverse(as.symmetric.matrix(matrix(sigma[id,], ncol = length(X[i,]),byrow=F))), diag(c(X[i,])))) 
  }
  quadFormVec <- Vectorize(quadFormVec,"i",SIMPLIFY = F) ## Resulting in a list if precision matrices
  quadFormVec <- cmpfun(quadFormVec)
  
  quadFormVec3 <- function(i,Mi,X,sigma,Val){
    id = Val[i]
    c(quad.3form.inv(matrix(sigma[id,], ncol = length(X[i,]),byrow=F), diag(X[i,]),Mi[i,]))
  }
  quadFormVec3 <- Vectorize(quadFormVec3,"i", SIMPLIFY = F)
  quadFormVec3 <- cmpfun(quadFormVec3)
  
  #SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),   
  # prior values
  mu0.del = 1.0; #sig02.del = 100
  updateDelta <- function(Thet, M, mu0.del, sig02.del){
    n = nrow(M)
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X), Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    sig20 <- 1/( 1/sig02.del + sum(Res))
    mu <- sig20*(sum(Res3) + mu0.del/sig02.del)
    rtruncnorm(n=1, a=0, b=10, mean = mu, sd = sqrt(sig20))
    
    #c(sum(Res3), sum(Res),mu, sqrt(sig20), rtruncnorm(n=1, a=0, b=Inf, mean = mu, sd = sqrt(sig20)))
    
  }
  updateDelta <- cmpfun(updateDelta)
  
  
  # prior values
  mu0.del = rep(0,length(Thet$Betadel)); #sig02.del = 20
  Sig0.del = P.mat(length(Thet$Betadel))*.1   #Thet$siggam) # Inverse or precision covariance matrix
  
  updateFunDelta <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDelta <- cmpfun(updateFunDelta)
  
  
  SimDElVec <- function(i, Mi, X, Sigma, Val){
    id <- Val[i]
    Ot = mvtnorm::rmvnorm(n = 1,mean = Mi[i,],sigma = as.symmetric.matrix(matrix(Sigma[id,], ncol = length(X[i,]),byrow=F)))
  }
  SimDElVec <- Vectorize(SimDElVec, "i")
  
  #-- 
  DeltComp <- function(k,Xst, X){
    abs(sum(Xst[,k]*X[,k])/(sum(X[,k]^2)))
    #abs(diag(as.inverse(crossprod(X))%*%crossprod(X,Xst)))
  }
  DeltComp <- Vectorize(DeltComp,"k")
  
  updateFunDeltaSt <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Xst <- base::t(SimDElVec(1:nrow(M), Mitl, Thet$X, Thet$Sigk.m, Thet$Cik.m)) ## K by n matrix and now I get a n by K
    c(DeltComp(1:ncol(Thet$X), Xst, Thet$X))
    #c(DeltComp(Xst, Thet$X))
    
    # Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    # Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    # Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    # Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    # c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDeltaSt  <- cmpfun(updateFunDeltaSt )
  
  
  #----------------------------------------------
  # Update things related to X 
  #--------------------------------------------------  
  #--- update Pik.m for epsilon_i
  #alpha.x <- 1.0
  updatePik.x <- function(Thet, alpha.x){
    nk <- numeric(nrow(Thet$Muk.x))
    for(k in 1:nrow(Thet$Muk.x)){ nk[k] = sum(Thet$Cik.x == k)}
    
    #update the Pis
    alpha.xnew <- alpha.x/nrow(Thet$Muk.x) + nk
    
    c(rdirichlet(n=1, alpha.xnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  updateCik.x <- function(i, Thet){ # mixture of normals 
    #Kx = ncol(Thet$X) # J x J
    #id <- Thet$Cik.x[i]
    #Sig <- matrix(Thet$Sigk.x[id,], ncol = length(Thet$Gamma), byrow = F)
    Xi = Thet$X[i,]
    #pi.k  = Thet$Pik.x*mapply(dmvnormVec, k = 1:Kx, x = Xi,  Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x)
    pi.k  = Thet$Pik.x*dmvnormVec(x=Xi,Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x = numeric(pn); Sig0.x = .5*diag(pn); inSig0.x = 2*diag(pn)#as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x = rep(0, pn);  Sig0.x = .25*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w)+colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.x = as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.x <- function(Thet, Mu0.x, inSig0.x, Sig0.x){
    K = length(Thet$Pik.x)
    J = ncol(Thet$X)
    Res0 <- matrix(0, nrow = K, ncol = J)
    Mu.k <- matrix(0, nrow = K, ncol = J) # K x J
    #SigKWg <- matrix(0, ncol = ncol(Y),nrow=ncol(Y)) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = Thet$X 
    Muk <- matrix(0, ncol = J, nrow = K)
    for(k in 1:K){
      idk <- which(Thet$Cik.x == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 <- matrix(c(Yi.tld[idk,]), nrow=nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[k,], ncol=J, byrow=F)))
        SigK <- as.inverse(as.symmetric.matrix((nk*TpSi + inSig0.x))) ## sum of matrices inverse
        Muk <- SigK%*%(crossprod(TpSi, c(colSums(Yi0))) + inSig0.x%*%Mu0.x)      ## some of mat
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Muk, sigma = SigK))
      }else{
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Mu0.x, sigma = Sig0.x ))
      }
    }
    #--- we impose some constrainsts on the mean
    Mu.k
    
  }
  updateMuk.x <- cmpfun(updateMuk.x)
  
  nu0.x <- pn + 2 ; Psi0.x <- .5*diag(pn) #as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2 ; Psi0.x <- .5*(nu0.x-pn-1)*cov(.5*(Wn+Mn)) #as.symmetric.matrix(.25*cov(Wn))
  updateSigk.x <- function(Thet, nu0.x, Psi0.x){
    
    Ku = nrow(Thet$Muk.x)
    m = ncol(Thet$Muk.x)
    Xtl <- Thet$X - Thet$Muk.x[Thet$Cik.x,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.x == i)
      nk = length(id)
      if(nk > 0){
        nux <- nu0.x  + nk
        Yi0 <- matrix(c(Xtl[id,]), nrow=nk, byrow=F)
        Psin.x <- Psi0.x + crossprod(Yi0)
        Res[i, ] <- c(rinvwishart(nux, Psin.x))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.x, Psi0.x))
      }
    }
    Res
  }
  updateSigk.x <- cmpfun(updateSigk.x)
  # 
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn))) 
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn))) 
  
  A <- min(c(Wn, Mn)) - .1*diff(range(c(Wn, Mn))) 
  B <- max(c(Wn, Mn)) + .1*diff(range(c(Wn, Mn))) 
  
  udapateXi <- function(i, Thet, Y, W, M, Z, A=A, B=B){
    ie =Thet$Cik.e[i]
    iw = Thet$Cik.w[i]
    im = Thet$Cik.m[i]
    ix = Thet$Cik.x[i]
    J = ncol(Thet$X)
    
    Zmod <- Z #cbind(1,Z)
    Ytl = (Y - Zmod%*%Thet$BetaZ - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Wtl = W - Thet$Muk.w[Thet$Cik.w,]
    Mtl = (M - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
    Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
    Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
    
    #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
    Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
    
    Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))) 
    
    #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi <- cmpfun(udapateXi)
  
  
  udapateXi2A <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2A <- cmpfun(udapateXi2A)
  
  ## This approach of simulating Xi only uses W, M, and X
  
  udapateXi2 <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #Mu <- c(Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2 <- cmpfun(udapateXi2)
  
  
  udapateXi2V <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    #Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    #GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- (Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi2V <- Vectorize(udapateXi2V,"i")
  udapateXi2V <- cmpfun(udapateXi2V)
  
  #--- Write Rcpp function to speed things up
  {
    #
    
    
  }
  
  
  
  #Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)
  
  ##----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  #Nsim <- 5000
  #Nsim = 5000 
  MCMCSample <- list()
  
  #-- for X
  MCMCSample$X <- matrix(0,nrow = Nsim,ncol=length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined) 
  MCMCSample$Muk.x <- matrix(0,nrow = Nsim, ncol = Kx*pn)
  MCMCSample$Sigk.x <- matrix(0,nrow = Nsim,ncol=Kx*pn*pn)
  MCMCSample$Pik.x <- matrix(0,nrow = Nsim,ncol=Kx)
  MCMCSample$Cik.x <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for W
  MCMCSample$Muk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn)
  MCMCSample$Sigk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn*pn)
  MCMCSample$Pik.w <- matrix(0,nrow = Nsim,ncol=Kw)
  MCMCSample$Cik.w <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for M
  MCMCSample$Muk.m <- matrix(0,nrow = Nsim,ncol=Km*pn)
  MCMCSample$Sigk.m <- matrix(0,nrow = Nsim,ncol=Km*pn*pn)
  MCMCSample$Pik.m <- matrix(0,nrow = Nsim,ncol=Km)
  MCMCSample$Cik.m <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow=Nsim, ncol=length(Thet$Betadel))
  
  # For Y
  MCMCSample$gamk <- matrix(0,nrow = Nsim, ncol=Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Pik.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Cik.e <- matrix(0,nrow = Nsim, ncol=nrow(Wn))
  
  MCMCSample$Gamma <- matrix(0,nrow = Nsim, ncol=length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)
  
  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  # 
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  # 
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  # 
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  #Accept <- 
  # 
  MCMCSample$AcceptX <- matrix(0,nrow = Nsim, ncol=Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  #stop()
  #------------------------------------------------------------------
  #Nsim = 5000 
  #Nsim = 1000
  #Nsim = 3000
  #library(profvis)
  #profvis(
  #system.time(
  #cat("\n #--- MCMC is starting ----> \n")
  Thet$X <- (as.matrix(FUI(model="gaussian",smooth=T, W,silent = FALSE)) %*%bs2)/length(a)
  for(iter in 1:Nsim){
    if(iter %% 500 == 0){cat("iter=",iter,"\n")}
    #----------------------------------------------
    # Update things related to X 
    #--------------------------------------------  
    # #--- update Pik.m for epsilon_i
    # Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    # 
    # #--- update Cik.m for epsilon_i based on mixture of normals
    # Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    # #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x)) 
    # 
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    # 
    # Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    # Thet$InvSigk.x <- InvCov(Thet$Sigk.x)
    
    #Ytd = Y - Zmod%*%Thet$BetaZ
    #if(iter%%10 == 0){
    #Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B) # Xn #
    #Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    #Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    #Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w, 
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)
    
    #Thet$X <- Xn #UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,]) 
    
    #}
    #dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))
    
    #Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    #Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    #Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    #Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w, 
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J))) 
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Kw)
    # for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    # #update the Pis
    # alpha.wnew <- alpha.W/Kw + nk
    # Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    # 
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    # Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    # Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    # Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Km)
    # for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    # #update the Pis
    # alpha.mnew <- alpha.m/Kw + nk
    # Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    # 
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    # Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    # #Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))  
    # #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    # Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    # Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # #if(iter%%50 == 0){
    # Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    # MCMCSample$Betdel[iter,] <- Thet$Betadel
    # #}
    #---------------------------------------------
    #cat("#--- update things related to Y or epsilon ")
    #--------------------------------------------- 
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for(k in 1:length(Thet$sig2k.e)){ nk[k] = sum(Thet$Cik.e == k)}
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n=1, alpha.enew))
    
    #update the Cik.e
    ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
    #Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) #rep(1,n) #
    #Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    #update nui  and si
    #Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    #nuisi <- updatenuisi(Thet, Y, Zmod)
    #Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    #update Sigk.e
    #Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    #update sigk and gamk
    #--- update jointly sig2ke and gamke
    #Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi)
    Thet$sig2k.e <- Temp[1,];  Thet$gamk <- Temp[2,]; Thet$p <- Temp[3,]; MCMCSample$AcceptX[iter,] <- Temp[4,]; Thet$Ak <- Temp[5,]; Thet$Bk <- Temp[6,]; Thet$Ck <- Temp[7,];
    
    #update Gamma
    #Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    #Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <-  updateSigGam(Thet, al0, bet0, order = 2)# sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#
    
    #Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    # 
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    # 
    # #--- upodate beta_{z,m}
    # #--  Prior 
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    # 
    # #--- update X_i 
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX) 
    # 
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]); 
    # 
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb 
    
    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x))  ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise
    
    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w
    
    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    #MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e
    
    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  } 
  
  #)
  #)
  
  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample$Xpred <- Thet$X
  
  MCMCSample
}



#--- Naive approach using W
#-- Using replicated W not M
BQBayes.ME_SoFRFuncSimFxNaiveW <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0 = 0.5,X, g0,tunprop){
  
  #------------------------------------
  #------------
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
  met <- c("f1","f2","f3","f4","f5")
  
  
  func <- function(Pik, n=1, size=1){
    which.max(rmultinom(n=1,size=1,prob = Pik))
  }
  funcCmp <- cmpfun(func)
  
  F2 <- function(Pik, n=1, size=1){
    
    apply(Pik,1,funcCmp)
  }
  F2Cmp <- cmpfun(F2)
  
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
  
  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  #pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  #pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  n = length(Y) 
  Zmod = cbind(1,Z)
  pn = c(5, 6, 20, 20)[which(n == c(100, 200,500,1000))]
  alpha.e = alpx
  alpha.w = alpha.W = alpx
  alpha.x = alpha.X = alpx
  alpha.m = alpx
  pn = pn #min(15, ceiling(length(Y)^{1/3.8})+4)
  
  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt #c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) # 
  Mn_Mod <- M%*%diag(1/Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20,"knots"), 1) ))
  Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a) 
  Wn <- (W[,,1]%*%bs2)/length(a)
  Xn <- (X%*%bs2)/length(a)
  
  cat("\n #--- Initial values...\n")  
  P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D #+ diag(rep(1e-3, K))
    return(P)
  }
  
  
  InitialValues <- function(Y, W, M, Z, df0=3, pn, Gx, a){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- Z #cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    lmFit <- lm(Y ~ -1 + Zmod + W_i) 
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- residuals.lm(lmFit)
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  InitialValuesQR <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z+Wn)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    #plot(a, bs2%*%lmFit$coefficients[-c(1:(ncol(Z+1)))], type="l")
    
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    Tp0 = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd)), upper=c(30, 0.95*max(bnd) ))
    Res$Hess <- Tp0$hessian
    
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  #Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, apply(W,c(1,2), mean), M, Z, df0=3, pn, Gx=1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  #Kvalues
  #Kc <- 5
  
  Kx = Thet0$Kx               ## number of X  cluster
  Ke = Thet0$Ke               ## number of ei cluster
  Km = Thet0$Km               ## number of ei cluster
  Kw = Thet0$Kw              ## number of omega_i cluster 
  
  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta   #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n=pn) #as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ #rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del#0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  #Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1]); Thet$gamU <- max(bnd[2])
  
  #bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) #c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
  
  Thet$Cik.x =  sample.int(Kx,size=n, replace = T) #Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e =  sample.int(Ke,size=n, replace = T) #Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w = sample.int(Kw,size=n, replace = T) #Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m = sample.int(Km,size=n, replace = T) #Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km
  
  
  Thet$Pik.x = rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)#Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e = rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)#Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w = rdirch(n=1,alpha = rep(alpha.e, Kw)/Kw) #Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m = rdirch(n=1,alpha = rep(alpha.m, Km)/Km)#Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw
  
  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw
  
  Thet$sig2k.e  = nimble::rinvgamma(n=Ke, shape=5/2,scale=8/2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n=Ke, min = .95*Thet$gamL, max=.95*Thet$gamU)
  Thet$p <- 1*(Thet$gamk < 0) + (Thet$tau0 - 1*(Thet$gamk < 0))/(2*pnorm(-abs(Thet$gamk))*exp(0.5*Thet$gamk*Thet$gamk))
  Thet$Ak <- (1-2*Thet$p)/(Thet$p*(1-Thet$p)); Thet$Bk <- 2/(Thet$p*(1-Thet$p)); Thet$Ck <- 1/( 1*(Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean=0, sd=sqrt(Thet$sig2k.e[1])) ; Thet$nui <- rexp(n=n, rate = 1/Thet$sig2k.e[1]) 
  #Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix
  
  Thet$Sigk.w <- repmat(c(.5*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw*pn), nrow = Kw)          #- vectors of length Kw and Kw respectively
  
  Thet$Sigk.x <- repmat(c(.5*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$Muk.x  <- matrix(rnorm(n=Kx*pn), nrow=Kx)   #- vectors of length Kx and Kx respectively
  
  Thet$Sigk.m <- repmat(c(.5*diag(pn)), Km, 1)     #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2*diag(pn)), Km, 1)
  Thet$Muk.m  <- matrix(rnorm(n=Km*pn), nrow = Km)   #- vectors of length Kx and Kx respectively
  
  A <- min(c(c(Wn), c(Mn))) - .5*diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5*diff(range(c(Wn, Mn)))
  #Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  #matrix(rnorm(n = n*pn), ncol=pn)
  #Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5*(Mn+Wn) #Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))
  
  
  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
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
  
  
  #method = "L-BFGS-B"
  #method = "BFGS"
  
  #Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  #Res
  
  #Res$hessian
  #Res$hessian/n
  #solve(Res$hessian/n)
  
  #LogPlostGalLik(c(.05,), c(ei), Thet)
  
  #-------- Update Beta (vector of length pn) 
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #   
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma 
  #   
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #   
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #   
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #   
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)
  
  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together) 
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 = ncol(Z) +1
  Pgam <- P.mat(pn)*tauval
  Sig0.gam0 = diag(pn0); Mu0.gam0 = numeric(pn0+pn);
  #InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))
  
  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.gam0%*%Mu0.gam0))
    
    #c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb =c(rep(-Inf, ncol(Zmod)),rep(-g0,length(Mu.gam)-ncol(Zmod))),
                                ub = c(rep(Inf, ncol(Zmod)),rep(g0,length(Mu.gam)-ncol(Zmod)))   ))
    
  }
  updateGam <- cmpfun(updateGam)
  
  #siggam ~ IG(al0, bet0)
  al0 = 0.001
  bet0 = .005
  al0 = 2.5978252 #1
  bet0 = 0.4842963 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    val <- nimble::rinvgamma(n = 1, shape = al, scale = bet)
    ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  
  al0 = 0.001 #1 #1
  bet0 = 0.001 #0.005 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    #rgamma(n = 1, shape = al, scale = bet)
    nimble::rinvgamma(n = 1, shape = al, scale = bet)
    #ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e 
  #alpha.e = 1.0
  updatePik.e <- function(Thet, alpha.e){
    
    nk <- numeric(length(Thet$Cik.e))
    for(i in 1:length(Thet$Cik.e)){ nk[i] = sum(Thet$Cik.e == i)}
    
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$Cik.e) + nk
    
    rdirichlet(n=1, alpha.enew)
    
  }
  
  #-------- Update Cik.e  
  
  #ei <- Y - Z%*%Thet$Betaz - Thet$X%*%Thet$Gamma
  updateCik.e <- function(i, Thet,ei){
    #Y - Thet$C*abs(Thet$gam)*Thet$si - Thet$A*Thet$nui
    
    #  Yi.tld <- Y[i] - c(crossprod(Thet$Gamma, Thet$X[i,]))
    #pi.k  <- Thet$Pik.e*mapply(dnorm, x = Yi.tld, mean=Thet$muk.e, sd = sqrt(Thet$sig2k.e)) + .Machine$double.eps
    pik <- Thet$Pik.e*mapply(fGal.p0, e = ei[i], sig2 = sqrt(Thet$sig2k.e), gam = Thet$gamk, p = Thet$p, tau0 = Thet$tau0) #+ .Machine$double.eps
    which(rmultinom(n=1, size=1, prob = pik) == 1)
  }
  updateCik.e <- Vectorize(updateCik.e,"i") 
  
  #--- update Si
  #-- calss id for individual i (Matrix with columns of nui and si)
  updatenui <- function(Thet, Y, Z){
    
    ki = Thet$Cik.e
    bi <- ( ((Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ck[ki]*abs(Thet$gamk[ki])*Thet$si)^2)/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])))
    ai = ((Thet$Ak[ki]*Thet$Ak[ki])/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])) + 2/sqrt(Thet$sig2k.e[ki]))
    
    #mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) 
    #val = mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    #ifelse(val < .001, rexp(n = length(muui), rate = 1/sqrt(Thet$sig2k.e[Thet$Cik.e] )), val)
  }
  
  updatesi <- function(Thet, Y, Z){   
    #-- update si
    ki = Thet$Cik.e
    sig2si <- 1 /( 1/Thet$sig2k.e[ki] +  ((Thet$Ck[ki]*Thet$gamk[ki])^2)/ (Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui))
    muui <- sig2si*(Thet$Ck[ki]*abs(Thet$gamk[ki])*(Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ak[ki]*Thet$nui))/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui)
    
    #mapply(truncnorm::rtruncnorm, n=1, a = 0.01, mean = muui, sd = sqrt(sig2si))
    mapply(truncnorm::rtruncnorm, n=1, a = 0.001, mean = muui, sd = sqrt(sig2si))
    
    
  }
  
  
  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  #a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig=5/2; b.sig = 8/2; sdsig.prop = 0.2; sdgam.prop = .3  #2.153640 1.167331
  a.sig=2.153640; b.sig = 1.167331; sdsig.prop = 0.2; sdgam.prop = .3 # 7.133445 4.303263
  a.sig=7.133445; b.sig = 4.303263; sdsig.prop = 0.2; sdgam.prop = .3 #
  updatesig_gam <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop){
    
    a0 = 0.001
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      sig_newk <- truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gam <- Vectorize(updatesig_gam,"k")
  
  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10,10,10)*1
  #varxi <- c(10,10,10)*.25
  #varxi <- c(10,10,10)*.15 ## high acceptance rate
  varxi <- c(10,10,10)*.35 ##-- High acceptance rate
  varxi <- c(10,10,10)*tunprop ##-- 
  #varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi){
    
    a0 = 0.1
    ab = 5 ## This works well
    ab = 10
    #varxi = 5
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]),Thet$gamk[k]),sigma = Sdsig.prop*varxi[k], lb = c(a0,0.95*Thet$gamL), ub = c(ab, .95*Thet$gamU)  )
      
      
      sig_newk <- Val[1] #truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] #truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu=c(sig_newk, gam_newk), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE) )
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes,"k")
  
  
  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices
  InvCov <- function(Mat){
    K <- nrow(Mat)
    J <- round(sqrt(ncol(Mat)))
    Res0 = matrix(0, nrow = K, ncol=ncol(Mat))
    for(k in 1:K){ Res0[k,] <- c(as.inverse(as.symmetric.matrix(matrix(Mat[k,],ncol=J, byrow=F)))) }
    Res0
  }
  
  
  #----------------------------------------------
  # Update things related to Wi 
  #--------------------------------------------
  #Piw
  #Cik.w
  #--- Update Pik.x  
  #alpha.W = alpha.w = 1.0
  updatePik.w <- function(Thet, alpha.W){
    Ku = nrow(Thet$Muk.w)
    nk <- numeric(Ku)
    for(k in 1:Ku){ nk[k] = sum(Thet$Cik.w == k)}
    
    #update the Pis
    alpha.wnew <- alpha.W/length(Thet$muk.w) + nk
    
    rdirichlet(n=1, alpha.wnew)
    
  }
  
  #--- Update Ci.k mixture of normals 
  dmvnormVec <- function(x, Mean, SigmaVc){
    Ku <- nrow(Mean)
    Val <- numeric(Ku)
    for(k in 1:Ku){ Val[k] <- mvtnorm::dmvnorm(x = x, mean = c(Mean[k,]), sigma = as.symmetric.matrix(matrix(SigmaVc[k,],ncol=length(x),byrow=F))) }
    Val
  }
  #dmvnormVec <- Vectorize(dmvnormVec,"k")
  #dmvnormVec(x = rep(0,6), Mn=Thet$Muk.w,SigmaVc = Thet$Sigk.w)
  
  updateCik.w <- function(i, Thet, W){ # mixture of normals 
    #Sig <- matrix(Thet$Sigk.w[id,], ncol = ncol(W), byrow = F)
    #kw = nrow(Thet$Muk.w)
    xi =  c(W[i,] - Thet$X[i,])
    #pi.k  = Thet$Pik.w*mapply(dmvnormVec, k = 1:Kw, MoreArgs = list(x = xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w))
    pi.k  = Thet$Pik.w*dmvnormVec(x=xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  
  #updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  #Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol=pn, nrow=pn));inSig0.w = as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.w <- function(Thet, W, Mu0.w, inSig0.w, Sig0.w){
    Ku = length(Thet$Pik.w)
    J = ncol(Thet$X)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol=J, nrow=J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = W - Thet$X 
    #Muk <- matrix(0, ncol=J,nrow=Ku)
    SigK <- list()
    for(k in 1:Ku){
      idk <- which(Thet$Cik.w == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow= nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.w)) ## sum of matrices inverse
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi.tld[idk,])) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
        Mu.k[k,] <- c(SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.w%*%Mu0.w))      ## some of mat
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(apply(t(Yi.tld[idk,]), 2, sum)) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
      }else{
        SigK[[k]] <- Sig0.w
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.w
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL; for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.w[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.w) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- 1.0*as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], sigma = SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.w[-Ku]) %*% SimMU.k)/Thet$Pik.w[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.w <- cmpfun(updateMuk.w)
  
  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w = pn + 2; Psi0.w = 0.5*(nu0.w-pn-1)*cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  updateSigk.w <- function(Thet, W, nu0.w, Psi0.w){
    
    Ku = nrow(Thet$Muk.w)
    m = ncol(W)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Wtl <- W - Thet$X - Thet$Muk.w[Thet$Cik.w,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.w == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.w  + nk
        Yi0 = matrix(c(Wtl[id,]), nrow= nk, byrow=F)
        Psin.w <- as.positive.definite(Psi0.w + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.w))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.w, Psi0.w))
      }
    }
    Res
  }
  updateSigk.w <- cmpfun(updateSigk.w)
  
  #----------------------------------------------
  # Update things related to M 
  #--------------------------------------------  
  
  #--- update Pik.m for epsilon_i
  #alpha.m = 1.0
  updatePik.m <- function(Thet, alpha.m){
    nk <- numeric(nrow(Thet$Muk.m))
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    
    #update the Pis
    alpha.mnew <- alpha.m/nrow(Thet$Muk.m) + nk
    
    c(rdirichlet(n=1, alpha.mnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  
  updateCik.m <- function(i, Thet, M){ # mixture of normals 
    #id <- Thet$Cik.m[i]
    #Sig <- matrix(Thet$Sigk.m[id,], ncol = ncol(M), byrow = F)
    Km = nrow(Thet$Muk.m)
    xi = M[i,] - Thet$Delta*Thet$X[i,]
    #pi.k  = Thet$Pik.m*mapply(dmvnormVec, k = 1:Km, MoreArgs = list(x = xi , Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m))
    pi.k  = Thet$Pik.m*dmvnormVec(x = xi, Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m = rep(0, pn);  Sig0.m = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.m = as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.m <- function(Thet, M, Mu0.m, inSig0.m, Sig0.m){
    #cat(dim(inSig0.m),"(--1)\n")
    Ku = length(Thet$Pik.m)
    J = ncol(M)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol = J, nrow = J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = M - Thet$X%*%diag(Thet$Betadel) 
    Muk <- matrix(0, ncol=J, nrow=Ku)
    for(k in 1:Ku){
      idk <- which(Thet$Cik.m == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow = nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.m)) ## sum of matrices inverse
        #cat(dim(SigK[[k]]),"(--2 \n")
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.m%*%Mu0.m)      ## some of mat
        #cat(dim(SigKWg),"(--3 \n")
      }else{
        SigK[[k]] <- Sig0.m
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.m
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL;  for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.m[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.m) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.m[-Ku]) %*% SimMU.k)/Thet$Pik.m[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.m <- cmpfun(updateMuk.m)
  
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m = pn + 2; Psi0.m = .5*(nu0.m-pn-1)*as.symmetric.matrix(cov(Wn))  #diag(pn)
  
  updateSigk.m <- function(Thet, M, nu0.m, Psi0.m){
    
    Ku = nrow(Thet$Muk.m)
    m = ncol(M)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Mtl <- M - Thet$X%*%diag(Thet$Betadel) - Thet$Muk.m[Thet$Cik.m,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.m == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.m  + nk
        Yi0 <- matrix(c(Mtl[id,]), nrow=nk, byrow=F)
        Psin.m <- as.positive.definite(Psi0.m + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.m))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.m, Psi0.m))
      }
    }
    Res
  }
  updateSigk.m <- cmpfun(updateSigk.m)
  
  #---- Update things related to delta
  quadFormVec <- function(i,X,sigma,Val){
    id = Val[i]
    as.symmetric.matrix(quad.form(as.inverse(as.symmetric.matrix(matrix(sigma[id,], ncol = length(X[i,]),byrow=F))), diag(c(X[i,])))) 
  }
  quadFormVec <- Vectorize(quadFormVec,"i",SIMPLIFY = F) ## Resulting in a list if precision matrices
  quadFormVec <- cmpfun(quadFormVec)
  
  quadFormVec3 <- function(i,Mi,X,sigma,Val){
    id = Val[i]
    c(quad.3form.inv(matrix(sigma[id,], ncol = length(X[i,]),byrow=F), diag(X[i,]),Mi[i,]))
  }
  quadFormVec3 <- Vectorize(quadFormVec3,"i", SIMPLIFY = F)
  quadFormVec3 <- cmpfun(quadFormVec3)
  
  #SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),   
  # prior values
  mu0.del = 1.0; #sig02.del = 100
  updateDelta <- function(Thet, M, mu0.del, sig02.del){
    n = nrow(M)
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X), Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    sig20 <- 1/( 1/sig02.del + sum(Res))
    mu <- sig20*(sum(Res3) + mu0.del/sig02.del)
    rtruncnorm(n=1, a=0, b=10, mean = mu, sd = sqrt(sig20))
    
    #c(sum(Res3), sum(Res),mu, sqrt(sig20), rtruncnorm(n=1, a=0, b=Inf, mean = mu, sd = sqrt(sig20)))
    
  }
  updateDelta <- cmpfun(updateDelta)
  
  
  # prior values
  mu0.del = rep(0,length(Thet$Betadel)); #sig02.del = 20
  Sig0.del = P.mat(length(Thet$Betadel))*.1   #Thet$siggam) # Inverse or precision covariance matrix
  
  updateFunDelta <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDelta <- cmpfun(updateFunDelta)
  
  
  SimDElVec <- function(i, Mi, X, Sigma, Val){
    id <- Val[i]
    Ot = mvtnorm::rmvnorm(n = 1,mean = Mi[i,],sigma = as.symmetric.matrix(matrix(Sigma[id,], ncol = length(X[i,]),byrow=F)))
  }
  SimDElVec <- Vectorize(SimDElVec, "i")
  
  #-- 
  DeltComp <- function(k,Xst, X){
    abs(sum(Xst[,k]*X[,k])/(sum(X[,k]^2)))
    #abs(diag(as.inverse(crossprod(X))%*%crossprod(X,Xst)))
  }
  DeltComp <- Vectorize(DeltComp,"k")
  
  updateFunDeltaSt <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Xst <- base::t(SimDElVec(1:nrow(M), Mitl, Thet$X, Thet$Sigk.m, Thet$Cik.m)) ## K by n matrix and now I get a n by K
    c(DeltComp(1:ncol(Thet$X), Xst, Thet$X))
    #c(DeltComp(Xst, Thet$X))
    
    # Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    # Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    # Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    # Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    # c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDeltaSt  <- cmpfun(updateFunDeltaSt )
  
  
  #----------------------------------------------
  # Update things related to X 
  #--------------------------------------------------  
  #--- update Pik.m for epsilon_i
  #alpha.x <- 1.0
  updatePik.x <- function(Thet, alpha.x){
    nk <- numeric(nrow(Thet$Muk.x))
    for(k in 1:nrow(Thet$Muk.x)){ nk[k] = sum(Thet$Cik.x == k)}
    
    #update the Pis
    alpha.xnew <- alpha.x/nrow(Thet$Muk.x) + nk
    
    c(rdirichlet(n=1, alpha.xnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  updateCik.x <- function(i, Thet){ # mixture of normals 
    #Kx = ncol(Thet$X) # J x J
    #id <- Thet$Cik.x[i]
    #Sig <- matrix(Thet$Sigk.x[id,], ncol = length(Thet$Gamma), byrow = F)
    Xi = Thet$X[i,]
    #pi.k  = Thet$Pik.x*mapply(dmvnormVec, k = 1:Kx, x = Xi,  Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x)
    pi.k  = Thet$Pik.x*dmvnormVec(x=Xi,Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x = numeric(pn); Sig0.x = .5*diag(pn); inSig0.x = 2*diag(pn)#as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x = rep(0, pn);  Sig0.x = .25*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w)+colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.x = as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.x <- function(Thet, Mu0.x, inSig0.x, Sig0.x){
    K = length(Thet$Pik.x)
    J = ncol(Thet$X)
    Res0 <- matrix(0, nrow = K, ncol = J)
    Mu.k <- matrix(0, nrow = K, ncol = J) # K x J
    #SigKWg <- matrix(0, ncol = ncol(Y),nrow=ncol(Y)) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = Thet$X 
    Muk <- matrix(0, ncol = J, nrow = K)
    for(k in 1:K){
      idk <- which(Thet$Cik.x == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 <- matrix(c(Yi.tld[idk,]), nrow=nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[k,], ncol=J, byrow=F)))
        SigK <- as.inverse(as.symmetric.matrix((nk*TpSi + inSig0.x))) ## sum of matrices inverse
        Muk <- SigK%*%(crossprod(TpSi, c(colSums(Yi0))) + inSig0.x%*%Mu0.x)      ## some of mat
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Muk, sigma = SigK))
      }else{
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Mu0.x, sigma = Sig0.x ))
      }
    }
    #--- we impose some constrainsts on the mean
    Mu.k
    
  }
  updateMuk.x <- cmpfun(updateMuk.x)
  
  nu0.x <- pn + 2 ; Psi0.x <- .5*diag(pn) #as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2 ; Psi0.x <- .5*(nu0.x-pn-1)*cov(.5*(Wn+Mn)) #as.symmetric.matrix(.25*cov(Wn))
  updateSigk.x <- function(Thet, nu0.x, Psi0.x){
    
    Ku = nrow(Thet$Muk.x)
    m = ncol(Thet$Muk.x)
    Xtl <- Thet$X - Thet$Muk.x[Thet$Cik.x,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.x == i)
      nk = length(id)
      if(nk > 0){
        nux <- nu0.x  + nk
        Yi0 <- matrix(c(Xtl[id,]), nrow=nk, byrow=F)
        Psin.x <- Psi0.x + crossprod(Yi0)
        Res[i, ] <- c(rinvwishart(nux, Psin.x))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.x, Psi0.x))
      }
    }
    Res
  }
  updateSigk.x <- cmpfun(updateSigk.x)
  # 
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn))) 
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn))) 
  
  A <- min(c(Wn, Mn)) - .1*diff(range(c(Wn, Mn))) 
  B <- max(c(Wn, Mn)) + .1*diff(range(c(Wn, Mn))) 
  
  udapateXi <- function(i, Thet, Y, W, M, Z, A=A, B=B){
    ie =Thet$Cik.e[i]
    iw = Thet$Cik.w[i]
    im = Thet$Cik.m[i]
    ix = Thet$Cik.x[i]
    J = ncol(Thet$X)
    
    Zmod <- Z #cbind(1,Z)
    Ytl = (Y - Zmod%*%Thet$BetaZ - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Wtl = W - Thet$Muk.w[Thet$Cik.w,]
    Mtl = (M - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
    Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
    Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
    
    #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
    Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
    
    Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))) 
    
    #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi <- cmpfun(udapateXi)
  
  
  udapateXi2A <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2A <- cmpfun(udapateXi2A)
  
  ## This approach of simulating Xi only uses W, M, and X
  
  udapateXi2 <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #Mu <- c(Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2 <- cmpfun(udapateXi2)
  
  
  udapateXi2V <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    #Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    #GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- (Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi2V <- Vectorize(udapateXi2V,"i")
  udapateXi2V <- cmpfun(udapateXi2V)
  
  #--- Write Rcpp function to speed things up
  {
    #
    
    
  }
  
  
  
  #Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)
  
  ##----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  #Nsim <- 5000
  #Nsim = 5000 
  MCMCSample <- list()
  
  #-- for X
  MCMCSample$X <- matrix(0,nrow = Nsim,ncol=length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined) 
  MCMCSample$Muk.x <- matrix(0,nrow = Nsim, ncol = Kx*pn)
  MCMCSample$Sigk.x <- matrix(0,nrow = Nsim,ncol=Kx*pn*pn)
  MCMCSample$Pik.x <- matrix(0,nrow = Nsim,ncol=Kx)
  MCMCSample$Cik.x <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for W
  MCMCSample$Muk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn)
  MCMCSample$Sigk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn*pn)
  MCMCSample$Pik.w <- matrix(0,nrow = Nsim,ncol=Kw)
  MCMCSample$Cik.w <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for M
  MCMCSample$Muk.m <- matrix(0,nrow = Nsim,ncol=Km*pn)
  MCMCSample$Sigk.m <- matrix(0,nrow = Nsim,ncol=Km*pn*pn)
  MCMCSample$Pik.m <- matrix(0,nrow = Nsim,ncol=Km)
  MCMCSample$Cik.m <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow=Nsim, ncol=length(Thet$Betadel))
  
  # For Y
  MCMCSample$gamk <- matrix(0,nrow = Nsim, ncol=Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Pik.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Cik.e <- matrix(0,nrow = Nsim, ncol=nrow(Wn))
  
  MCMCSample$Gamma <- matrix(0,nrow = Nsim, ncol=length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)
  
  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  # 
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  # 
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  # 
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  #Accept <- 
  # 
  MCMCSample$AcceptX <- matrix(0,nrow = Nsim, ncol=Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  #stop()
  #------------------------------------------------------------------
  #Nsim = 5000 
  #Nsim = 1000
  #Nsim = 3000
  #library(profvis)
  #profvis(
  #system.time(
  #cat("\n #--- MCMC is starting ----> \n")
  Thet$X <- (apply(W,c(1,2), mean)%*%bs2)/length(a)
  for(iter in 1:Nsim){
    if(iter %% 500 == 0){cat("iter=",iter,"\n")}
    #----------------------------------------------
    # Update things related to X 
    #--------------------------------------------  
    # #--- update Pik.m for epsilon_i
    # Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    # 
    # #--- update Cik.m for epsilon_i based on mixture of normals
    # Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    # #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x)) 
    # 
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    # 
    # Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    # Thet$InvSigk.x <- InvCov(Thet$Sigk.x)
    
    #Ytd = Y - Zmod%*%Thet$BetaZ
    #if(iter%%10 == 0){
    #Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B) # Xn #
    #Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    #Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    #Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w, 
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)
    
    #Thet$X <- Xn #UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,]) 
    
    #}
    #dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))
    
    #Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    #Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    #Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    #Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w, 
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J))) 
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Kw)
    # for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    # #update the Pis
    # alpha.wnew <- alpha.W/Kw + nk
    # Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    # 
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    # Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    # Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    # Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Km)
    # for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    # #update the Pis
    # alpha.mnew <- alpha.m/Kw + nk
    # Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    # 
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    # Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    # #Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))  
    # #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    # Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    # Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # #if(iter%%50 == 0){
    # Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    # MCMCSample$Betdel[iter,] <- Thet$Betadel
    # #}
    #---------------------------------------------
    #cat("#--- update things related to Y or epsilon ")
    #--------------------------------------------- 
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for(k in 1:length(Thet$sig2k.e)){ nk[k] = sum(Thet$Cik.e == k)}
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n=1, alpha.enew))
    
    #update the Cik.e
    ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
    #Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) #rep(1,n) #
    #Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    #update nui  and si
    #Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    #nuisi <- updatenuisi(Thet, Y, Zmod)
    #Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    #update Sigk.e
    #Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    #update sigk and gamk
    #--- update jointly sig2ke and gamke
    #Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi)
    Thet$sig2k.e <- Temp[1,];  Thet$gamk <- Temp[2,]; Thet$p <- Temp[3,]; MCMCSample$AcceptX[iter,] <- Temp[4,]; Thet$Ak <- Temp[5,]; Thet$Bk <- Temp[6,]; Thet$Ck <- Temp[7,];
    
    #update Gamma
    #Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    #Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <-  updateSigGam(Thet, al0, bet0, order = 2)# sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#
    
    #Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    # 
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    # 
    # #--- upodate beta_{z,m}
    # #--  Prior 
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    # 
    # #--- update X_i 
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX) 
    # 
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]); 
    # 
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb 
    
    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x))  ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise
    
    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w
    
    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    #MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e
    
    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  } 
  
  #)
  #)
  
  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample$Xpred <- Thet$X
  
  MCMCSample
}



#-- Using replicated W not M
BQBayes.ME_SoFRFuncSimFxFSMI <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0 = 0.5,X, g0,tunprop){
  
  #------------------------------------
  #------------
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
  met <- c("f1","f2","f3","f4","f5")
  
  
  func <- function(Pik, n=1, size=1){
    which.max(rmultinom(n=1,size=1,prob = Pik))
  }
  funcCmp <- cmpfun(func)
  
  F2 <- function(Pik, n=1, size=1){
    
    apply(Pik,1,funcCmp)
  }
  F2Cmp <- cmpfun(F2)
  
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
  
  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  #pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  #pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  n = length(Y) 
  Zmod = cbind(1,Z)
  pn = c(5, 6, 20, 20)[which(n == c(100, 200,500,1000))]
  alpha.e = alpx
  alpha.w = alpha.W = alpx
  alpha.x = alpha.X = alpx
  alpha.m = alpx
  pn = pn #min(15, ceiling(length(Y)^{1/3.8})+4)
  
  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt #c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) # 
  Mn_Mod <- M%*%diag(1/Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20,"knots"), 1) ))
  Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a) 
  Wn <- (W[,,1]%*%bs2)/length(a)
  Xn <- (X%*%bs2)/length(a)
  
  cat("\n #--- Initial values...\n")  
  P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D #+ diag(rep(1e-3, K))
    return(P)
  }
  
  
  InitialValues <- function(Y, W, M, Z, df0=3, pn, Gx, a){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- Z #cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    lmFit <- lm(Y ~ -1 + Zmod + W_i) 
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- residuals.lm(lmFit)
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  InitialValuesQR <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z+Wn)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    #plot(a, bs2%*%lmFit$coefficients[-c(1:3)], type="l")
    
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    Tp0 = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd)), upper=c(30, 0.95*max(bnd) ))
    Res$Hess <- Tp0$hessian
    
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  #Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, apply(W,c(1,2), mean), M, Z, df0=3, pn, Gx=1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  #Kvalues
  #Kc <- 5
  
  Kx = Thet0$Kx               ## number of X  cluster
  Ke = Thet0$Ke               ## number of ei cluster
  Km = Thet0$Km               ## number of ei cluster
  Kw = Thet0$Kw              ## number of omega_i cluster 
  
  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta   #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n=pn) #as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ #rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del#0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  #Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1]); Thet$gamU <- max(bnd[2])
  
  #bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) #c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
  
  Thet$Cik.x =  sample.int(Kx,size=n, replace = T) #Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e =  sample.int(Ke,size=n, replace = T) #Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w = sample.int(Kw,size=n, replace = T) #Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m = sample.int(Km,size=n, replace = T) #Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km
  
  
  Thet$Pik.x = rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)#Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e = rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)#Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w = rdirch(n=1,alpha = rep(alpha.e, Kw)/Kw) #Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m = rdirch(n=1,alpha = rep(alpha.m, Km)/Km)#Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw
  
  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw
  
  Thet$sig2k.e  = nimble::rinvgamma(n=Ke, shape=5/2,scale=8/2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n=Ke, min = .95*Thet$gamL, max=.95*Thet$gamU)
  Thet$p <- 1*(Thet$gamk < 0) + (Thet$tau0 - 1*(Thet$gamk < 0))/(2*pnorm(-abs(Thet$gamk))*exp(0.5*Thet$gamk*Thet$gamk))
  Thet$Ak <- (1-2*Thet$p)/(Thet$p*(1-Thet$p)); Thet$Bk <- 2/(Thet$p*(1-Thet$p)); Thet$Ck <- 1/( 1*(Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean=0, sd=sqrt(Thet$sig2k.e[1])) ; Thet$nui <- rexp(n=n, rate = 1/Thet$sig2k.e[1]) 
  #Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix
  
  Thet$Sigk.w <- repmat(c(.5*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw*pn), nrow = Kw)          #- vectors of length Kw and Kw respectively
  
  Thet$Sigk.x <- repmat(c(.5*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$Muk.x  <- matrix(rnorm(n=Kx*pn), nrow=Kx)   #- vectors of length Kx and Kx respectively
  
  Thet$Sigk.m <- repmat(c(.5*diag(pn)), Km, 1)     #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2*diag(pn)), Km, 1)
  Thet$Muk.m  <- matrix(rnorm(n=Km*pn), nrow = Km)   #- vectors of length Kx and Kx respectively
  
  A <- min(c(c(Wn), c(Mn))) - .5*diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5*diff(range(c(Wn, Mn)))
  #Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  #matrix(rnorm(n = n*pn), ncol=pn)
  #Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5*(Mn+Wn) #Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))
  
  
  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
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
  
  
  #method = "L-BFGS-B"
  #method = "BFGS"
  
  #Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  #Res
  
  #Res$hessian
  #Res$hessian/n
  #solve(Res$hessian/n)
  
  #LogPlostGalLik(c(.05,), c(ei), Thet)
  
  #-------- Update Beta (vector of length pn) 
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #   
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma 
  #   
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #   
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #   
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #   
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)
  
  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together) 
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 = ncol(Z) +1
  Pgam <- P.mat(pn)*tauval
  Sig0.gam0 = diag(pn0); Mu0.gam0 = numeric(pn0+pn);
  #InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))
  
  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.gam0%*%Mu0.gam0))
    
    #c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb =c(rep(-Inf, ncol(Zmod)),rep(-g0,length(Mu.gam)-ncol(Zmod))),
                                ub = c(rep(Inf, ncol(Zmod)),rep(g0,length(Mu.gam)-ncol(Zmod)))   ))
    
  }
  updateGam <- cmpfun(updateGam)
  
  #siggam ~ IG(al0, bet0)
  al0 = 0.001
  bet0 = .005
  al0 = 2.5978252 #1
  bet0 = 0.4842963 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    val <- nimble::rinvgamma(n = 1, shape = al, scale = bet)
    ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  
  al0 = 0.001 #1 #1
  bet0 = 0.001 #0.005 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    #rgamma(n = 1, shape = al, scale = bet)
    nimble::rinvgamma(n = 1, shape = al, scale = bet)
    #ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e 
  #alpha.e = 1.0
  updatePik.e <- function(Thet, alpha.e){
    
    nk <- numeric(length(Thet$Cik.e))
    for(i in 1:length(Thet$Cik.e)){ nk[i] = sum(Thet$Cik.e == i)}
    
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$Cik.e) + nk
    
    rdirichlet(n=1, alpha.enew)
    
  }
  
  #-------- Update Cik.e  
  
  #ei <- Y - Z%*%Thet$Betaz - Thet$X%*%Thet$Gamma
  updateCik.e <- function(i, Thet,ei){
    #Y - Thet$C*abs(Thet$gam)*Thet$si - Thet$A*Thet$nui
    
    #  Yi.tld <- Y[i] - c(crossprod(Thet$Gamma, Thet$X[i,]))
    #pi.k  <- Thet$Pik.e*mapply(dnorm, x = Yi.tld, mean=Thet$muk.e, sd = sqrt(Thet$sig2k.e)) + .Machine$double.eps
    pik <- Thet$Pik.e*mapply(fGal.p0, e = ei[i], sig2 = sqrt(Thet$sig2k.e), gam = Thet$gamk, p = Thet$p, tau0 = Thet$tau0) #+ .Machine$double.eps
    which(rmultinom(n=1, size=1, prob = pik) == 1)
  }
  updateCik.e <- Vectorize(updateCik.e,"i") 
  
  #--- update Si
  #-- calss id for individual i (Matrix with columns of nui and si)
  updatenui <- function(Thet, Y, Z){
    
    ki = Thet$Cik.e
    bi <- ( ((Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ck[ki]*abs(Thet$gamk[ki])*Thet$si)^2)/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])))
    ai = ((Thet$Ak[ki]*Thet$Ak[ki])/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])) + 2/sqrt(Thet$sig2k.e[ki]))
    
    #mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) 
    #val = mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    #ifelse(val < .001, rexp(n = length(muui), rate = 1/sqrt(Thet$sig2k.e[Thet$Cik.e] )), val)
  }
  
  updatesi <- function(Thet, Y, Z){   
    #-- update si
    ki = Thet$Cik.e
    sig2si <- 1 /( 1/Thet$sig2k.e[ki] +  ((Thet$Ck[ki]*Thet$gamk[ki])^2)/ (Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui))
    muui <- sig2si*(Thet$Ck[ki]*abs(Thet$gamk[ki])*(Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ak[ki]*Thet$nui))/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui)
    
    #mapply(truncnorm::rtruncnorm, n=1, a = 0.01, mean = muui, sd = sqrt(sig2si))
    mapply(truncnorm::rtruncnorm, n=1, a = 0.001, mean = muui, sd = sqrt(sig2si))
    
    
  }
  
  
  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  #a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig=5/2; b.sig = 8/2; sdsig.prop = 0.2; sdgam.prop = .3  #2.153640 1.167331
  a.sig=2.153640; b.sig = 1.167331; sdsig.prop = 0.2; sdgam.prop = .3 # 7.133445 4.303263
  a.sig=7.133445; b.sig = 4.303263; sdsig.prop = 0.2; sdgam.prop = .3 #
  updatesig_gam <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop){
    
    a0 = 0.001
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      sig_newk <- truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gam <- Vectorize(updatesig_gam,"k")
  
  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10,10,10)*1
  #varxi <- c(10,10,10)*.25
  #varxi <- c(10,10,10)*.15 ## high acceptance rate
  varxi <- c(10,10,10)*.35 ##-- High acceptance rate
  varxi <- c(10,10,10)*tunprop ##-- 
  #varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi){
    
    a0 = 0.1
    ab = 5 ## This works well
    ab = 10
    #varxi = 5
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]),Thet$gamk[k]),sigma = Sdsig.prop*varxi[k], lb = c(a0,0.95*Thet$gamL), ub = c(ab, .95*Thet$gamU)  )
      
      
      sig_newk <- Val[1] #truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] #truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu=c(sig_newk, gam_newk), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE) )
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes,"k")
  
  
  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices
  InvCov <- function(Mat){
    K <- nrow(Mat)
    J <- round(sqrt(ncol(Mat)))
    Res0 = matrix(0, nrow = K, ncol=ncol(Mat))
    for(k in 1:K){ Res0[k,] <- c(as.inverse(as.symmetric.matrix(matrix(Mat[k,],ncol=J, byrow=F)))) }
    Res0
  }
  
  
  #----------------------------------------------
  # Update things related to Wi 
  #--------------------------------------------
  #Piw
  #Cik.w
  #--- Update Pik.x  
  #alpha.W = alpha.w = 1.0
  updatePik.w <- function(Thet, alpha.W){
    Ku = nrow(Thet$Muk.w)
    nk <- numeric(Ku)
    for(k in 1:Ku){ nk[k] = sum(Thet$Cik.w == k)}
    
    #update the Pis
    alpha.wnew <- alpha.W/length(Thet$muk.w) + nk
    
    rdirichlet(n=1, alpha.wnew)
    
  }
  
  #--- Update Ci.k mixture of normals 
  dmvnormVec <- function(x, Mean, SigmaVc){
    Ku <- nrow(Mean)
    Val <- numeric(Ku)
    for(k in 1:Ku){ Val[k] <- mvtnorm::dmvnorm(x = x, mean = c(Mean[k,]), sigma = as.symmetric.matrix(matrix(SigmaVc[k,],ncol=length(x),byrow=F))) }
    Val
  }
  #dmvnormVec <- Vectorize(dmvnormVec,"k")
  #dmvnormVec(x = rep(0,6), Mn=Thet$Muk.w,SigmaVc = Thet$Sigk.w)
  
  updateCik.w <- function(i, Thet, W){ # mixture of normals 
    #Sig <- matrix(Thet$Sigk.w[id,], ncol = ncol(W), byrow = F)
    #kw = nrow(Thet$Muk.w)
    xi =  c(W[i,] - Thet$X[i,])
    #pi.k  = Thet$Pik.w*mapply(dmvnormVec, k = 1:Kw, MoreArgs = list(x = xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w))
    pi.k  = Thet$Pik.w*dmvnormVec(x=xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  
  #updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  #Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol=pn, nrow=pn));inSig0.w = as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.w <- function(Thet, W, Mu0.w, inSig0.w, Sig0.w){
    Ku = length(Thet$Pik.w)
    J = ncol(Thet$X)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol=J, nrow=J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = W - Thet$X 
    #Muk <- matrix(0, ncol=J,nrow=Ku)
    SigK <- list()
    for(k in 1:Ku){
      idk <- which(Thet$Cik.w == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow= nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.w)) ## sum of matrices inverse
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi.tld[idk,])) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
        Mu.k[k,] <- c(SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.w%*%Mu0.w))      ## some of mat
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(apply(t(Yi.tld[idk,]), 2, sum)) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
      }else{
        SigK[[k]] <- Sig0.w
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.w
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL; for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.w[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.w) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- 1.0*as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], sigma = SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.w[-Ku]) %*% SimMU.k)/Thet$Pik.w[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.w <- cmpfun(updateMuk.w)
  
  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w = pn + 2; Psi0.w = 0.5*(nu0.w-pn-1)*cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  updateSigk.w <- function(Thet, W, nu0.w, Psi0.w){
    
    Ku = nrow(Thet$Muk.w)
    m = ncol(W)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Wtl <- W - Thet$X - Thet$Muk.w[Thet$Cik.w,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.w == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.w  + nk
        Yi0 = matrix(c(Wtl[id,]), nrow= nk, byrow=F)
        Psin.w <- as.positive.definite(Psi0.w + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.w))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.w, Psi0.w))
      }
    }
    Res
  }
  updateSigk.w <- cmpfun(updateSigk.w)
  
  #----------------------------------------------
  # Update things related to M 
  #--------------------------------------------  
  
  #--- update Pik.m for epsilon_i
  #alpha.m = 1.0
  updatePik.m <- function(Thet, alpha.m){
    nk <- numeric(nrow(Thet$Muk.m))
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    
    #update the Pis
    alpha.mnew <- alpha.m/nrow(Thet$Muk.m) + nk
    
    c(rdirichlet(n=1, alpha.mnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  
  updateCik.m <- function(i, Thet, M){ # mixture of normals 
    #id <- Thet$Cik.m[i]
    #Sig <- matrix(Thet$Sigk.m[id,], ncol = ncol(M), byrow = F)
    Km = nrow(Thet$Muk.m)
    xi = M[i,] - Thet$Delta*Thet$X[i,]
    #pi.k  = Thet$Pik.m*mapply(dmvnormVec, k = 1:Km, MoreArgs = list(x = xi , Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m))
    pi.k  = Thet$Pik.m*dmvnormVec(x = xi, Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m = rep(0, pn);  Sig0.m = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.m = as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.m <- function(Thet, M, Mu0.m, inSig0.m, Sig0.m){
    #cat(dim(inSig0.m),"(--1)\n")
    Ku = length(Thet$Pik.m)
    J = ncol(M)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol = J, nrow = J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = M - Thet$X%*%diag(Thet$Betadel) 
    Muk <- matrix(0, ncol=J, nrow=Ku)
    for(k in 1:Ku){
      idk <- which(Thet$Cik.m == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow = nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.m)) ## sum of matrices inverse
        #cat(dim(SigK[[k]]),"(--2 \n")
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.m%*%Mu0.m)      ## some of mat
        #cat(dim(SigKWg),"(--3 \n")
      }else{
        SigK[[k]] <- Sig0.m
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.m
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL;  for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.m[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.m) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.m[-Ku]) %*% SimMU.k)/Thet$Pik.m[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.m <- cmpfun(updateMuk.m)
  
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m = pn + 2; Psi0.m = .5*(nu0.m-pn-1)*as.symmetric.matrix(cov(Wn))  #diag(pn)
  
  updateSigk.m <- function(Thet, M, nu0.m, Psi0.m){
    
    Ku = nrow(Thet$Muk.m)
    m = ncol(M)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Mtl <- M - Thet$X%*%diag(Thet$Betadel) - Thet$Muk.m[Thet$Cik.m,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.m == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.m  + nk
        Yi0 <- matrix(c(Mtl[id,]), nrow=nk, byrow=F)
        Psin.m <- as.positive.definite(Psi0.m + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.m))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.m, Psi0.m))
      }
    }
    Res
  }
  updateSigk.m <- cmpfun(updateSigk.m)
  
  #---- Update things related to delta
  quadFormVec <- function(i,X,sigma,Val){
    id = Val[i]
    as.symmetric.matrix(quad.form(as.inverse(as.symmetric.matrix(matrix(sigma[id,], ncol = length(X[i,]),byrow=F))), diag(c(X[i,])))) 
  }
  quadFormVec <- Vectorize(quadFormVec,"i",SIMPLIFY = F) ## Resulting in a list if precision matrices
  quadFormVec <- cmpfun(quadFormVec)
  
  quadFormVec3 <- function(i,Mi,X,sigma,Val){
    id = Val[i]
    c(quad.3form.inv(matrix(sigma[id,], ncol = length(X[i,]),byrow=F), diag(X[i,]),Mi[i,]))
  }
  quadFormVec3 <- Vectorize(quadFormVec3,"i", SIMPLIFY = F)
  quadFormVec3 <- cmpfun(quadFormVec3)
  
  #SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),   
  # prior values
  mu0.del = 1.0; #sig02.del = 100
  updateDelta <- function(Thet, M, mu0.del, sig02.del){
    n = nrow(M)
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X), Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    sig20 <- 1/( 1/sig02.del + sum(Res))
    mu <- sig20*(sum(Res3) + mu0.del/sig02.del)
    rtruncnorm(n=1, a=0, b=10, mean = mu, sd = sqrt(sig20))
    
    #c(sum(Res3), sum(Res),mu, sqrt(sig20), rtruncnorm(n=1, a=0, b=Inf, mean = mu, sd = sqrt(sig20)))
    
  }
  updateDelta <- cmpfun(updateDelta)
  
  
  # prior values
  mu0.del = rep(0,length(Thet$Betadel)); #sig02.del = 20
  Sig0.del = P.mat(length(Thet$Betadel))*.1   #Thet$siggam) # Inverse or precision covariance matrix
  
  updateFunDelta <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDelta <- cmpfun(updateFunDelta)
  
  
  SimDElVec <- function(i, Mi, X, Sigma, Val){
    id <- Val[i]
    Ot = mvtnorm::rmvnorm(n = 1,mean = Mi[i,],sigma = as.symmetric.matrix(matrix(Sigma[id,], ncol = length(X[i,]),byrow=F)))
  }
  SimDElVec <- Vectorize(SimDElVec, "i")
  
  #-- 
  DeltComp <- function(k,Xst, X){
    abs(sum(Xst[,k]*X[,k])/(sum(X[,k]^2)))
    #abs(diag(as.inverse(crossprod(X))%*%crossprod(X,Xst)))
  }
  DeltComp <- Vectorize(DeltComp,"k")
  
  updateFunDeltaSt <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Xst <- base::t(SimDElVec(1:nrow(M), Mitl, Thet$X, Thet$Sigk.m, Thet$Cik.m)) ## K by n matrix and now I get a n by K
    c(DeltComp(1:ncol(Thet$X), Xst, Thet$X))
    #c(DeltComp(Xst, Thet$X))
    
    # Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    # Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    # Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    # Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    # c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDeltaSt  <- cmpfun(updateFunDeltaSt )
  
  
  #----------------------------------------------
  # Update things related to X 
  #--------------------------------------------------  
  #--- update Pik.m for epsilon_i
  #alpha.x <- 1.0
  updatePik.x <- function(Thet, alpha.x){
    nk <- numeric(nrow(Thet$Muk.x))
    for(k in 1:nrow(Thet$Muk.x)){ nk[k] = sum(Thet$Cik.x == k)}
    
    #update the Pis
    alpha.xnew <- alpha.x/nrow(Thet$Muk.x) + nk
    
    c(rdirichlet(n=1, alpha.xnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  updateCik.x <- function(i, Thet){ # mixture of normals 
    #Kx = ncol(Thet$X) # J x J
    #id <- Thet$Cik.x[i]
    #Sig <- matrix(Thet$Sigk.x[id,], ncol = length(Thet$Gamma), byrow = F)
    Xi = Thet$X[i,]
    #pi.k  = Thet$Pik.x*mapply(dmvnormVec, k = 1:Kx, x = Xi,  Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x)
    pi.k  = Thet$Pik.x*dmvnormVec(x=Xi,Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x = numeric(pn); Sig0.x = .5*diag(pn); inSig0.x = 2*diag(pn)#as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x = rep(0, pn);  Sig0.x = .25*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w)+colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.x = as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.x <- function(Thet, Mu0.x, inSig0.x, Sig0.x){
    K = length(Thet$Pik.x)
    J = ncol(Thet$X)
    Res0 <- matrix(0, nrow = K, ncol = J)
    Mu.k <- matrix(0, nrow = K, ncol = J) # K x J
    #SigKWg <- matrix(0, ncol = ncol(Y),nrow=ncol(Y)) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = Thet$X 
    Muk <- matrix(0, ncol = J, nrow = K)
    for(k in 1:K){
      idk <- which(Thet$Cik.x == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 <- matrix(c(Yi.tld[idk,]), nrow=nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[k,], ncol=J, byrow=F)))
        SigK <- as.inverse(as.symmetric.matrix((nk*TpSi + inSig0.x))) ## sum of matrices inverse
        Muk <- SigK%*%(crossprod(TpSi, c(colSums(Yi0))) + inSig0.x%*%Mu0.x)      ## some of mat
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Muk, sigma = SigK))
      }else{
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Mu0.x, sigma = Sig0.x ))
      }
    }
    #--- we impose some constrainsts on the mean
    Mu.k
    
  }
  updateMuk.x <- cmpfun(updateMuk.x)
  
  nu0.x <- pn + 2 ; Psi0.x <- .5*diag(pn) #as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2 ; Psi0.x <- .5*(nu0.x-pn-1)*cov(.5*(Wn+Mn)) #as.symmetric.matrix(.25*cov(Wn))
  updateSigk.x <- function(Thet, nu0.x, Psi0.x){
    
    Ku = nrow(Thet$Muk.x)
    m = ncol(Thet$Muk.x)
    Xtl <- Thet$X - Thet$Muk.x[Thet$Cik.x,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.x == i)
      nk = length(id)
      if(nk > 0){
        nux <- nu0.x  + nk
        Yi0 <- matrix(c(Xtl[id,]), nrow=nk, byrow=F)
        Psin.x <- Psi0.x + crossprod(Yi0)
        Res[i, ] <- c(rinvwishart(nux, Psin.x))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.x, Psi0.x))
      }
    }
    Res
  }
  updateSigk.x <- cmpfun(updateSigk.x)
  # 
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn))) 
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn))) 
  
  A <- min(c(Wn, Mn)) - .1*diff(range(c(Wn, Mn))) 
  B <- max(c(Wn, Mn)) + .1*diff(range(c(Wn, Mn))) 
  
  udapateXi <- function(i, Thet, Y, W, M, Z, A=A, B=B){
    ie =Thet$Cik.e[i]
    iw = Thet$Cik.w[i]
    im = Thet$Cik.m[i]
    ix = Thet$Cik.x[i]
    J = ncol(Thet$X)
    
    Zmod <- Z #cbind(1,Z)
    Ytl = (Y - Zmod%*%Thet$BetaZ - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Wtl = W - Thet$Muk.w[Thet$Cik.w,]
    Mtl = (M - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
    Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
    Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
    
    #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
    Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
    
    Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))) 
    
    #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi <- cmpfun(udapateXi)
  
  
  udapateXi2A <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2A <- cmpfun(udapateXi2A)
  
  ## This approach of simulating Xi only uses W, M, and X
  
  udapateXi2 <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #Mu <- c(Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2 <- cmpfun(udapateXi2)
  
  
  udapateXi2V <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    #Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    #GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- (Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi2V <- Vectorize(udapateXi2V,"i")
  udapateXi2V <- cmpfun(udapateXi2V)
  
  #--- Write Rcpp function to speed things up
  {
    #
    
    
  }
  
  
  
  #Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)
  
  ##----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  #Nsim <- 5000
  #Nsim = 5000 
  MCMCSample <- list()
  
  #-- for X
  MCMCSample$X <- matrix(0,nrow = Nsim,ncol=length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined) 
  MCMCSample$Muk.x <- matrix(0,nrow = Nsim, ncol = Kx*pn)
  MCMCSample$Sigk.x <- matrix(0,nrow = Nsim,ncol=Kx*pn*pn)
  MCMCSample$Pik.x <- matrix(0,nrow = Nsim,ncol=Kx)
  MCMCSample$Cik.x <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for W
  MCMCSample$Muk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn)
  MCMCSample$Sigk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn*pn)
  MCMCSample$Pik.w <- matrix(0,nrow = Nsim,ncol=Kw)
  MCMCSample$Cik.w <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  
  #-- for M
  MCMCSample$Muk.m <- matrix(0,nrow = Nsim,ncol=Km*pn)
  MCMCSample$Sigk.m <- matrix(0,nrow = Nsim,ncol=Km*pn*pn)
  MCMCSample$Pik.m <- matrix(0,nrow = Nsim,ncol=Km)
  MCMCSample$Cik.m <- matrix(0,nrow = Nsim,ncol=nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow=Nsim, ncol=length(Thet$Betadel))
  
  # For Y
  MCMCSample$gamk <- matrix(0,nrow = Nsim, ncol=Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Pik.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Cik.e <- matrix(0,nrow = Nsim, ncol=nrow(Wn))
  
  MCMCSample$Gamma <- matrix(0,nrow = Nsim, ncol=length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)
  
  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  # 
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  # 
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  # 
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  #Accept <- 
  # 
  MCMCSample$AcceptX <- matrix(0,nrow = Nsim, ncol=Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  #stop()
  #------------------------------------------------------------------
  #Nsim = 5000 
  #Nsim = 1000
  #Nsim = 3000
  #library(profvis)
  #profvis(
  #system.time(
  #cat("\n #--- MCMC is starting ----> \n")
  #Thet$X <- (as.matrix(FUI(model="gaussian",smooth=T, W,silent = FALSE)) %*%bs2)/length(a)
  Thet$X <- (FSMI(model="gaussian", cov.model ="us", W)%*%bs2)/length(a)
  for(iter in 1:Nsim){
    if(iter %% 500 == 0){cat("iter=",iter,"\n")}
    #----------------------------------------------
    # Update things related to X 
    #--------------------------------------------  
    # #--- update Pik.m for epsilon_i
    # Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    # 
    # #--- update Cik.m for epsilon_i based on mixture of normals
    # Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    # #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x)) 
    # 
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    # 
    # Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    # Thet$InvSigk.x <- InvCov(Thet$Sigk.x)
    
    #Ytd = Y - Zmod%*%Thet$BetaZ
    #if(iter%%10 == 0){
    #Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B) # Xn #
    #Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    #Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    #Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w, 
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)
    
    #Thet$X <- Xn #UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,]) 
    
    #}
    #dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))
    
    #Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    #Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    #Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    #Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w, 
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J))) 
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Kw)
    # for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    # #update the Pis
    # alpha.wnew <- alpha.W/Kw + nk
    # Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    # 
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    # Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    # Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    # Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Km)
    # for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    # #update the Pis
    # alpha.mnew <- alpha.m/Kw + nk
    # Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    # 
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    # Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    # #Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))  
    # #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    # Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    # Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # #if(iter%%50 == 0){
    # Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    # MCMCSample$Betdel[iter,] <- Thet$Betadel
    # #}
    #---------------------------------------------
    #cat("#--- update things related to Y or epsilon ")
    #--------------------------------------------- 
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for(k in 1:length(Thet$sig2k.e)){ nk[k] = sum(Thet$Cik.e == k)}
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n=1, alpha.enew))
    
    #update the Cik.e
    ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
    #Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) #rep(1,n) #
    #Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    #update nui  and si
    #Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    #nuisi <- updatenuisi(Thet, Y, Zmod)
    #Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    #update Sigk.e
    #Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    #update sigk and gamk
    #--- update jointly sig2ke and gamke
    #Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi)
    Thet$sig2k.e <- Temp[1,];  Thet$gamk <- Temp[2,]; Thet$p <- Temp[3,]; MCMCSample$AcceptX[iter,] <- Temp[4,]; Thet$Ak <- Temp[5,]; Thet$Bk <- Temp[6,]; Thet$Ck <- Temp[7,];
    
    #update Gamma
    #Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    #Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <-  updateSigGam(Thet, al0, bet0, order = 2)# sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#
    
    #Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    # 
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    # 
    # #--- upodate beta_{z,m}
    # #--  Prior 
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    # 
    # #--- update X_i 
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX) 
    # 
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]); 
    # 
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb 
    
    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x))  ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise
    
    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w
    
    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    #MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e
    
    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  } 
  
  #)
  #)
  
  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample$Xpred <- Thet$X
  
  MCMCSample
}

#----- Case of replicated days
#sig02.del = 0.1; tauval = .1;  Delt= rep(1, ncol(M)); alpx=1; tau0 = 0.5
BQBayes.ME_SoFRFuncSimWrep <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0 = 0.5, X, g0,tunprop){
  
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
  met <- c("f1","f2","f3","f4","f5")
  
  
  func <- function(Pik, Pi0, n=1, size=1){
    which.max(rmultinom(n=1,size=1,prob = exp(Pik + abs(max(Pik)))*Pi0))
    #which.max(rmultinom(n=1,size=1,prob = Pik))
  }
  funcCmp <- cmpfun(func)
  
  F2 <- function(Pik, Pi0, n=1, size=1){
    
    apply(Pik,1,funcCmp,Pi0 = Pi0)
  }
  F2Cmp <- cmpfun(F2)
  
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
  #--------------------------------------------
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
  
  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  #pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  #pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  n = length(Y) 
  Zmod = cbind(1,Z)
  pn = c(5, 6, 20, 20)[which(n == c(100, 200,500,1000))]
  alpha.e = alpx
  alpha.w = alpha.W = alpx
  alpha.x = alpha.X = alpx
  alpha.m = alpx
  pn = pn #min(15, ceiling(length(Y)^{1/3.8})+4)
  
  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt #c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) # 
  Mn_Mod <- M%*%diag(1/Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20,"knots"), 1) ))
  Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a) 
  Wn <- array(NA, c(dim(W)[1], pn, n)) #(W%*%bs2)/length(a)
  
  #--- Reduce the dimension of the data
  for(i in 1:n){Wn[,,i] <- (W[,,i]%*%bs2)/length(a)}
  
  
  Xn <- (X%*%bs2)/length(a)
  
  cat("\n #--- Initial values...\n")  
  P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D #+ diag(rep(1e-6, K))
    return(P)
  }
  
  
  InitialValues <- function(Y, W, M, Z, df0=3, pn, Gx, a){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- Z #cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    lmFit <- lm(Y ~ -1 + Zmod + W_i) 
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- residuals.lm(lmFit)
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  InitialValuesQR <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z+Wn)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    #plot(a, bs2%*%lmFit$coefficients[-c(1:3)], type="l")
    
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    Tp0 = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd)), upper=c(30, 0.95*max(bnd) ))
    Res$Hess <- Tp0$hessian
    
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  #Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, t(apply(W,c(2,3), mean)), apply(M,c(1,2), mean), Z, df0=3, pn, Gx=1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  #Kvalues
  #Kc <- 5
  
  Kx = Thet0$Kx               ## number of X  cluster
  Ke = Thet0$Ke               ## number of ei cluster
  Km = Thet0$Km               ## number of ei cluster
  Kw = Thet0$Kw              ## number of omega_i cluster 
  
  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta   #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n=pn) #as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ #rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del#0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  #Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1]); Thet$gamU <- max(bnd[2])
  
  #bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) #c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
  
  Thet$Cik.x =  sample.int(Kx,size=n, replace = T) #Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e =  sample.int(Ke,size=n, replace = T) #Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w = sample.int(Kw,size=n, replace = T) #Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m = sample.int(Km,size=n, replace = T) #Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km
  
  
  Thet$Pik.x = rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)#Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e = rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)#Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w = rdirch(n=1,alpha = rep(alpha.e, Kw)/Kw) #Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m = rdirch(n=1,alpha = rep(alpha.m, Km)/Km)#Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw
  
  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw
  
  Thet$sig2k.e  = nimble::rinvgamma(n=Ke, shape=5/2,scale=8/2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n=Ke, min = .95*Thet$gamL, max=.95*Thet$gamU)
  Thet$p <- 1*(Thet$gamk < 0) + (Thet$tau0 - 1*(Thet$gamk < 0))/(2*pnorm(-abs(Thet$gamk))*exp(0.5*Thet$gamk*Thet$gamk))
  Thet$Ak <- (1-2*Thet$p)/(Thet$p*(1-Thet$p)); Thet$Bk <- 2/(Thet$p*(1-Thet$p)); Thet$Ck <- 1/( 1*(Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean=0, sd=sqrt(Thet$sig2k.e[1])) ; Thet$nui <- rexp(n=n, rate = 1/Thet$sig2k.e[1]) 
  #Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix
  
  Thet$Sigk.w <- repmat(c(.5*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw*pn), nrow = Kw)          #- vectors of length Kw and Kw respectively
  
  Thet$Sigk.x <- repmat(c(.5*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$Muk.x  <- matrix(rnorm(n=Kx*pn), nrow=Kx)   #- vectors of length Kx and Kx respectively
  
  Thet$Sigk.m <- repmat(c(.5*diag(pn)), Km, 1)     #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2*diag(pn)), Km, 1)
  Thet$Muk.m  <- matrix(rnorm(n=Km*pn), nrow = Km)   #- vectors of length Kx and Kx respectively
  
  A <- min(c(c(Wn), c(Mn))) - .5*diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5*diff(range(c(Wn, Mn)))
  #Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  #matrix(rnorm(n = n*pn), ncol=pn)
  #Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- t(apply(Wn,c(2,3), mean)) #Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))
  
  
  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
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
  
  
  #method = "L-BFGS-B"
  #method = "BFGS"
  
  #Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  #Res
  
  #Res$hessian
  #Res$hessian/n
  #solve(Res$hessian/n)
  
  #LogPlostGalLik(c(.05,), c(ei), Thet)
  
  #-------- Update Beta (vector of length pn) 
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #   
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma 
  #   
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #   
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #   
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #   
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)
  
  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together) 
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 = ncol(Z) +1
  Pgam <- P.mat(pn)*tauval
  Sig0.gam0 = diag(pn0); Mu0.gam0 = numeric(pn0+pn);
  #InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))
  
  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.gam0%*%Mu0.gam0))
    
    #c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb =c(rep(-Inf, ncol(Zmod)),rep(-g0,length(Mu.gam)-ncol(Zmod))),
                                ub = c(rep(Inf, ncol(Zmod)),rep(g0,length(Mu.gam)-ncol(Zmod)))   ))
    
  }
  updateGam <- cmpfun(updateGam)
  
  #siggam ~ IG(al0, bet0)
  al0 = 0.001
  bet0 = .005
  al0 = 2.5978252 #1
  bet0 = 0.4842963 #.005
  updateSigGam0 <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    val <- nimble::rinvgamma(n = 1, shape = al, scale = bet)
    ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  
  al0 = 0.001#1 #1
  bet0 = 0.001 #0.005 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    #val <- 
    #rgamma(n = 1, shape = al, scale = bet)
    nimble::rinvgamma(n = 1, shape = al, scale = bet)
    #ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  
  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e 
  #alpha.e = 1.0
  updatePik.e <- function(Thet, alpha.e){
    
    nk <- numeric(length(Thet$Cik.e))
    for(i in 1:length(Thet$Cik.e)){ nk[i] = sum(Thet$Cik.e == i)}
    
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$Cik.e) + nk
    
    rdirichlet(n=1, alpha.enew)
    
  }
  
  #-------- Update Cik.e  
  
  #ei <- Y - Z%*%Thet$Betaz - Thet$X%*%Thet$Gamma
  updateCik.e <- function(i, Thet,ei){
    #Y - Thet$C*abs(Thet$gam)*Thet$si - Thet$A*Thet$nui
    
    #  Yi.tld <- Y[i] - c(crossprod(Thet$Gamma, Thet$X[i,]))
    #pi.k  <- Thet$Pik.e*mapply(dnorm, x = Yi.tld, mean=Thet$muk.e, sd = sqrt(Thet$sig2k.e)) + .Machine$double.eps
    pik <- Thet$Pik.e*mapply(fGal.p0, e = ei[i], sig2 = sqrt(Thet$sig2k.e), gam = Thet$gamk, p = Thet$p, tau0 = Thet$tau0) #+ .Machine$double.eps
    which(rmultinom(n=1, size=1, prob = pik) == 1)
  }
  updateCik.e <- Vectorize(updateCik.e,"i") 
  
  #--- update Si
  #-- calss id for individual i (Matrix with columns of nui and si)
  updatenui <- function(Thet, Y, Z){
    
    ki = Thet$Cik.e
    bi <- ( ((Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ck[ki]*abs(Thet$gamk[ki])*Thet$si)^2)/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])))
    ai = ((Thet$Ak[ki]*Thet$Ak[ki])/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])) + 2/sqrt(Thet$sig2k.e[ki]))
    
    #mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) 
    #val = mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    #ifelse(val < .001, rexp(n = length(muui), rate = 1/sqrt(Thet$sig2k.e[Thet$Cik.e] )), val)
  }
  
  updatesi <- function(Thet, Y, Z){   
    #-- update si
    ki = Thet$Cik.e
    sig2si <- 1 /( 1/Thet$sig2k.e[ki] +  ((Thet$Ck[ki]*Thet$gamk[ki])^2)/ (Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui))
    muui <- sig2si*(Thet$Ck[ki]*abs(Thet$gamk[ki])*(Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ak[ki]*Thet$nui))/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui)
    
    #mapply(truncnorm::rtruncnorm, n=1, a = 0.01, mean = muui, sd = sqrt(sig2si))
    mapply(truncnorm::rtruncnorm, n=1, a = 0.001, mean = muui, sd = sqrt(sig2si))
    
    
  }
  
  
  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  #a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig=5/2; b.sig = 8/2; sdsig.prop = 0.2; sdgam.prop = .3  #2.153640 1.167331
  a.sig=2.153640; b.sig = 1.167331; sdsig.prop = 0.2; sdgam.prop = .3 # 7.133445 4.303263
  a.sig=7.133445; b.sig = 4.303263; sdsig.prop = 0.2; sdgam.prop = .3 #
  updatesig_gam <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop){
    
    a0 = 0.001
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      sig_newk <- truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gam <- Vectorize(updatesig_gam,"k")
  
  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10,10,10)*1
  varxi <- c(10,10,10)*.25
  varxi <- c(10,10,10)*.45 ## high aceeptance tunprop
  varxi <- c(10,10,10)*tunprop ##  tunprop
  #varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi){
    
    a0 = 0.1
    ab = 5 # THis works well
    ab = 10 # THis works well
    #varxi = 5
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]),Thet$gamk[k]),sigma = Sdsig.prop*varxi[k], lb = c(a0,0.95*Thet$gamL), ub = c(ab, .95*Thet$gamU)  )
      
      
      sig_newk <- Val[1] #truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] #truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu=c(sig_newk, gam_newk), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE) )
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes,"k")
  
  
  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices
  InvCov <- function(Mat){
    K <- nrow(Mat)
    J <- round(sqrt(ncol(Mat)))
    Res0 = matrix(0, nrow = K, ncol=ncol(Mat))
    for(k in 1:K){ Res0[k,] <- c(as.inverse(as.symmetric.matrix(matrix(Mat[k,],ncol=J, byrow=F)))) }
    Res0
  }
  
  
  #----------------------------------------------
  # Update things related to Wi 
  #--------------------------------------------
  #Piw
  #Cik.w
  #--- Update Pik.x  
  #alpha.W = alpha.w = 1.0
  updatePik.w <- function(Thet, alpha.W){
    Ku = nrow(Thet$Muk.w)
    nk <- numeric(Ku)
    for(k in 1:Ku){ nk[k] = sum(Thet$Cik.w == k)}
    
    #update the Pis
    alpha.wnew <- alpha.W/length(Thet$muk.w) + nk
    
    rdirichlet(n=1, alpha.wnew)
    
  }
  
  #--- Update Ci.k mixture of normals 
  dmvnormVec <- function(x, Mean, SigmaVc){
    Ku <- nrow(Mean)
    Val <- numeric(Ku)
    for(k in 1:Ku){ Val[k] <- mvtnorm::dmvnorm(x = x, mean = c(Mean[k,]), sigma = as.symmetric.matrix(matrix(SigmaVc[k,],ncol=length(x),byrow=F))) }
    Val
  }
  #dmvnormVec <- Vectorize(dmvnormVec,"k")
  #dmvnormVec(x = rep(0,6), Mn=Thet$Muk.w,SigmaVc = Thet$Sigk.w)
  
  updateCik.w <- function(i, Thet, W){ # mixture of normals 
    #Sig <- matrix(Thet$Sigk.w[id,], ncol = ncol(W), byrow = F)
    #kw = nrow(Thet$Muk.w)
    xi =  c(W[i,] - Thet$X[i,])
    #pi.k  = Thet$Pik.w*mapply(dmvnormVec, k = 1:Kw, MoreArgs = list(x = xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w))
    pi.k  = Thet$Pik.w*dmvnormVec(x=xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  
  #updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  #Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol=pn, nrow=pn));inSig0.w = as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.w <- function(Thet, W, Mu0.w, inSig0.w, Sig0.w){
    Ku = length(Thet$Pik.w)
    J = ncol(Thet$X) 
    nrep = dim(W)[1]
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol=J, nrow=J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = apply(W, 2, c) - matrix(rep(c((Thet$X)),each=dim(W)[1]),ncol=ncol(Thet$X),byrow=F)  #Thet$X 
    Yi.tld = W - array(matrix(rep(c((Thet$X)),each=dim(W)[1]),ncol=ncol(Thet$X),byrow=F),c(dim(W)[1], ncol(Thet$X), nrow(Thet$X))) ## This is now a array  where the rows of X are replicated nrep times in the array 
    
    #Muk <- matrix(0, ncol=J,nrow=Ku)
    SigK <- list()
    for(k in 1:Ku){
      idk <- which(Thet$Cik.w == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = apply(Yi.tld[,,idk], 2 ,c)   #matrix(c(Yi.tld[idk,]), nrow= nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nrep*nk*TpSi + inSig0.w)) ## sum of matrices inverse
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi.tld[idk,])) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
        Mu.k[k,] <- c(SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.w%*%Mu0.w))      ## some of mat
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(apply(t(Yi.tld[idk,]), 2, sum)) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
      }else{
        SigK[[k]] <- Sig0.w
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.w
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL; for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.w[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.w) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- 1.0*as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], sigma = SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.w[-Ku]) %*% SimMU.k)/Thet$Pik.w[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.w <- cmpfun(updateMuk.w)
  
  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w = pn + 2; Psi0.w = 0.5*(nu0.w-pn-1)*cov(apply(Wn, 2,c)) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  updateSigk.w <- function(Thet, W, nu0.w, Psi0.w){
    
    Ku = nrow(Thet$Muk.w)
    nrep = dim(W)[1]
      m = dim(W)[2]
    al <- numeric(Ku)
    bet <- numeric(Ku)
    #Wtl <- W -  Thet$X - Thet$Muk.w[Thet$Cik.w,]
    Wtl <- W - array(matrix(rep(c((Thet$X + Thet$Muk.w[Thet$Cik.w,])),each=dim(W)[1]),ncol=ncol(Thet$X),byrow=F),c(dim(W)[1], ncol(Thet$X), nrow(Thet$X))) ## An array (nrep, Pn, n)
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.w == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.w  + nk*nrep
        #Yi0 = matrix(c(Wtl[id,]), nrow= nk, byrow=F)
        Yi0 = apply(Wtl[,,id], 2, c)
        Psin.w <- as.positive.definite(Psi0.w + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.w))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.w, Psi0.w))
      }
    }
    Res
  }
  updateSigk.w <- cmpfun(updateSigk.w)
  
  #----------------------------------------------
  # Update things related to M 
  #--------------------------------------------  
  
  #--- update Pik.m for epsilon_i
  #alpha.m = 1.0
  updatePik.m <- function(Thet, alpha.m){
    nk <- numeric(nrow(Thet$Muk.m))
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    
    #update the Pis
    alpha.mnew <- alpha.m/nrow(Thet$Muk.m) + nk
    
    c(rdirichlet(n=1, alpha.mnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  
  updateCik.m <- function(i, Thet, M){ # mixture of normals 
    #id <- Thet$Cik.m[i]
    #Sig <- matrix(Thet$Sigk.m[id,], ncol = ncol(M), byrow = F)
    Km = nrow(Thet$Muk.m)
    xi = M[i,] - Thet$Delta*Thet$X[i,]
    #pi.k  = Thet$Pik.m*mapply(dmvnormVec, k = 1:Km, MoreArgs = list(x = xi , Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m))
    pi.k  = Thet$Pik.m*dmvnormVec(x = xi, Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m = rep(0, pn);  Sig0.m = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.m = as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.m <- function(Thet, M, Mu0.m, inSig0.m, Sig0.m){
    #cat(dim(inSig0.m),"(--1)\n")
    Ku = length(Thet$Pik.m)
    J = ncol(M)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol = J, nrow = J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = M - Thet$X%*%diag(Thet$Betadel) 
    Muk <- matrix(0, ncol=J, nrow=Ku)
    for(k in 1:Ku){
      idk <- which(Thet$Cik.m == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow = nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.m)) ## sum of matrices inverse
        #cat(dim(SigK[[k]]),"(--2 \n")
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.m%*%Mu0.m)      ## some of mat
        #cat(dim(SigKWg),"(--3 \n")
      }else{
        SigK[[k]] <- Sig0.m
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.m
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL;  for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.m[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.m) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.m[-Ku]) %*% SimMU.k)/Thet$Pik.m[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.m <- cmpfun(updateMuk.m)
  
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m = pn + 2; Psi0.m = .5*(nu0.m-pn-1)*as.symmetric.matrix(cov(t(apply(W,c(2,3), mean))))  #diag(pn)
  
  updateSigk.m <- function(Thet, M, nu0.m, Psi0.m){
    
    Ku = nrow(Thet$Muk.m)
    m = ncol(M)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Mtl <- M - Thet$X%*%diag(Thet$Betadel) - Thet$Muk.m[Thet$Cik.m,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.m == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.m  + nk
        Yi0 <- matrix(c(Mtl[id,]), nrow=nk, byrow=F)
        Psin.m <- as.positive.definite(Psi0.m + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.m))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.m, Psi0.m))
      }
    }
    Res
  }
  updateSigk.m <- cmpfun(updateSigk.m)
  
  #---- Update things related to delta
  quadFormVec <- function(i,X,sigma,Val){
    id = Val[i]
    as.symmetric.matrix(quad.form(as.inverse(as.symmetric.matrix(matrix(sigma[id,], ncol = length(X[i,]),byrow=F))), diag(c(X[i,])))) 
  }
  quadFormVec <- Vectorize(quadFormVec,"i",SIMPLIFY = F) ## Resulting in a list if precision matrices
  quadFormVec <- cmpfun(quadFormVec)
  
  quadFormVec3 <- function(i,Mi,X,sigma,Val){
    id = Val[i]
    c(quad.3form.inv(matrix(sigma[id,], ncol = length(X[i,]),byrow=F), diag(X[i,]),Mi[i,]))
  }
  quadFormVec3 <- Vectorize(quadFormVec3,"i", SIMPLIFY = F)
  quadFormVec3 <- cmpfun(quadFormVec3)
  
  #SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),   
  # prior values
  mu0.del = 1.0; #sig02.del = 100
  updateDelta <- function(Thet, M, mu0.del, sig02.del){
    n = nrow(M)
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X), Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    sig20 <- 1/( 1/sig02.del + sum(Res))
    mu <- sig20*(sum(Res3) + mu0.del/sig02.del)
    rtruncnorm(n=1, a=0, b=10, mean = mu, sd = sqrt(sig20))
    
    #c(sum(Res3), sum(Res),mu, sqrt(sig20), rtruncnorm(n=1, a=0, b=Inf, mean = mu, sd = sqrt(sig20)))
    
  }
  updateDelta <- cmpfun(updateDelta)
  
  
  # prior values
  mu0.del = rep(0,length(Thet$Betadel)); #sig02.del = 20
  Sig0.del = P.mat(length(Thet$Betadel))*.1   #Thet$siggam) # Inverse or precision covariance matrix
  
  updateFunDelta <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDelta <- cmpfun(updateFunDelta)
  
  
  SimDElVec <- function(i, Mi, X, Sigma, Val){
    id <- Val[i]
    Ot = mvtnorm::rmvnorm(n = 1,mean = Mi[i,],sigma = as.symmetric.matrix(matrix(Sigma[id,], ncol = length(X[i,]),byrow=F)))
  }
  SimDElVec <- Vectorize(SimDElVec, "i")
  
  #-- 
  DeltComp <- function(k,Xst, X){
    abs(sum(Xst[,k]*X[,k])/(sum(X[,k]^2)))
    #abs(diag(as.inverse(crossprod(X))%*%crossprod(X,Xst)))
  }
  DeltComp <- Vectorize(DeltComp,"k")
  
  updateFunDeltaSt <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Xst <- base::t(SimDElVec(1:nrow(M), Mitl, Thet$X, Thet$Sigk.m, Thet$Cik.m)) ## K by n matrix and now I get a n by K
    c(DeltComp(1:ncol(Thet$X), Xst, Thet$X))
    #c(DeltComp(Xst, Thet$X))
    
    # Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    # Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    # Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    # Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    # c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDeltaSt  <- cmpfun(updateFunDeltaSt )
  
  
  #----------------------------------------------
  # Update things related to X 
  #--------------------------------------------------  
  #--- update Pik.m for epsilon_i
  #alpha.x <- 1.0
  updatePik.x <- function(Thet, alpha.x){
    nk <- numeric(nrow(Thet$Muk.x))
    for(k in 1:nrow(Thet$Muk.x)){ nk[k] = sum(Thet$Cik.x == k)}
    
    #update the Pis
    alpha.xnew <- alpha.x/nrow(Thet$Muk.x) + nk
    
    c(rdirichlet(n=1, alpha.xnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  updateCik.x <- function(i, Thet){ # mixture of normals 
    #Kx = ncol(Thet$X) # J x J
    #id <- Thet$Cik.x[i]
    #Sig <- matrix(Thet$Sigk.x[id,], ncol = length(Thet$Gamma), byrow = F)
    Xi = Thet$X[i,]
    #pi.k  = Thet$Pik.x*mapply(dmvnormVec, k = 1:Kx, x = Xi,  Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x)
    pi.k  = Thet$Pik.x*dmvnormVec(x=Xi,Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x = numeric(pn); Sig0.x = .5*diag(pn); inSig0.x = 2*diag(pn)#as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x = rep(0, pn);  Sig0.x = .25*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w)+colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.x = as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.x <- function(Thet, Mu0.x, inSig0.x, Sig0.x){
    K = length(Thet$Pik.x)
    J = ncol(Thet$X)
    Res0 <- matrix(0, nrow = K, ncol = J)
    Mu.k <- matrix(0, nrow = K, ncol = J) # K x J
    #SigKWg <- matrix(0, ncol = ncol(Y),nrow=ncol(Y)) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = Thet$X 
    Muk <- matrix(0, ncol = J, nrow = K)
    for(k in 1:K){
      idk <- which(Thet$Cik.x == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 <- matrix(c(Yi.tld[idk,]), nrow=nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[k,], ncol=J, byrow=F)))
        SigK <- as.inverse(as.symmetric.matrix((nk*TpSi + inSig0.x))) ## sum of matrices inverse
        Muk <- SigK%*%(crossprod(TpSi, c(colSums(Yi0))) + inSig0.x%*%Mu0.x)      ## some of mat
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Muk, sigma = SigK))
      }else{
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Mu0.x, sigma = Sig0.x ))
      }
    }
    #--- we impose some constrainsts on the mean
    Mu.k
    
  }
  updateMuk.x <- cmpfun(updateMuk.x)
  
  nu0.x <- pn + 2 ; Psi0.x <- .5*diag(pn) #as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2 ; Psi0.x <- .5*(nu0.x-pn-1)*cov(.5*(t(apply(Wn,c(2,3), mean)))) #as.symmetric.matrix(.25*cov(Wn))
  updateSigk.x <- function(Thet, nu0.x, Psi0.x){
    
    Ku = nrow(Thet$Muk.x)
    m = ncol(Thet$Muk.x)
    Xtl <- Thet$X - Thet$Muk.x[Thet$Cik.x,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.x == i)
      nk = length(id)
      if(nk > 0){
        nux <- nu0.x  + nk
        Yi0 <- matrix(c(Xtl[id,]), nrow=nk, byrow=F)
        Psin.x <- Psi0.x + crossprod(Yi0)
        Res[i, ] <- c(rinvwishart(nux, Psin.x))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.x, Psi0.x))
      }
    }
    Res
  }
  updateSigk.x <- cmpfun(updateSigk.x)
  # 
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn))) 
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn))) 
  
  A <- min(c(Wn, Mn)) - .1*diff(range(c(Wn, Mn))) 
  B <- max(c(Wn, Mn)) + .1*diff(range(c(Wn, Mn))) 
  
  udapateXi <- function(i, Thet, Y, W, M, Z, A=A, B=B){
    ie =Thet$Cik.e[i]
    iw = Thet$Cik.w[i]
    im = Thet$Cik.m[i]
    ix = Thet$Cik.x[i]
    J = ncol(Thet$X)
    
    Zmod <- Z #cbind(1,Z)
    Ytl = (Y - Zmod%*%Thet$BetaZ - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Wtl = W - Thet$Muk.w[Thet$Cik.w,]
    Mtl = (M - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
    Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
    Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
    
    #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
    Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
    
    Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))) 
    
    #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi <- cmpfun(udapateXi)
  
  
  udapateXi2A <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2A <- cmpfun(udapateXi2A)
  
  ## This approach of simulating Xi only uses W, M, and X
  
  udapateXi2 <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    #Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    
    Wtl = Wn - array(matrix(rep(c((Thet$Muk.w[Thet$Cik.w,])),each=dim(Wn)[1]),ncol=ncol(Thet$X),byrow=F),c(dim(Wn)[1], ncol(Thet$X), nrow(Thet$X))) ## This is an array (nrep, pn,n)
    #Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)*0
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + (dim(Wn)[1])*Sigw + Sigm*0 + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      #Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw, matrix(colSums(Wtl[,,i])), ncol=1) + 0*crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,])));
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw, matrix(colSums(Wtl[,,i])), ncol=1) + crossprod(Sigx, Thet$Muk.x[ix,])));
      
      #Mu <- c(Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2 <- cmpfun(udapateXi2)
  
  
  udapateXi2V <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    #Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    #GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- (Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi2V <- Vectorize(udapateXi2V,"i")
  udapateXi2V <- cmpfun(udapateXi2V)
  
  #--- Write Rcpp function to speed things up
  {
    #
    
    
  }
  
  
  
  #Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)
  
  ##----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  #Nsim <- 5000
  #Nsim = 5000 
  MCMCSample <- list()
  
  #-- for X
  MCMCSample$X <- matrix(0,nrow = Nsim,ncol=length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined) 
  MCMCSample$Muk.x <- matrix(0,nrow = Nsim, ncol = Kx*pn)
  MCMCSample$Sigk.x <- matrix(0,nrow = Nsim,ncol=Kx*pn*pn)
  MCMCSample$Pik.x <- matrix(0,nrow = Nsim,ncol=Kx)
  MCMCSample$Cik.x <- matrix(0,nrow = Nsim,ncol=length(Y))
  
  #-- for W
  MCMCSample$Muk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn)
  MCMCSample$Sigk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn*pn)
  MCMCSample$Pik.w <- matrix(0,nrow = Nsim,ncol=Kw)
  MCMCSample$Cik.w <- matrix(0,nrow = Nsim,ncol=length(Y))
  
  #-- for M
  MCMCSample$Muk.m <- matrix(0,nrow = Nsim,ncol=Km*pn)
  MCMCSample$Sigk.m <- matrix(0,nrow = Nsim,ncol=Km*pn*pn)
  MCMCSample$Pik.m <- matrix(0,nrow = Nsim,ncol=Km)
  MCMCSample$Cik.m <- matrix(0,nrow = Nsim,ncol=length(Y))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow=Nsim, ncol=length(Thet$Betadel))
  
  # For Y
  MCMCSample$gamk <- matrix(0,nrow = Nsim, ncol=Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Pik.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Cik.e <- matrix(0,nrow = Nsim, ncol=length(Y))
  
  MCMCSample$Gamma <- matrix(0,nrow = Nsim, ncol=length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)
  
  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  # 
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  # 
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  # 
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  #Accept <- 
  # 
  MCMCSample$AcceptX <- matrix(0,nrow = Nsim, ncol=Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  #stop()
  #------------------------------------------------------------------
  #Nsim = 5000 
  #Nsim = 1000
  #Nsim = 3000
  #library(profvis)
  #profvis(
  #system.time(
  cat(dim(Wn),"\n")
  cat("\n #--- MCMC is starting ----> \n")
  
  for(iter in 1:Nsim){
    if(iter %% 500 == 0){cat("iter=",iter,"\n")}
    #----------------------------------------------
    # Update things related to X 
    #--------------------------------------------  
    #--- update Pik.m for epsilon_i
    Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    
    #--- update Cik.m for epsilon_i based on mixture of normals
    Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x)) 
    
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    
    Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    Thet$InvSigk.x <- InvCov(Thet$Sigk.x)
    
    #Ytd = Y - Zmod%*%Thet$BetaZ
    if(iter%%100 == 0){
      Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B)}
    # Xn #
    #Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    #Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    #Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w, 
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)
    
    #Thet$X <- Xn #UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,]) 
    
    #}
    #dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))
    
    #Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    #Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    #Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    #Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w, 
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J))) 
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Kw)
    for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    #update the Pis
    alpha.wnew <- alpha.W/Kw + nk
    Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    
    #--- Update Ci.k mixture of normals
    #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    #Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    #Thet$Cik.w <- F2Cmp(CikWRep(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w)) #F2LCmp
    #W0 <- CikWrep(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w)
    #W0 <- CikW(t(apply(Wn, c(2,3), mean)), Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w)
    Thet$Cik.w <- F2Cmp(CikWrep(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w), Thet$Pik.w) #F2LCmp
    #Thet$Cik.w <- F2Cmp(CikWRepVec(c(dim(Wn)), c(Wn), Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w), )
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w  
    # #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Km)
    # for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    # #update the Pis
    # alpha.mnew <- alpha.m/Kw + nk
    # Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    # 
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    # Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    # #Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))  
    # #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    # Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    # Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # #if(iter%%50 == 0){
    # Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    # MCMCSample$Betdel[iter,] <- Thet$Betadel
    #}
    #---------------------------------------------
    #cat("#--- update things related to Y or epsilon ")
    #--------------------------------------------- 
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for(k in 1:length(Thet$sig2k.e)){ nk[k] = sum(Thet$Cik.e == k)}
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n=1, alpha.enew))
    
    #update the Cik.e
    ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
    #Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) #rep(1,n) #
    #Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    #update nui  and si
    #Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    #nuisi <- updatenuisi(Thet, Y, Zmod)
    #Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    #update Sigk.e
    #Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    #update sigk and gamk
    #--- update jointly sig2ke and gamke
    #Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi)
    Thet$sig2k.e <- Temp[1,];  Thet$gamk <- Temp[2,]; Thet$p <- Temp[3,]; MCMCSample$AcceptX[iter,] <- Temp[4,]; Thet$Ak <- Temp[5,]; Thet$Bk <- Temp[6,]; Thet$Ck <- Temp[7,];
    
    #update Gamma
    #Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    #Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <-  updateSigGam(Thet, al0, bet0, order = 2)# sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#
    
    #Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    # 
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    # 
    # #--- upodate beta_{z,m}
    # #--  Prior 
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    # 
    # #--- update X_i 
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX) 
    # 
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]); 
    # 
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb 
    
    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x))  ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise
    
    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w
    
    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    #MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e
    
    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  } 
  
  #)
  #)
  
  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample
}



#--- replicate W
#-- THis function will perform the estimation
#-- replicates values of W

BQBayes.ME_SoFRFuncSimRepW <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0 = 0.5, X, g0,tunprop){
  
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
  met <- c("f1","f2","f3","f4","f5")
  
  
  func <- function(Pik, n=1, size=1){
    which.max(rmultinom(n=1,size=1,prob = Pik))
  }
  funcCmp <- cmpfun(func)
  
  F2 <- function(Pik, n=1, size=1){
    
    apply(Pik,1,funcCmp)
  }
  F2Cmp <- cmpfun(F2)
  
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
  
  
  
  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  #pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  #pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  n = length(Y) 
  Zmod = cbind(1,Z)
  pn = c(5, 6, 20, 20)[which(n == c(100, 200,500,1000))]
  alpha.e = alpx
  alpha.w = alpha.W = alpx
  alpha.x = alpha.X = alpx
  alpha.m = alpx
  pn = pn #min(15, ceiling(length(Y)^{1/3.8})+4)
  
  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt #c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) # 
  Mn_Mod <- M%*%diag(1/Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20,"knots"), 1) ))
  Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a) 
  Wn <- (W%*%bs2)/length(a)
  Xn <- (X%*%bs2)/length(a)
  
  cat("\n #--- Initial values...\n")  
  P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D #+ diag(rep(1e-6, K))
    return(P)
  }
  
  
  InitialValues <- function(Y, W, M, Z, df0=3, pn, Gx, a){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- Z #cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    lmFit <- lm(Y ~ -1 + Zmod + W_i) 
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- residuals.lm(lmFit)
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  InitialValuesQR <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z+Wn)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    plot(a, bs2%*%lmFit$coefficients[-c(1:3)], type="l")
    
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    Tp0 = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd)), upper=c(30, 0.95*max(bnd) ))
    Res$Hess <- Tp0$hessian
    
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  #Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, W, M, Z, df0=3, pn, Gx=1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  #Kvalues
  #Kc <- 5
  
  Kx = Thet0$Kx               ## number of X  cluster
  Ke = Thet0$Ke               ## number of ei cluster
  Km = Thet0$Km               ## number of ei cluster
  Kw = Thet0$Kw              ## number of omega_i cluster 
  
  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta   #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n=pn) #as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ #rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del#0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  #Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1]); Thet$gamU <- max(bnd[2])
  
  #bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) #c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
  
  Thet$Cik.x =  sample.int(Kx,size=n, replace = T) #Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e =  sample.int(Ke,size=n, replace = T) #Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w = sample.int(Kw,size=n, replace = T) #Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m = sample.int(Km,size=n, replace = T) #Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km
  
  
  Thet$Pik.x = rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)#Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e = rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)#Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w = rdirch(n=1,alpha = rep(alpha.e, Kw)/Kw) #Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m = rdirch(n=1,alpha = rep(alpha.m, Km)/Km)#Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw
  
  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw
  
  Thet$sig2k.e  = nimble::rinvgamma(n=Ke, shape=5/2,scale=8/2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n=Ke, min = .95*Thet$gamL, max=.95*Thet$gamU)
  Thet$p <- 1*(Thet$gamk < 0) + (Thet$tau0 - 1*(Thet$gamk < 0))/(2*pnorm(-abs(Thet$gamk))*exp(0.5*Thet$gamk*Thet$gamk))
  Thet$Ak <- (1-2*Thet$p)/(Thet$p*(1-Thet$p)); Thet$Bk <- 2/(Thet$p*(1-Thet$p)); Thet$Ck <- 1/( 1*(Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean=0, sd=sqrt(Thet$sig2k.e[1])) ; Thet$nui <- rexp(n=n, rate = 1/Thet$sig2k.e[1]) 
  #Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix
  
  Thet$Sigk.w <- repmat(c(.5*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw*pn), nrow = Kw)          #- vectors of length Kw and Kw respectively
  
  Thet$Sigk.x <- repmat(c(.5*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$Muk.x  <- matrix(rnorm(n=Kx*pn), nrow=Kx)   #- vectors of length Kx and Kx respectively
  
  Thet$Sigk.m <- repmat(c(.5*diag(pn)), Km, 1)     #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2*diag(pn)), Km, 1)
  Thet$Muk.m  <- matrix(rnorm(n=Km*pn), nrow = Km)   #- vectors of length Kx and Kx respectively
  
  A <- min(c(c(Wn), c(Mn))) - .5*diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5*diff(range(c(Wn, Mn)))
  #Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  #matrix(rnorm(n = n*pn), ncol=pn)
  #Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5*(Mn+Wn) #Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))
  
  
  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
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
  
  
  #method = "L-BFGS-B"
  #method = "BFGS"
  
  #Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  #Res
  
  #Res$hessian
  #Res$hessian/n
  #solve(Res$hessian/n)
  
  #LogPlostGalLik(c(.05,), c(ei), Thet)
  
  #-------- Update Beta (vector of length pn) 
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #   
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma 
  #   
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #   
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #   
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #   
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)
  
  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together) 
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 = ncol(Z) +1
  Pgam <- P.mat(pn)*tauval
  Sig0.gam0 = diag(pn0); Mu0.gam0 = numeric(pn0+pn);
  #InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))
  
  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.gam0%*%Mu0.gam0))
    
    #c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    #c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb =c(rep(-Inf, ncol(Zmod)),rep(-g0,length(Mu.gam)-ncol(Zmod))),
                                ub = c(rep(Inf, ncol(Zmod)),rep(g0,length(Mu.gam)-ncol(Zmod)))   ))
    
  }
  updateGam <- cmpfun(updateGam)
  
  #siggam ~ IG(al0, bet0)
  al0 = 0.001
  bet0 = .005
  al0 = 2.5978252 #1
  bet0 = 0.4842963 #.005
  updateSigGam0 <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    val <- nimble::rinvgamma(n = 1, shape = al, scale = bet)
    ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  
  al0 = 0.001#1 #1
  bet0 = 0.001 #0.005 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    #val <- 
    #rgamma(n = 1, shape = al, scale = bet)
    nimble::rinvgamma(n = 1, shape = al, scale = bet)
    #ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  
  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e 
  #alpha.e = 1.0
  updatePik.e <- function(Thet, alpha.e){
    
    nk <- numeric(length(Thet$Cik.e))
    for(i in 1:length(Thet$Cik.e)){ nk[i] = sum(Thet$Cik.e == i)}
    
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$Cik.e) + nk
    
    rdirichlet(n=1, alpha.enew)
    
  }
  
  #-------- Update Cik.e  
  
  #ei <- Y - Z%*%Thet$Betaz - Thet$X%*%Thet$Gamma
  updateCik.e <- function(i, Thet,ei){
    #Y - Thet$C*abs(Thet$gam)*Thet$si - Thet$A*Thet$nui
    
    #  Yi.tld <- Y[i] - c(crossprod(Thet$Gamma, Thet$X[i,]))
    #pi.k  <- Thet$Pik.e*mapply(dnorm, x = Yi.tld, mean=Thet$muk.e, sd = sqrt(Thet$sig2k.e)) + .Machine$double.eps
    pik <- Thet$Pik.e*mapply(fGal.p0, e = ei[i], sig2 = sqrt(Thet$sig2k.e), gam = Thet$gamk, p = Thet$p, tau0 = Thet$tau0) #+ .Machine$double.eps
    which(rmultinom(n=1, size=1, prob = pik) == 1)
  }
  updateCik.e <- Vectorize(updateCik.e,"i") 
  
  #--- update Si
  #-- calss id for individual i (Matrix with columns of nui and si)
  updatenui <- function(Thet, Y, Z){
    
    ki = Thet$Cik.e
    bi <- ( ((Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ck[ki]*abs(Thet$gamk[ki])*Thet$si)^2)/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])))
    ai = ((Thet$Ak[ki]*Thet$Ak[ki])/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])) + 2/sqrt(Thet$sig2k.e[ki]))
    
    #mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) 
    #val = mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    #ifelse(val < .001, rexp(n = length(muui), rate = 1/sqrt(Thet$sig2k.e[Thet$Cik.e] )), val)
  }
  
  updatesi <- function(Thet, Y, Z){   
    #-- update si
    ki = Thet$Cik.e
    sig2si <- 1 /( 1/Thet$sig2k.e[ki] +  ((Thet$Ck[ki]*Thet$gamk[ki])^2)/ (Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui))
    muui <- sig2si*(Thet$Ck[ki]*abs(Thet$gamk[ki])*(Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ak[ki]*Thet$nui))/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui)
    
    #mapply(truncnorm::rtruncnorm, n=1, a = 0.01, mean = muui, sd = sqrt(sig2si))
    mapply(truncnorm::rtruncnorm, n=1, a = 0.001, mean = muui, sd = sqrt(sig2si))
    
    
  }
  
  
  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  #a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig=5/2; b.sig = 8/2; sdsig.prop = 0.2; sdgam.prop = .3  #2.153640 1.167331
  a.sig=2.153640; b.sig = 1.167331; sdsig.prop = 0.2; sdgam.prop = .3 # 7.133445 4.303263
  a.sig=7.133445; b.sig = 4.303263; sdsig.prop = 0.2; sdgam.prop = .3 #
  updatesig_gam <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop){
    
    a0 = 0.001
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      sig_newk <- truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gam <- Vectorize(updatesig_gam,"k")
  
  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10,10,10)*1
  varxi <- c(10,10,10)*.25
  varxi <- c(10,10,10)*.45 ## high aceeptance tunprop
  varxi <- c(10,10,10)*tunprop ##  tunprop
  #varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi){
    
    a0 = 0.1
    ab = 5 # THis works well
    ab = 10 # THis works well
    #varxi = 5
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]),Thet$gamk[k]),sigma = Sdsig.prop*varxi[k], lb = c(a0,0.95*Thet$gamL), ub = c(ab, .95*Thet$gamU)  )
      
      
      sig_newk <- Val[1] #truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] #truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu=c(sig_newk, gam_newk), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE) )
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes,"k")
  
  
  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices
  InvCov <- function(Mat){
    K <- nrow(Mat)
    J <- round(sqrt(ncol(Mat)))
    Res0 = matrix(0, nrow = K, ncol=ncol(Mat))
    for(k in 1:K){ Res0[k,] <- c(as.inverse(as.symmetric.matrix(matrix(Mat[k,],ncol=J, byrow=F)))) }
    Res0
  }
  
  
  #----------------------------------------------
  # Update things related to Wi 
  #--------------------------------------------
  #Piw
  #Cik.w
  #--- Update Pik.x  
  #alpha.W = alpha.w = 1.0
  updatePik.w <- function(Thet, alpha.W){
    Ku = nrow(Thet$Muk.w)
    nk <- numeric(Ku)
    for(k in 1:Ku){ nk[k] = sum(Thet$Cik.w == k)}
    
    #update the Pis
    alpha.wnew <- alpha.W/length(Thet$muk.w) + nk
    
    rdirichlet(n=1, alpha.wnew)
    
  }
  
  #--- Update Ci.k mixture of normals 
  dmvnormVec <- function(x, Mean, SigmaVc){
    Ku <- nrow(Mean)
    Val <- numeric(Ku)
    for(k in 1:Ku){ Val[k] <- mvtnorm::dmvnorm(x = x, mean = c(Mean[k,]), sigma = as.symmetric.matrix(matrix(SigmaVc[k,],ncol=length(x),byrow=F))) }
    Val
  }
  #dmvnormVec <- Vectorize(dmvnormVec,"k")
  #dmvnormVec(x = rep(0,6), Mn=Thet$Muk.w,SigmaVc = Thet$Sigk.w)
  
  updateCik.w <- function(i, Thet, W){ # mixture of normals 
    #Sig <- matrix(Thet$Sigk.w[id,], ncol = ncol(W), byrow = F)
    #kw = nrow(Thet$Muk.w)
    xi =  c(W[i,] - Thet$X[i,])
    #pi.k  = Thet$Pik.w*mapply(dmvnormVec, k = 1:Kw, MoreArgs = list(x = xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w))
    pi.k  = Thet$Pik.w*dmvnormVec(x=xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  
  #updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  #Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol=pn, nrow=pn));inSig0.w = as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.w <- function(Thet, W, Mu0.w, inSig0.w, Sig0.w){
    Ku = length(Thet$Pik.w)
    J = ncol(Thet$X)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol=J, nrow=J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = W - Thet$X 
    #Muk <- matrix(0, ncol=J,nrow=Ku)
    SigK <- list()
    for(k in 1:Ku){
      idk <- which(Thet$Cik.w == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow= nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.w)) ## sum of matrices inverse
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi.tld[idk,])) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
        Mu.k[k,] <- c(SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.w%*%Mu0.w))      ## some of mat
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(apply(t(Yi.tld[idk,]), 2, sum)) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
      }else{
        SigK[[k]] <- Sig0.w
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.w
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL; for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.w[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.w) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- 1.0*as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], sigma = SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.w[-Ku]) %*% SimMU.k)/Thet$Pik.w[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.w <- cmpfun(updateMuk.w)
  
  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w = pn + 2; Psi0.w = 0.5*(nu0.w-pn-1)*cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  updateSigk.w <- function(Thet, W, nu0.w, Psi0.w){
    
    Ku = nrow(Thet$Muk.w)
    m = ncol(W)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Wtl <- W - Thet$X - Thet$Muk.w[Thet$Cik.w,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.w == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.w  + nk
        Yi0 = matrix(c(Wtl[id,]), nrow= nk, byrow=F)
        Psin.w <- as.positive.definite(Psi0.w + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.w))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.w, Psi0.w))
      }
    }
    Res
  }
  updateSigk.w <- cmpfun(updateSigk.w)
  
  #----------------------------------------------
  # Update things related to M 
  #--------------------------------------------  
  
  #--- update Pik.m for epsilon_i
  #alpha.m = 1.0
  updatePik.m <- function(Thet, alpha.m){
    nk <- numeric(nrow(Thet$Muk.m))
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    
    #update the Pis
    alpha.mnew <- alpha.m/nrow(Thet$Muk.m) + nk
    
    c(rdirichlet(n=1, alpha.mnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  
  updateCik.m <- function(i, Thet, M){ # mixture of normals 
    #id <- Thet$Cik.m[i]
    #Sig <- matrix(Thet$Sigk.m[id,], ncol = ncol(M), byrow = F)
    Km = nrow(Thet$Muk.m)
    xi = M[i,] - Thet$Delta*Thet$X[i,]
    #pi.k  = Thet$Pik.m*mapply(dmvnormVec, k = 1:Km, MoreArgs = list(x = xi , Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m))
    pi.k  = Thet$Pik.m*dmvnormVec(x = xi, Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m = rep(0, pn);  Sig0.m = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.m = as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.m <- function(Thet, M, Mu0.m, inSig0.m, Sig0.m){
    #cat(dim(inSig0.m),"(--1)\n")
    Ku = length(Thet$Pik.m)
    J = ncol(M)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol = J, nrow = J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = M - Thet$X%*%diag(Thet$Betadel) 
    Muk <- matrix(0, ncol=J, nrow=Ku)
    for(k in 1:Ku){
      idk <- which(Thet$Cik.m == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow = nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.m)) ## sum of matrices inverse
        #cat(dim(SigK[[k]]),"(--2 \n")
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.m%*%Mu0.m)      ## some of mat
        #cat(dim(SigKWg),"(--3 \n")
      }else{
        SigK[[k]] <- Sig0.m
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.m
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL;  for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.m[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.m) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.m[-Ku]) %*% SimMU.k)/Thet$Pik.m[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.m <- cmpfun(updateMuk.m)
  
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m = pn + 2; Psi0.m = .5*(nu0.m-pn-1)*as.symmetric.matrix(cov(Wn))  #diag(pn)
  
  updateSigk.m <- function(Thet, M, nu0.m, Psi0.m){
    
    Ku = nrow(Thet$Muk.m)
    m = ncol(M)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Mtl <- M - Thet$X%*%diag(Thet$Betadel) - Thet$Muk.m[Thet$Cik.m,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.m == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.m  + nk
        Yi0 <- matrix(c(Mtl[id,]), nrow=nk, byrow=F)
        Psin.m <- as.positive.definite(Psi0.m + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.m))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.m, Psi0.m))
      }
    }
    Res
  }
  updateSigk.m <- cmpfun(updateSigk.m)
  
  #---- Update things related to delta
  quadFormVec <- function(i,X,sigma,Val){
    id = Val[i]
    as.symmetric.matrix(quad.form(as.inverse(as.symmetric.matrix(matrix(sigma[id,], ncol = length(X[i,]),byrow=F))), diag(c(X[i,])))) 
  }
  quadFormVec <- Vectorize(quadFormVec,"i",SIMPLIFY = F) ## Resulting in a list if precision matrices
  quadFormVec <- cmpfun(quadFormVec)
  
  quadFormVec3 <- function(i,Mi,X,sigma,Val){
    id = Val[i]
    c(quad.3form.inv(matrix(sigma[id,], ncol = length(X[i,]),byrow=F), diag(X[i,]),Mi[i,]))
  }
  quadFormVec3 <- Vectorize(quadFormVec3,"i", SIMPLIFY = F)
  quadFormVec3 <- cmpfun(quadFormVec3)
  
  #SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),   
  # prior values
  mu0.del = 1.0; #sig02.del = 100
  updateDelta <- function(Thet, M, mu0.del, sig02.del){
    n = nrow(M)
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X), Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    sig20 <- 1/( 1/sig02.del + sum(Res))
    mu <- sig20*(sum(Res3) + mu0.del/sig02.del)
    rtruncnorm(n=1, a=0, b=10, mean = mu, sd = sqrt(sig20))
    
    #c(sum(Res3), sum(Res),mu, sqrt(sig20), rtruncnorm(n=1, a=0, b=Inf, mean = mu, sd = sqrt(sig20)))
    
  }
  updateDelta <- cmpfun(updateDelta)
  
  
  # prior values
  mu0.del = rep(0,length(Thet$Betadel)); #sig02.del = 20
  Sig0.del = P.mat(length(Thet$Betadel))*.1   #Thet$siggam) # Inverse or precision covariance matrix
  
  updateFunDelta <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDelta <- cmpfun(updateFunDelta)
  
  
  SimDElVec <- function(i, Mi, X, Sigma, Val){
    id <- Val[i]
    Ot = mvtnorm::rmvnorm(n = 1,mean = Mi[i,],sigma = as.symmetric.matrix(matrix(Sigma[id,], ncol = length(X[i,]),byrow=F)))
  }
  SimDElVec <- Vectorize(SimDElVec, "i")
  
  #-- 
  DeltComp <- function(k,Xst, X){
    abs(sum(Xst[,k]*X[,k])/(sum(X[,k]^2)))
    #abs(diag(as.inverse(crossprod(X))%*%crossprod(X,Xst)))
  }
  DeltComp <- Vectorize(DeltComp,"k")
  
  updateFunDeltaSt <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Xst <- base::t(SimDElVec(1:nrow(M), Mitl, Thet$X, Thet$Sigk.m, Thet$Cik.m)) ## K by n matrix and now I get a n by K
    c(DeltComp(1:ncol(Thet$X), Xst, Thet$X))
    #c(DeltComp(Xst, Thet$X))
    
    # Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    # Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    # Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    # Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    # c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDeltaSt  <- cmpfun(updateFunDeltaSt )
  
  
  #----------------------------------------------
  # Update things related to X 
  #--------------------------------------------------  
  #--- update Pik.m for epsilon_i
  #alpha.x <- 1.0
  updatePik.x <- function(Thet, alpha.x){
    nk <- numeric(nrow(Thet$Muk.x))
    for(k in 1:nrow(Thet$Muk.x)){ nk[k] = sum(Thet$Cik.x == k)}
    
    #update the Pis
    alpha.xnew <- alpha.x/nrow(Thet$Muk.x) + nk
    
    c(rdirichlet(n=1, alpha.xnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  updateCik.x <- function(i, Thet){ # mixture of normals 
    #Kx = ncol(Thet$X) # J x J
    #id <- Thet$Cik.x[i]
    #Sig <- matrix(Thet$Sigk.x[id,], ncol = length(Thet$Gamma), byrow = F)
    Xi = Thet$X[i,]
    #pi.k  = Thet$Pik.x*mapply(dmvnormVec, k = 1:Kx, x = Xi,  Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x)
    pi.k  = Thet$Pik.x*dmvnormVec(x=Xi,Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x = numeric(pn); Sig0.x = .5*diag(pn); inSig0.x = 2*diag(pn)#as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x = rep(0, pn);  Sig0.x = .25*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w)+colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.x = as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.x <- function(Thet, Mu0.x, inSig0.x, Sig0.x){
    K = length(Thet$Pik.x)
    J = ncol(Thet$X)
    Res0 <- matrix(0, nrow = K, ncol = J)
    Mu.k <- matrix(0, nrow = K, ncol = J) # K x J
    #SigKWg <- matrix(0, ncol = ncol(Y),nrow=ncol(Y)) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = Thet$X 
    Muk <- matrix(0, ncol = J, nrow = K)
    for(k in 1:K){
      idk <- which(Thet$Cik.x == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 <- matrix(c(Yi.tld[idk,]), nrow=nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[k,], ncol=J, byrow=F)))
        SigK <- as.inverse(as.symmetric.matrix((nk*TpSi + inSig0.x))) ## sum of matrices inverse
        Muk <- SigK%*%(crossprod(TpSi, c(colSums(Yi0))) + inSig0.x%*%Mu0.x)      ## some of mat
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Muk, sigma = SigK))
      }else{
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Mu0.x, sigma = Sig0.x ))
      }
    }
    #--- we impose some constrainsts on the mean
    Mu.k
    
  }
  updateMuk.x <- cmpfun(updateMuk.x)
  
  nu0.x <- pn + 2 ; Psi0.x <- .5*diag(pn) #as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2 ; Psi0.x <- .5*(nu0.x-pn-1)*cov(.5*(Wn+Mn)) #as.symmetric.matrix(.25*cov(Wn))
  updateSigk.x <- function(Thet, nu0.x, Psi0.x){
    
    Ku = nrow(Thet$Muk.x)
    m = ncol(Thet$Muk.x)
    Xtl <- Thet$X - Thet$Muk.x[Thet$Cik.x,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.x == i)
      nk = length(id)
      if(nk > 0){
        nux <- nu0.x  + nk
        Yi0 <- matrix(c(Xtl[id,]), nrow=nk, byrow=F)
        Psin.x <- Psi0.x + crossprod(Yi0)
        Res[i, ] <- c(rinvwishart(nux, Psin.x))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.x, Psi0.x))
      }
    }
    Res
  }
  updateSigk.x <- cmpfun(updateSigk.x)
  # 
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn))) 
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn))) 
  
  A <- min(c(Wn, Mn)) - .1*diff(range(c(Wn, Mn))) 
  B <- max(c(Wn, Mn)) + .1*diff(range(c(Wn, Mn))) 
  
  udapateXi <- function(i, Thet, Y, W, M, Z, A=A, B=B){
    ie =Thet$Cik.e[i]
    iw = Thet$Cik.w[i]
    im = Thet$Cik.m[i]
    ix = Thet$Cik.x[i]
    J = ncol(Thet$X)
    
    Zmod <- Z #cbind(1,Z)
    Ytl = (Y - Zmod%*%Thet$BetaZ - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Wtl = W - Thet$Muk.w[Thet$Cik.w,]
    Mtl = (M - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
    Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
    Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
    
    #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
    Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
    
    Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))) 
    
    #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi <- cmpfun(udapateXi)
  
  
  udapateXi2A <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2A <- cmpfun(udapateXi2A)
  
  ## This approach of simulating Xi only uses W, M, and X
  
  udapateXi2 <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #Mu <- c(Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2 <- cmpfun(udapateXi2)
  
  
  udapateXi2V <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    #Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    #GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- (Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi2V <- Vectorize(udapateXi2V,"i")
  udapateXi2V <- cmpfun(udapateXi2V)
  
  #--- Write Rcpp function to speed things up
  {
    #
    
    
  }
  
  
  
  #Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)
  
  ##----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  #Nsim <- 5000
  #Nsim = 5000 
  MCMCSample <- list()
  
  #-- for X
  MCMCSample$X <- matrix(0,nrow = Nsim,ncol=length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined) 
  MCMCSample$Muk.x <- matrix(0,nrow = Nsim, ncol = Kx*pn)
  MCMCSample$Sigk.x <- matrix(0,nrow = Nsim,ncol=Kx*pn*pn)
  MCMCSample$Pik.x <- matrix(0,nrow = Nsim,ncol=Kx)
  MCMCSample$Cik.x <- matrix(0,nrow = Nsim,ncol=length(Y))
  
  #-- for W
  MCMCSample$Muk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn)
  MCMCSample$Sigk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn*pn)
  MCMCSample$Pik.w <- matrix(0,nrow = Nsim,ncol=Kw)
  MCMCSample$Cik.w <- matrix(0,nrow = Nsim,ncol=length(Y))
  
  #-- for M
  MCMCSample$Muk.m <- matrix(0,nrow = Nsim,ncol=Km*pn)
  MCMCSample$Sigk.m <- matrix(0,nrow = Nsim,ncol=Km*pn*pn)
  MCMCSample$Pik.m <- matrix(0,nrow = Nsim,ncol=Km)
  MCMCSample$Cik.m <- matrix(0,nrow = Nsim,ncol=length(Y))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow=Nsim, ncol=length(Thet$Betadel))
  
  # For Y
  MCMCSample$gamk <- matrix(0,nrow = Nsim, ncol=Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Pik.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Cik.e <- matrix(0,nrow = Nsim, ncol=length(Y))
  
  MCMCSample$Gamma <- matrix(0,nrow = Nsim, ncol=length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)
  
  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  # 
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  # 
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  # 
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  #Accept <- 
  # 
  MCMCSample$AcceptX <- matrix(0,nrow = Nsim, ncol=Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  #stop()
  #------------------------------------------------------------------
  #Nsim = 5000 
  #Nsim = 1000
  #Nsim = 3000
  #library(profvis)
  #profvis(
  #system.time(
  #cat("\n #--- MCMC is starting ----> \n")
  
  for(iter in 1:Nsim){
    if(iter %% 500 == 0){cat("iter=",iter,"\n")}
    #----------------------------------------------
    # Update things related to X 
    #--------------------------------------------  
    #--- update Pik.m for epsilon_i
    Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    
    #--- update Cik.m for epsilon_i based on mixture of normals
    Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x)) 
    
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    
    Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    Thet$InvSigk.x <- InvCov(Thet$Sigk.x)
    
    #Ytd = Y - Zmod%*%Thet$BetaZ
    if(iter%%100 == 0){
      Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B)}
    # Xn #
    #Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    #Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    #Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w, 
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)
    
    #Thet$X <- Xn #UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,]) 
    
    #}
    #dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))
    
    #Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    #Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    #Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    #Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w, 
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J))) 
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Kw)
    for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    #update the Pis
    alpha.wnew <- alpha.W/Kw + nk
    Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    
    #--- Update Ci.k mixture of normals
    #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w  
    #updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Km)
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    #update the Pis
    alpha.mnew <- alpha.m/Kw + nk
    Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    
    #--- Update Ci.k mixture of normals
    #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    #Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))  
    #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    #if(iter%%50 == 0){
    Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    MCMCSample$Betdel[iter,] <- Thet$Betadel
    #}
    #---------------------------------------------
    #cat("#--- update things related to Y or epsilon ")
    #--------------------------------------------- 
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for(k in 1:length(Thet$sig2k.e)){ nk[k] = sum(Thet$Cik.e == k)}
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n=1, alpha.enew))
    
    #update the Cik.e
    ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
    #Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) #rep(1,n) #
    #Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    #update nui  and si
    #Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    #nuisi <- updatenuisi(Thet, Y, Zmod)
    #Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    #update Sigk.e
    #Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    #update sigk and gamk
    #--- update jointly sig2ke and gamke
    #Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi)
    Thet$sig2k.e <- Temp[1,];  Thet$gamk <- Temp[2,]; Thet$p <- Temp[3,]; MCMCSample$AcceptX[iter,] <- Temp[4,]; Thet$Ak <- Temp[5,]; Thet$Bk <- Temp[6,]; Thet$Ck <- Temp[7,];
    
    #update Gamma
    #Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    #Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <-  updateSigGam(Thet, al0, bet0, order = 2)# sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#
    
    #Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    # 
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    # 
    # #--- upodate beta_{z,m}
    # #--  Prior 
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    # 
    # #--- update X_i 
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX) 
    # 
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]); 
    # 
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb 
    
    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x))  ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise
    
    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w
    
    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    #MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e
    
    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  } 
  
  #)
  #)
  
  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample
}


#--- FInal function to be run
meth = c("Wrep","FUI","Wnaiv")


FinalMeth <- function(meth, Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0 , X, g0,tunprop){
  switch(meth,
         Wrep = BQBayes.ME_SoFRFuncSimWrep(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0, X, g0, tunprop),
         FUI = BQBayes.ME_SoFRFuncSimFxFUI(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0, X, g0, tunprop),
         Wnaiv = BQBayes.ME_SoFRFuncSimFxNaiveW(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx=1, tau0, X, g0,tunprop)
         )
}




#--- This will estimate the quantile assuming W or true X (No measurrement error)
BQBayes.SoFRFunc <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0 = 0.5,X){
  
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
  met <- c("f1","f2","f3","f4","f5")
  
  
  func <- function(Pik, n=1, size=1){
    which.max(rmultinom(n=1,size=1,prob = Pik))
  }
  funcCmp <- cmpfun(func)
  
  F2 <- function(Pik, n=1, size=1){
    
    apply(Pik,1,funcCmp)
  }
  F2Cmp <- cmpfun(F2)
  
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
  
  
  
  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  #pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  #pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  pn = c(5, 6, 15, 20)[which(n == c(100, 200,500,1000))]
  alpha.e = alpx
  alpha.w = alpha.W = alpx
  alpha.x = alpha.X = alpx
  alpha.m = alpx
  pn = pn #min(15, ceiling(length(Y)^{1/3.8})+4)
  n = length(Y) 
  Zmod = cbind(1,Z)
  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt #c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) # 
  Mn_Mod <- M%*%diag(1/Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20,"knots"), 1) ))
  Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a) 
  Wn <- (W%*%bs2)/length(a)
  Xn <- (X%*%bs2)/length(a)
  
  cat("\n #--- Initial values...\n")  
  P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D 
    return(P)
  }
  
  
  InitialValues <- function(Y, W, M, Z, df0=3, pn, Gx, a){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- Z #cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    lmFit <- lm(Y ~ -1 + Zmod + W_i) 
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- residuals.lm(lmFit)
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  InitialValuesQR <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z+Wn)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    plot(a, bs2%*%lmFit$coefficients[-c(1:3)], type="l")
    
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    Tp0 = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd)), upper=c(30, 0.95*max(bnd) ))
    Res$Hess <- Tp0$hessian
    
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  #Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, W, M, Z, df0=3, pn, Gx=1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  #Kvalues
  #Kc <- 5
  
  Kx = Thet0$Kx               ## number of X  cluster
  Ke = Thet0$Ke               ## number of ei cluster
  Km = Thet0$Km               ## number of ei cluster
  Kw = Thet0$Kw              ## number of omega_i cluster 
  
  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta   #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n=pn) #as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ #rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del#0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  #Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1]); Thet$gamU <- max(bnd[2])
  
  #bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) #c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
  
  Thet$Cik.x =  sample.int(Kx,size=n, replace = T) #Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e =  sample.int(Ke,size=n, replace = T) #Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w = sample.int(Kw,size=n, replace = T) #Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m = sample.int(Km,size=n, replace = T) #Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km
  
  
  Thet$Pik.x = rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)#Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e = rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)#Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w = rdirch(n=1,alpha = rep(alpha.e, Kw)/Kw) #Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m = rdirch(n=1,alpha = rep(alpha.m, Km)/Km)#Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw
  
  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw
  
  Thet$sig2k.e  = nimble::rinvgamma(n=Ke, shape=5/2,scale=8/2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n=Ke, min = .95*Thet$gamL, max=.95*Thet$gamU)
  Thet$p <- 1*(Thet$gamk < 0) + (Thet$tau0 - 1*(Thet$gamk < 0))/(2*pnorm(-abs(Thet$gamk))*exp(0.5*Thet$gamk*Thet$gamk))
  Thet$Ak <- (1-2*Thet$p)/(Thet$p*(1-Thet$p)); Thet$Bk <- 2/(Thet$p*(1-Thet$p)); Thet$Ck <- 1/( 1*(Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean=0, sd=sqrt(Thet$sig2k.e[1])) ; Thet$nui <- rexp(n=n, rate = 1/Thet$sig2k.e[1]) 
  #Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix
  
  Thet$Sigk.w <- repmat(c(.5*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2*diag(pn)), n = Kw, m = 1)     #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw*pn), nrow = Kw)          #- vectors of length Kw and Kw respectively
  
  Thet$Sigk.x <- repmat(c(.5*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2*diag(pn)), Kx, 1)     #-- vector of length Kx
  Thet$Muk.x  <- matrix(rnorm(n=Kx*pn), nrow=Kx)   #- vectors of length Kx and Kx respectively
  
  Thet$Sigk.m <- repmat(c(.5*diag(pn)), Km, 1)     #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2*diag(pn)), Km, 1)
  Thet$Muk.m  <- matrix(rnorm(n=Km*pn), nrow = Km)   #- vectors of length Kx and Kx respectively
  
  A <- min(c(c(Wn), c(Mn))) - .5*diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5*diff(range(c(Wn, Mn)))
  #Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  #matrix(rnorm(n = n*pn), ncol=pn)
  #Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5*(Mn+Wn) #Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))
  
  
  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
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
  
  
  #method = "L-BFGS-B"
  #method = "BFGS"
  
  #Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  #Res
  
  #Res$hessian
  #Res$hessian/n
  #solve(Res$hessian/n)
  
  #LogPlostGalLik(c(.05,), c(ei), Thet)
  
  #-------- Update Beta (vector of length pn) 
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #   
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma 
  #   
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #   
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #   
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #   
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)
  
  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together) 
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 = ncol(Z) +1
  Pgam <- P.mat(pn)*tauval
  Sig0.gam0 = diag(pn0); Mu0.gam0 = numeric(pn0+pn);
  #InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))
  
  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0){
    
    #Zmod <-  Z #cbind(1,Z)
    Y.td = Y - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma))/Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$Bk[Thet$Cik.e]*Thet$nui), FUN="/") ## n*pn
    
    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam%*%(crossprod(X_sc, Y.td) + InvSig0.gam0%*%Mu0.gam0))
    
    #c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    
  }
  updateGam <- cmpfun(updateGam)
  
  #siggam ~ IG(al0, bet0)
  al0 = 0.001
  bet0 = .005
  al0 = 2.5978252 #1
  bet0 = 0.4842963 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2){
    K = length(Thet$Gamma)
    al = al0 + .5*(K - order)
    bet = .5*quad.form(P.mat(K), Thet$Gamma) + bet0
    #1/rgamma(n = 1, shape = al, rate = bet)
    val <- nimble::rinvgamma(n = 1, shape = al, scale = bet)
    ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    #tb = 1000
    #1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    #tb = .01
    #ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }
  
  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e 
  #alpha.e = 1.0
  updatePik.e <- function(Thet, alpha.e){
    
    nk <- numeric(length(Thet$Cik.e))
    for(i in 1:length(Thet$Cik.e)){ nk[i] = sum(Thet$Cik.e == i)}
    
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$Cik.e) + nk
    
    rdirichlet(n=1, alpha.enew)
    
  }
  
  #-------- Update Cik.e  
  
  #ei <- Y - Z%*%Thet$Betaz - Thet$X%*%Thet$Gamma
  updateCik.e <- function(i, Thet,ei){
    #Y - Thet$C*abs(Thet$gam)*Thet$si - Thet$A*Thet$nui
    
    #  Yi.tld <- Y[i] - c(crossprod(Thet$Gamma, Thet$X[i,]))
    #pi.k  <- Thet$Pik.e*mapply(dnorm, x = Yi.tld, mean=Thet$muk.e, sd = sqrt(Thet$sig2k.e)) + .Machine$double.eps
    pik <- Thet$Pik.e*mapply(fGal.p0, e = ei[i], sig2 = sqrt(Thet$sig2k.e), gam = Thet$gamk, p = Thet$p, tau0 = Thet$tau0) #+ .Machine$double.eps
    which(rmultinom(n=1, size=1, prob = pik) == 1)
  }
  updateCik.e <- Vectorize(updateCik.e,"i") 
  
  #--- update Si
  #-- calss id for individual i (Matrix with columns of nui and si)
  updatenui <- function(Thet, Y, Z){
    
    ki = Thet$Cik.e
    bi <- ( ((Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ck[ki]*abs(Thet$gamk[ki])*Thet$si)^2)/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])))
    ai = ((Thet$Ak[ki]*Thet$Ak[ki])/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])) + 2/sqrt(Thet$sig2k.e[ki]))
    
    #mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) 
    #val = mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
    #ifelse(val < .001, rexp(n = length(muui), rate = 1/sqrt(Thet$sig2k.e[Thet$Cik.e] )), val)
  }
  
  updatesi <- function(Thet, Y, Z){   
    #-- update si
    ki = Thet$Cik.e
    sig2si <- 1 /( 1/Thet$sig2k.e[ki] +  ((Thet$Ck[ki]*Thet$gamk[ki])^2)/ (Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui))
    muui <- sig2si*(Thet$Ck[ki]*abs(Thet$gamk[ki])*(Y - Z%*%Thet$BetaZ - Thet$X%*%Thet$Gamma - Thet$Ak[ki]*Thet$nui))/(Thet$Bk[ki]*sqrt(Thet$sig2k.e[ki])*Thet$nui)
    
    #mapply(truncnorm::rtruncnorm, n=1, a = 0.01, mean = muui, sd = sqrt(sig2si))
    mapply(truncnorm::rtruncnorm, n=1, a = 0.001, mean = muui, sd = sqrt(sig2si))
    
    
  }
  
  
  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  #a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig=5/2; b.sig = 8/2; sdsig.prop = 0.2; sdgam.prop = .3  #2.153640 1.167331
  a.sig=2.153640; b.sig = 1.167331; sdsig.prop = 0.2; sdgam.prop = .3 # 7.133445 4.303263
  a.sig=7.133445; b.sig = 4.303263; sdsig.prop = 0.2; sdgam.prop = .3 #
  updatesig_gam <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop){
    
    a0 = 0.001
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      sig_newk <- truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T) 
                   - log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- rgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gam <- Vectorize(updatesig_gam,"k")
  
  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10,10,10)*1
  varxi <- c(10,10,10)*.25
  #varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi){
    
    a0 = 0.1
    ab = 5
    #varxi = 5
    idk = which(Thet$Cik.e == k)
    if(length(idk) > 0){
      ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]),Thet$gamk[k]),sigma = Sdsig.prop*varxi[k], lb = c(a0,0.95*Thet$gamL), ub = c(ab, .95*Thet$gamU)  )
      
      
      sig_newk <- Val[1] #truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] #truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      
      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p =Thet$p[k], tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu=c(sig_newk, gam_newk), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 =Thet$tau0) + .Machine$double.eps))
                   + nimble::dinvgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
                   - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop*varxi[k], lb=c(a0, 0.95*Thet$gamL), ub=c(ab, .95*Thet$gamU), log=TRUE) )
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))
      
      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      #print(sig_newk)
      uv =  logLknew - logLkold 
      if(is.numeric(uv)){
        jump = runif(n=1); indic <- 1*(log(jump) < uv)
        Res = indic*c(sig_newk^2, gam_newk, pnew , log(jump) < uv,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) ) + (1-indic)*c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      }else{
        Res = c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k] , 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) #log(jump) <= uv
      }
      
      if(is.na(uv)||is.nan(uv)){
        sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) 
        pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
        Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
      }
      
    }else{
      sig_newk <- nimble::rinvgamma(n=1, shape = a.sig,scale = b.sig)
      gam_newk <-  runif(n = 1, min = 0.95*Thet$gamL , max = .95*Thet$gamU) #runif(n = 1, min = Thet$gamL , max = Thet$gamU) 
      pnew = 1*(gam_newk < 0) + (Thet$tau0 - 1*(gam_newk < 0))/(2*pnorm(-abs(gam_newk))*exp(0.5*gam_newk*gam_newk))
      Res = c(sig_newk^2, gam_newk, pnew , 0,  (1-2*pnew)/(pnew*(1-pnew)),  2/(pnew*(1-pnew)), 1/(1*(gam_newk > 0) - pnew) )
    }
    
    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes,"k")
  
  
  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices
  InvCov <- function(Mat){
    K <- nrow(Mat)
    J <- round(sqrt(ncol(Mat)))
    Res0 = matrix(0, nrow = K, ncol=ncol(Mat))
    for(k in 1:K){ Res0[k,] <- c(as.inverse(as.symmetric.matrix(matrix(Mat[k,],ncol=J, byrow=F)))) }
    Res0
  }
  
  
  #----------------------------------------------
  # Update things related to Wi 
  #--------------------------------------------
  #Piw
  #Cik.w
  #--- Update Pik.x  
  #alpha.W = alpha.w = 1.0
  updatePik.w <- function(Thet, alpha.W){
    Ku = nrow(Thet$Muk.w)
    nk <- numeric(Ku)
    for(k in 1:Ku){ nk[k] = sum(Thet$Cik.w == k)}
    
    #update the Pis
    alpha.wnew <- alpha.W/length(Thet$muk.w) + nk
    
    rdirichlet(n=1, alpha.wnew)
    
  }
  
  #--- Update Ci.k mixture of normals 
  dmvnormVec <- function(x, Mean, SigmaVc){
    Ku <- nrow(Mean)
    Val <- numeric(Ku)
    for(k in 1:Ku){ Val[k] <- mvtnorm::dmvnorm(x = x, mean = c(Mean[k,]), sigma = as.symmetric.matrix(matrix(SigmaVc[k,],ncol=length(x),byrow=F))) }
    Val
  }
  #dmvnormVec <- Vectorize(dmvnormVec,"k")
  #dmvnormVec(x = rep(0,6), Mn=Thet$Muk.w,SigmaVc = Thet$Sigk.w)
  
  updateCik.w <- function(i, Thet, W){ # mixture of normals 
    #Sig <- matrix(Thet$Sigk.w[id,], ncol = ncol(W), byrow = F)
    #kw = nrow(Thet$Muk.w)
    xi =  c(W[i,] - Thet$X[i,])
    #pi.k  = Thet$Pik.w*mapply(dmvnormVec, k = 1:Kw, MoreArgs = list(x = xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w))
    pi.k  = Thet$Pik.w*dmvnormVec(x=xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  
  #updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  #Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w = rep(0, pn);  Sig0.w = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol=pn, nrow=pn));inSig0.w = as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.w <- function(Thet, W, Mu0.w, inSig0.w, Sig0.w){
    Ku = length(Thet$Pik.w)
    J = ncol(Thet$X)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol=J, nrow=J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = W - Thet$X 
    #Muk <- matrix(0, ncol=J,nrow=Ku)
    SigK <- list()
    for(k in 1:Ku){
      idk <- which(Thet$Cik.w == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow= nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.w)) ## sum of matrices inverse
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi.tld[idk,])) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
        Mu.k[k,] <- c(SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.w%*%Mu0.w))      ## some of mat
        #Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(apply(t(Yi.tld[idk,]), 2, sum)) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
      }else{
        SigK[[k]] <- Sig0.w
        SigKWg <- SigKWg + (Thet$Pik.w[k]*Thet$Pik.w[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.w
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL; for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.w[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.w) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- 1.0*as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], sigma = SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.w[-Ku]) %*% SimMU.k)/Thet$Pik.w[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.w <- cmpfun(updateMuk.w)
  
  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w = pn + 2; Psi0.w = 0.5*(nu0.w-pn-1)*cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  updateSigk.w <- function(Thet, W, nu0.w, Psi0.w){
    
    Ku = nrow(Thet$Muk.w)
    m = ncol(W)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Wtl <- W - Thet$X - Thet$Muk.w[Thet$Cik.w,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.w == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.w  + nk
        Yi0 = matrix(c(Wtl[id,]), nrow= nk, byrow=F)
        Psin.w <- as.positive.definite(Psi0.w + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.w))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.w, Psi0.w))
      }
    }
    Res
  }
  updateSigk.w <- cmpfun(updateSigk.w)
  
  #----------------------------------------------
  # Update things related to M 
  #--------------------------------------------  
  
  #--- update Pik.m for epsilon_i
  #alpha.m = 1.0
  updatePik.m <- function(Thet, alpha.m){
    nk <- numeric(nrow(Thet$Muk.m))
    for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    
    #update the Pis
    alpha.mnew <- alpha.m/nrow(Thet$Muk.m) + nk
    
    c(rdirichlet(n=1, alpha.mnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  
  updateCik.m <- function(i, Thet, M){ # mixture of normals 
    #id <- Thet$Cik.m[i]
    #Sig <- matrix(Thet$Sigk.m[id,], ncol = ncol(M), byrow = F)
    Km = nrow(Thet$Muk.m)
    xi = M[i,] - Thet$Delta*Thet$X[i,]
    #pi.k  = Thet$Pik.m*mapply(dmvnormVec, k = 1:Km, MoreArgs = list(x = xi , Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m))
    pi.k  = Thet$Pik.m*dmvnormVec(x = xi, Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m = rep(0, pn);  Sig0.m = .5*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.m = as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.m <- function(Thet, M, Mu0.m, inSig0.m, Sig0.m){
    #cat(dim(inSig0.m),"(--1)\n")
    Ku = length(Thet$Pik.m)
    J = ncol(M)  
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list()  #-- store the covariances
    SigKWg <- matrix(0, ncol = J, nrow = J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = M - Thet$X%*%diag(Thet$Betadel) 
    Muk <- matrix(0, ncol=J, nrow=Ku)
    for(k in 1:Ku){
      idk <- which(Thet$Cik.m == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 = matrix(c(Yi.tld[idk,]), nrow = nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[k,],ncol=J,byrow=F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nk*TpSi + inSig0.m)) ## sum of matrices inverse
        #cat(dim(SigK[[k]]),"(--2 \n")
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi0)) ) + inSig0.m%*%Mu0.m)      ## some of mat
        #cat(dim(SigKWg),"(--3 \n")
      }else{
        SigK[[k]] <- Sig0.m
        SigKWg <- SigKWg + (Thet$Pik.m[k]*Thet$Pik.m[k])*SigK[[k]] # J x J
        Mu.k[k,] <- Mu0.m
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL;  for(k in 1:Ku){SIGR0 <- rbind(SIGR0,Thet$Pik.m[k]*SigK[[k]])} ## K*J x J
    MUR0 <-  colSums(diag(Thet$Pik.m) %*% Mu.k) # return a evector of length J
    
    MURFin <- c(t(Mu.k)) - SIGR0%*%solve(SigKWg)%*%MUR0 ## J*K
    SIGR0fin <- as.matrix(nearPD(SIg0 - SIGR0%*%solve(SigKWg)%*%t(SIGR0),doSym = T)$mat) ## J*K x J*K
    
    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J*(Ku-1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n=1, mean = MURFin[id], SIGR0fin[id,id])), ncol=J,byrow=T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.m[-Ku]) %*% SimMU.k)/Thet$Pik.m[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k,SimMU.k1)) ## k * J
  }
  updateMuk.m <- cmpfun(updateMuk.m)
  
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m = pn + 2; Psi0.m = .5*(nu0.m-pn-1)*as.symmetric.matrix(cov(Wn))  #diag(pn)
  
  updateSigk.m <- function(Thet, M, nu0.m, Psi0.m){
    
    Ku = nrow(Thet$Muk.m)
    m = ncol(M)
    al <- numeric(Ku)
    bet <- numeric(Ku)
    Mtl <- M - Thet$X%*%diag(Thet$Betadel) - Thet$Muk.m[Thet$Cik.m,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.m == i)
      nk = length(id)
      if(nk > 0){
        nun <- nu0.m  + nk
        Yi0 <- matrix(c(Mtl[id,]), nrow=nk, byrow=F)
        Psin.m <- as.positive.definite(Psi0.m + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.m))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.m, Psi0.m))
      }
    }
    Res
  }
  updateSigk.m <- cmpfun(updateSigk.m)
  
  #---- Update things related to delta
  quadFormVec <- function(i,X,sigma,Val){
    id = Val[i]
    as.symmetric.matrix(quad.form(as.inverse(as.symmetric.matrix(matrix(sigma[id,], ncol = length(X[i,]),byrow=F))), diag(c(X[i,])))) 
  }
  quadFormVec <- Vectorize(quadFormVec,"i",SIMPLIFY = F) ## Resulting in a list if precision matrices
  quadFormVec <- cmpfun(quadFormVec)
  
  quadFormVec3 <- function(i,Mi,X,sigma,Val){
    id = Val[i]
    c(quad.3form.inv(matrix(sigma[id,], ncol = length(X[i,]),byrow=F), diag(X[i,]),Mi[i,]))
  }
  quadFormVec3 <- Vectorize(quadFormVec3,"i", SIMPLIFY = F)
  quadFormVec3 <- cmpfun(quadFormVec3)
  
  #SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),   
  # prior values
  mu0.del = 1.0; #sig02.del = 100
  updateDelta <- function(Thet, M, mu0.del, sig02.del){
    n = nrow(M)
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X), Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    sig20 <- 1/( 1/sig02.del + sum(Res))
    mu <- sig20*(sum(Res3) + mu0.del/sig02.del)
    rtruncnorm(n=1, a=0, b=10, mean = mu, sd = sqrt(sig20))
    
    #c(sum(Res3), sum(Res),mu, sqrt(sig20), rtruncnorm(n=1, a=0, b=Inf, mean = mu, sd = sqrt(sig20)))
    
  }
  updateDelta <- cmpfun(updateDelta)
  
  
  # prior values
  mu0.del = rep(0,length(Thet$Betadel)); #sig02.del = 20
  Sig0.del = P.mat(length(Thet$Betadel))*.1   #Thet$siggam) # Inverse or precision covariance matrix
  
  updateFunDelta <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDelta <- cmpfun(updateFunDelta)
  
  
  SimDElVec <- function(i, Mi, X, Sigma, Val){
    id <- Val[i]
    Ot = mvtnorm::rmvnorm(n = 1,mean = Mi[i,],sigma = as.symmetric.matrix(matrix(Sigma[id,], ncol = length(X[i,]),byrow=F)))
  }
  SimDElVec <- Vectorize(SimDElVec, "i")
  
  #-- 
  DeltComp <- function(k,Xst, X){
    abs(sum(Xst[,k]*X[,k])/(sum(X[,k]^2)))
    #abs(diag(as.inverse(crossprod(X))%*%crossprod(X,Xst)))
  }
  DeltComp <- Vectorize(DeltComp,"k")
  
  updateFunDeltaSt <- function(Thet, M, mu0.del, Sig0.del){
    Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
    Xst <- base::t(SimDElVec(1:nrow(M), Mitl, Thet$X, Thet$Sigk.m, Thet$Cik.m)) ## K by n matrix and now I get a n by K
    c(DeltComp(1:ncol(Thet$X), Xst, Thet$X))
    #c(DeltComp(Xst, Thet$X))
    
    # Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
    # Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
    # Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
    # Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
    # c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
  }
  updateFunDeltaSt  <- cmpfun(updateFunDeltaSt )
  
  
  #----------------------------------------------
  # Update things related to X 
  #--------------------------------------------------  
  #--- update Pik.m for epsilon_i
  #alpha.x <- 1.0
  updatePik.x <- function(Thet, alpha.x){
    nk <- numeric(nrow(Thet$Muk.x))
    for(k in 1:nrow(Thet$Muk.x)){ nk[k] = sum(Thet$Cik.x == k)}
    
    #update the Pis
    alpha.xnew <- alpha.x/nrow(Thet$Muk.x) + nk
    
    c(rdirichlet(n=1, alpha.xnew))
  }
  
  #--- update Cik.m for epsilon_i based on mixture of normals
  updateCik.x <- function(i, Thet){ # mixture of normals 
    #Kx = ncol(Thet$X) # J x J
    #id <- Thet$Cik.x[i]
    #Sig <- matrix(Thet$Sigk.x[id,], ncol = length(Thet$Gamma), byrow = F)
    Xi = Thet$X[i,]
    #pi.k  = Thet$Pik.x*mapply(dmvnormVec, k = 1:Kx, x = Xi,  Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x)
    pi.k  = Thet$Pik.x*dmvnormVec(x=Xi,Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x) + .Machine$double.eps
    which(rmultinom(n=1,size=1,prob = pi.k) == 1)
  }
  #updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x = numeric(pn); Sig0.x = .5*diag(pn); inSig0.x = 2*diag(pn)#as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x = rep(0, pn);  Sig0.x = .25*as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w)+colMeans(Thet0$Sigk.m), ncol=pn, nrow=pn));inSig0.x = as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.x <- function(Thet, Mu0.x, inSig0.x, Sig0.x){
    K = length(Thet$Pik.x)
    J = ncol(Thet$X)
    Res0 <- matrix(0, nrow = K, ncol = J)
    Mu.k <- matrix(0, nrow = K, ncol = J) # K x J
    #SigKWg <- matrix(0, ncol = ncol(Y),nrow=ncol(Y)) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld = Thet$X 
    Muk <- matrix(0, ncol = J, nrow = K)
    for(k in 1:K){
      idk <- which(Thet$Cik.x == k)
      nk <- length(idk)
      if(length(idk) > 0){
        Yi0 <- matrix(c(Yi.tld[idk,]), nrow=nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[k,], ncol=J, byrow=F)))
        SigK <- as.inverse(as.symmetric.matrix((nk*TpSi + inSig0.x))) ## sum of matrices inverse
        Muk <- SigK%*%(crossprod(TpSi, c(colSums(Yi0))) + inSig0.x%*%Mu0.x)      ## some of mat
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Muk, sigma = SigK))
      }else{
        Mu.k[k,] <- c(mvtnorm::rmvnorm(n=1, mean = Mu0.x, sigma = Sig0.x ))
      }
    }
    #--- we impose some constrainsts on the mean
    Mu.k
    
  }
  updateMuk.x <- cmpfun(updateMuk.x)
  
  nu0.x <- pn + 2 ; Psi0.x <- .5*diag(pn) #as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2 ; Psi0.x <- .5*(nu0.x-pn-1)*cov(.5*(Wn+Mn)) #as.symmetric.matrix(.25*cov(Wn))
  updateSigk.x <- function(Thet, nu0.x, Psi0.x){
    
    Ku = nrow(Thet$Muk.x)
    m = ncol(Thet$Muk.x)
    Xtl <- Thet$X - Thet$Muk.x[Thet$Cik.x,]
    Res <- matrix(0,ncol = m*m ,nrow = Ku)
    for(i in 1:Ku){
      id <- which(Thet$Cik.x == i)
      nk = length(id)
      if(nk > 0){
        nux <- nu0.x  + nk
        Yi0 <- matrix(c(Xtl[id,]), nrow=nk, byrow=F)
        Psin.x <- Psi0.x + crossprod(Yi0)
        Res[i, ] <- c(rinvwishart(nux, Psin.x))
      }else{
        Res[i, ] <- c(rinvwishart(nu0.x, Psi0.x))
      }
    }
    Res
  }
  updateSigk.x <- cmpfun(updateSigk.x)
  # 
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn))) 
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn))) 
  
  A <- min(c(Wn, Mn)) - .1*diff(range(c(Wn, Mn))) 
  B <- max(c(Wn, Mn)) + .1*diff(range(c(Wn, Mn))) 
  
  udapateXi <- function(i, Thet, Y, W, M, Z, A=A, B=B){
    ie =Thet$Cik.e[i]
    iw = Thet$Cik.w[i]
    im = Thet$Cik.m[i]
    ix = Thet$Cik.x[i]
    J = ncol(Thet$X)
    
    Zmod <- Z #cbind(1,Z)
    Ytl = (Y - Zmod%*%Thet$BetaZ - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Wtl = W - Thet$Muk.w[Thet$Cik.w,]
    Mtl = (M - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    
    Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
    Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
    Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
    
    #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
    Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
    
    Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))) 
    
    #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
    c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi <- cmpfun(udapateXi)
  
  
  udapateXi2A <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2A <- cmpfun(udapateXi2A)
  
  ## This approach of simulating Xi only uses W, M, and X
  
  udapateXi2 <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #Mu <- c(Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2 <- cmpfun(udapateXi2)
  
  
  udapateXi2V <- function(Thet, Y, Zmod, Wn, Mn, A=A, B=B){
    
    J = ncol(Thet$X)
    n = length(Y)
    #Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    #Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
    Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]
    Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res = matrix(0, ncol=J, nrow = n)
    Sigw = matrix(0, J, J)
    Sigm = matrix(0, J, J)
    Sigx = matrix(0, J, J)
    #GamMat= tcrossprod(c(Thet$Gamma))
    
    for(i in 1:n){
      ie =Thet$Cik.e[i]
      iw = Thet$Cik.w[i]
      im = Thet$Cik.m[i]
      ix = Thet$Cik.x[i]
      
      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#
      
      Sigw <- matrix(Thet$InvSigk.w[iw,],ncol=J, byrow=F)
      Sigm <- matrix(Thet$InvSigk.m[im,],ncol=J, byrow=F)
      Sigx <- matrix(Thet$InvSigk.x[ix,],ncol=J, byrow=F)
      
      #Thet$InvSigk.m
      
      #Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      #Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel)) 
      
      Mu <- (Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,]))); 
      
      #c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      #Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i,] = c(TruncatedNormal::rtmvnorm(n=1, mu = Mu, sigma =  Sig,lb = rep(A,J), ub = rep(B,J)))
    }
    Res
    
  }
  #udapateXi2V <- Vectorize(udapateXi2V,"i")
  udapateXi2V <- cmpfun(udapateXi2V)
  
  #--- Write Rcpp function to speed things up
  {
    #
    
    
  }
  
  
  
  #Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)
  
  ##----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  #Nsim <- 5000
  #Nsim = 5000 
  MCMCSample <- list()
  
  #-- for X
  MCMCSample$X <- matrix(0,nrow = Nsim,ncol=length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined) 
  MCMCSample$Muk.x <- matrix(0,nrow = Nsim, ncol = Kx*pn)
  MCMCSample$Sigk.x <- matrix(0,nrow = Nsim,ncol=Kx*pn*pn)
  MCMCSample$Pik.x <- matrix(0,nrow = Nsim,ncol=Kx)
  MCMCSample$Cik.x <- matrix(0,nrow = Nsim,ncol=length(Y))
  
  #-- for W
  MCMCSample$Muk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn)
  MCMCSample$Sigk.w <- matrix(0,nrow = Nsim,ncol=Kw*pn*pn)
  MCMCSample$Pik.w <- matrix(0,nrow = Nsim,ncol=Kw)
  MCMCSample$Cik.w <- matrix(0,nrow = Nsim,ncol=length(Y))
  
  #-- for M
  MCMCSample$Muk.m <- matrix(0,nrow = Nsim,ncol=Km*pn)
  MCMCSample$Sigk.m <- matrix(0,nrow = Nsim,ncol=Km*pn*pn)
  MCMCSample$Pik.m <- matrix(0,nrow = Nsim,ncol=Km)
  MCMCSample$Cik.m <- matrix(0,nrow = Nsim,ncol=length(Y))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow=Nsim, ncol=length(Thet$Betadel))
  
  # For Y
  MCMCSample$gamk <- matrix(0,nrow = Nsim, ncol=Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Pik.e <- matrix(0,nrow = Nsim, ncol=Ke)
  MCMCSample$Cik.e <- matrix(0,nrow = Nsim, ncol=length(Y))
  
  MCMCSample$Gamma <- matrix(0,nrow = Nsim, ncol=length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)
  
  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  # 
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  # 
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  # 
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  #Accept <- 
  # 
  MCMCSample$AcceptX <- matrix(0,nrow = Nsim, ncol=Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  #stop()
  #------------------------------------------------------------------
  #Nsim = 5000 
  #Nsim = 1000
  #Nsim = 3000
  #library(profvis)
  #profvis(
  #system.time(
  #cat("\n #--- MCMC is starting ----> \n")
  
  for(iter in 1:Nsim){
    if(iter %% 500 == 0){cat("iter=",iter,"\n")}
    # #----------------------------------------------
    # # Update things related to X 
    # #--------------------------------------------  
    # #--- update Pik.m for epsilon_i
    # Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    # 
    # #--- update Cik.m for epsilon_i based on mixture of normals
    # Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    # #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x)) 
    # 
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    # 
    # Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    # Thet$InvSigk.x <- InvCov(Thet$Sigk.x)
    # 
    # #Ytd = Y - Zmod%*%Thet$BetaZ
    # #if(iter%%10 == 0){
    # #Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B) # Xn #
    # #Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    # #Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    # #Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w, 
    # #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)
    # 
    # Thet$X <- UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,]) 
    # 
    # #}
    # #dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))
    # 
    # #Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    # #Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    # #Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    # 
    # #Thet$X <-
    # #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w, 
    # #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J))) 
    # #---------------------------------------------
    # #--- update things related to W
    # #---------------------------------------------
    # #--- Update Pik.w  
    # #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Kw)
    # for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    # #update the Pis
    # alpha.wnew <- alpha.W/Kw + nk
    # Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    # 
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    # Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    # Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    # Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    # #---------------------------------------------
    # #--- update things related to M
    # #---------------------------------------------
    # #--- Update Pik.w  
    # #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Km)
    # for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    # #update the Pis
    # alpha.mnew <- alpha.m/Kw + nk
    # Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    # 
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    # Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))  
    # #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    # Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    # Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # #if(iter%%50 == 0){
    # Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    # MCMCSample$Betdel[iter,] <- Thet$Betadel
    # #}
    #---------------------------------------------
    #cat("#--- update things related to Y or epsilon ")
    #--------------------------------------------- 
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for(k in 1:length(Thet$sig2k.e)){ nk[k] = sum(Thet$Cik.e == k)}
    #update the Pis
    alpha.enew <- alpha.e/length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n=1, alpha.enew))
    
    #update the Cik.e
    ei <- Y - Zmod%*%Thet$BetaZ - Thet$X%*%Thet$Gamma
    #Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) #rep(1,n) #
    #Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    #update nui  and si
    #Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    #nuisi <- updatenuisi(Thet, Y, Zmod)
    #Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    #update Sigk.e
    #Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    #update sigk and gamk
    #--- update jointly sig2ke and gamke
    #Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke,Thet, Y, Zmod,  a.sig=a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi=varxi)
    Thet$sig2k.e <- Temp[1,];  Thet$gamk <- Temp[2,]; Thet$p <- Temp[3,]; MCMCSample$AcceptX[iter,] <- Temp[4,]; Thet$Ak <- Temp[5,]; Thet$Bk <- Temp[6,]; Thet$Ck <- Temp[7,];
    
    #update Gamma
    #Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    #Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <- sig02.del# updateSigGam(Thet, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#
    
    #Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    # 
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    # 
    # #--- upodate beta_{z,m}
    # #--  Prior 
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    # 
    # #--- update X_i 
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX) 
    # 
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]); 
    # 
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb 
    
    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x))  ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise
    
    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w
    
    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    #MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e
    
    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  } 
  
  #)
  #)
  
  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample
}




#-- Function choices to run these in parallel

FuncChoic <- function(Val, Y, W, M, Z, a, Nsim, sig02.del, tauval,  Delt, alpx, tau0, X){
  switch(Val,
         "sim" = BQBayes.ME_SoFRFuncSim(Y, W, M, Z, a, Nsim, sig02.del, tauval,  Delt, alpx, tau0 ,X),
         "ind" = BQBayes.ME_SoFRFuncIndiv(Y, W, M, Z, a, Nsim, sig02.del, tauval,  Delt, alpx, tau0 ,X)
         )
}


#-------------------------------------------------------------------
#-------------------------------------------------------------------
#--- Data simulation 
#-------------------------------------------------------------------
#-------------------------------------------------------------------


#--- Basis function
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


ar1_cor <- function(n, rho) {
  exponent <- abs(matrix(1:n - 1, nrow = n, ncol = n, byrow = TRUE) - 
                    (1:n - 1))
  rho^exponent
}
#--- Distributon of errors for epsilon


#--  Options are: distF = "Rst", "Norm","MixNorm", "Gam","Gam2"
DistEps <- function(n, distF, sig.e){
  pi.g <- .5
  ei = switch(distF,
              "Gam2" = rgamma(n=n,shape=1,scale=1.5) - 1.5,
              "Gam" = rgamma(n=n,shape=1,scale=1.5) - 1.5,
              "Rst"   = sn::rst(n = n, xi = 0, omega = .5, nu = 5, alpha = 2),
              "Norm" = rnorm(n=n, sd = sig.e),
              #"MixNorm" = Prob0*rnorm(n=n, mean=-1.75, sd=.5*sig.e) + (1 - Prob0)*rnorm(n=n, mean=1.75*sig.e, sd=.5*sig.e)
              "MixNorm" = rnormmix(n=n, lambda = rep(pi.g,2), mu = c(-2.0,2.0), sigma = (rep(.75*sig.e,2)))
              #Prob0*rnorm(n=n, mean=-2., sd=.5) + (1 - Prob0)*rnorm(n=n, mean=2., sd=.5)
  )
  ei
}

#--- Distributon of errors for U (the measurement error)
#@options are : "Rmst" and "Mvnorm"
DistUw <- function(n,distF, Sig){
  p0 <- nrow(Sig)
  p10 = round(0.5*p0); p11 <- p0 - p10
  
  ei = switch(distF,
              "Rmst" =  sn::rmst(n=n, xi = c(rep(-2,p10), rep(2,p11)), Omega = Sig, alpha = rep(2,p0), nu= 5),
              "Mvnorm" = mvtnorm::rmvnorm(n=n,mean = rep(0,nrow(Sig)), sigma = Sig)
              #"MixNorm" = Prob0*rnorm(n=n, mean=-1.75, sd=.5*sig.e) + (1 - Prob0)*rnorm(n=n, mean=1.75*sig.e, sd=.5*sig.e)
              #"MixNorm" = rnormmix(n=n, lambda = rep(pi.g,2), mu = c(-2.0,2.0), sigma = (rep(sig.e,2)))
              #Prob0*rnorm(n=n, mean=-2., sd=.5) + (1 - Prob0)*rnorm(n=n, mean=2., sd=.5)
  )
  scale(ei, center = T, scale = F)
}

#---- Cov matrix type
#@Met = CS, Ar1
CovMat <- function(m, Met,sig,rho){
  
  ei = switch(Met,
              "CS" =   (ones(m,m) - diag(rep(1,m)))*((sig^2)*rho) + diag(rep(sig^2,m)),
              "Ar1" = ar1_cor(m, rho)*(sig^2) #CVTuningCov::AR1 
              #"MixNorm" = Prob0*rnorm(n=n, mean=-1.75, sd=.5*sig.e) + (1 - Prob0)*rnorm(n=n, mean=1.75*sig.e, sd=.5*sig.e)
              #"MixNorm" = rnormmix(n=n, lambda = rep(pi.g,2), mu = c(-2.0,2.0), sigma = (rep(sig.e,2)))
              #Prob0*rnorm(n=n, mean=-2., sd=.5) + (1 - Prob0)*rnorm(n=n, mean=2., sd=.5)
  )
  ei
}


#---  Beta function
Betafunc <- function(t, met){
  switch(met,
         f1 = 1.*sin(2*pi*t) + 1.5,
         f2 = 1/(1+exp(4*2*(t-.5))),
         f3 = 1.*sin(pi*(8*(t-.5))/2)/ (1 + (2*(8*(t-.5))^2)*(sign(8*(t-.5))+1)) + 1.5,
         f4 = sin(pi*(16*(t-.5))/2)/ (1 + (2*(16*(t-.5))^2)*(.5*sin(8*(t-.5))+1)),
         f5 = sin(pi*(16*(t-.5))/2)/ (1 + (2*(16*(t-.5))^2)*(sin(8*(t-.5))+1))
  )
}

#-- Simulate data with constant variance
DataSim <- function(pn, n, t, a, j0 = 2, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep=c("Rst", "Norm","MixNorm"), distFUe = c("Rmst","Mvnorm"), distFepWe=c("Rmst","Mvnorm"), CovMet =c("CS", "Ar1"), sig.e, idf=c(1:4)){
  
  
  #Nbasis0 =  c(5, 6, 15, 15) # pn = [which(n == c(100, 200,500,1000))]
  #J = K = k  = pn = pn00 =  Nbasis0[which(c(100,200,500, 1000) == n)]
  a <- seq(0, 1, length.out=t) ## time points at which obs. are takes
  
  met <- c("f1","f2","f3","f4","f5")
  
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
  
  
  P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D 
    return(P)
  }
  
  #Deltafun <- function(a, scale, loc){ (.25 - (a - .5)^2)*scale + rep(loc,length(a))}
  Deltafun <- function(a, scale, loc){  (1.25 + Betafunc(a, met[idf]))*scale + rep(loc,length(a))}
  
  Delt <- Deltafun(a, scale=1.,loc=.5)
  # P.mat <- function(K){
  #   return(ppls::Penalty.matrix(K,order=2))
  # }
  par(mfrow=c(2,2))
  plot(a,Betafunc(a,  met[idf]), type="l")
  plot(a, Delt, type="l")
  
  #P.mat(3)
  
  #---------------------------------------------------------------------
  #---- Simulate X
  #---------------------------------------------------------------------
  #sig.x = .5
  SigX <- CovMat(t, CovMet,sig.x,rhox) #ones(t,t)*rhox*sig.x*sig.x; diag(SigX) <- (sig.x^2) 
  MeanX <- Betafunc(a, met[idf]) #sinpi(2*a)
  MeanX2 <- Betafunc(a, met[idf]) #sinpi(2*a)
  X_t <- matrix(MeanX, ncol=t, nrow=n, byrow = T) + DistUw(n,distFUe, SigX)  #rbind(mvtnorm::rmvnorm(n=n*.5,mean = MeanX, sigma = SigX), mvtnorm::rmvnorm(n=n*.5,mean = MeanX2, sigma = SigX))
  plot(a,MeanX, type="l",lwd=3)
  
  # We generate the basis functions from bsplines 
  a   <- seq(0, 1,length.out=t)
  bs2 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0,attr(bs20,"knots"), 1) ))
  #bs2 <- B.basis(a, knots= seq(0,1, length.out = pn-1))
  
  # We generate Yi the scalar response with error terms from Normal (0,.05) (Equation (1)) 
  #if(distF == "Norm"){
  pi.g <- .5
  Prob0 <- rbinom(n=n,size=1,prob = pi.g)
  
  Z <- matrix(rnorm(n=n*j0, sd = 1), ncol = j0); Z <- scale(Z); 
  #Z[,2] <- c(mapply(rnorm, n=1, mean = crossprod(t(X_t),Betafunc(a,met[idf]))/length(a))); Z <- scale(Z); 
  
  Betaz <- rnorm(n = j0+1); 
  Betaz <- c( -1.05, .57)*c(1, .35,.35)[idf] #c(0., -1.05, .57) # c(1.87, -1.05, .57)
  Zmod <- Z #cbind(1, Z)
  Val =1 #<- caseN %in%paste("case",1:4,"a",sep="")
  #*(1-Val)
  ei <- DistEps(n, distFep, sig.e)*c(1,1,.5, 1)[idf]
  BetatF <- (Betafunc(a,met[idf]) - mean(Betafunc(a,met[idf])))*c(1,2,2)[idf]
  fx0 = crossprod(t(X_t),BetatF)/length(a)
  Y <- Zmod%*%Betaz*Val + fx0  + ei*fx0  #*(crossprod(t(X_t),Betafunc(a,met[idf]))/length(a)) #(ei-mean(ei))  
  # #d0*rnorm(n, sd = sig.e) + (1-d0)*(rgamma(n=n,shape=1,scale=1.5) - 1.5) # (n,1) Y matrix
  
  
  #f0 <- function(t) { X_t[1,]*Betafunc(t,met[idf])}
  par(mfrow=c(2,2))
  MASS::truehist(Y)
  MASS::truehist(ei)
  MASS::truehist(c(Zmod%*%Betaz))
  MASS::truehist(fx0)
  
  #print(mean(Zmod%*%Betaz*Val + crossprod(t(X_t),Betafunc(a,met[idf]))/length(a))/sd((ei-mean(ei))))
  print(mean(Zmod%*%Betaz*Val + fx0)/sd((ei-mean(ei))))
  
  
  #Y <- 0 + crossprod(t(X_t),Betafunc(a,met[idf]))/length(a) +  rgamma(n=n,shape=1,scale=1.5) - 1.5 # (n,1) Y matrix
  #---------------------------------------------------------------------
  # We generate the observed surrogates values W with errors from the Skew Normal distribution (Equation (2))
  #---------------------------------------------------------------------
  SigU <- CovMat(t, CovMet,sig.u,rho.u)#ones(t,t)*rho.u*sig.u*sig.u; diag(SigU) <- (sig.u^2) 
  MeanU <- numeric(t)
  U <- DistUw(n,distFUe, SigU)#mvtnorm::rmvnorm(n=n,mean = MeanU, sigma = SigU)
  W = X_t + U  # (n,t) W matrix of surrogate values
  Wn <- (W%*%bs2)/length(a)
  
  #---------------------------------------------------------------------
  # We generate the IV with errors from the Normal distribution (Equation (3))
  #---------------------------------------------------------------------
  SigW <- ones(t,t)*rho.m*sig.w*sig.w; diag(SigW) <- (sig.w^2) 
  MeanW <- numeric(t)
  omega <- DistUw(n,distFUe, SigW) #mvtnorm::rmvnorm(n=n,mean = MeanW, sigma = SigW)
  
  M <- X_t%*%diag(Delt) + omega  # (n,t) M matrix
  # Mn <- (M%*%bs2)/length(a)
  #Delthat <- rep(1, length(Delt))#Delt # lowess(a, (colMeans(M)/colMeans(W)), f=1/6)$y # c(abs(colMeans(M)/(colMeans(W)))) #   lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) # 
  #plot(a, Delthat, type="l")
  Mn_Mod <- M%*%diag(1/Delt)
  #bs2 <- bs(a, df = pn, intercept = T)
  Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a) 
  Wn <- (W%*%bs2)/length(a)
  
  Data <- list(a = a, bs2 = bs2, Z = Z,  Y = Y, X = X_t, DataN=list(Mn = Mn, Wn = Wn, Xn = (X_t%*%bs2)/length(a) ), 
               Data=list(M= M, W = W, X = X_t), Delt = Delt, BetaZ = Betaz, Betat = BetatF)
  Data
}


#-- Simulate data with constant variance
DataSimHeter <- function(pn, n, t, a, j0 = 2, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep=c("Rst", "Norm","MixNorm"), distFUe = c("Rmst","Mvnorm"), distFepWe=c("Rmst","Mvnorm"), CovMet =c("CS", "Ar1"), sig.e, idf=c(1:4)){
  
  
  #Nbasis0 =  c(5, 6, 15, 15) # pn = [which(n == c(100, 200,500,1000))]
  #J = K = k  = pn = pn00 =  Nbasis0[which(c(100,200,500, 1000) == n)]
  a <- seq(0, 1, length.out=t) ## time points at which obs. are takes
  
  met <- c("f1","f2","f3","f4","f5")
  
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
  
  
  P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D 
    return(P)
  }
  
  #Deltafun <- function(a, scale, loc){ (.25 - (a - .5)^2)*scale + rep(loc,length(a))}
  Deltafun <- function(a, scale, loc){  (1.25 + Betafunc(a, met[idf]))*scale + rep(loc,length(a))}
  
  Delt <- Deltafun(a, scale=1.,loc=.5)
  # P.mat <- function(K){
  #   return(ppls::Penalty.matrix(K,order=2))
  # }
  par(mfrow=c(2,2))
  plot(a,Betafunc(a,  met[idf]), type="l")
  plot(a, Delt, type="l")
  
  #P.mat(3)
  
  #---------------------------------------------------------------------
  #---- Simulate X
  #---------------------------------------------------------------------
  #sig.x = .5
  SigX <- CovMat(t, CovMet,sig.x,rhox) #ones(t,t)*rhox*sig.x*sig.x; diag(SigX) <- (sig.x^2) 
  MeanX <- Betafunc(a, met[idf]) #sinpi(2*a)
  MeanX2 <- Betafunc(a, met[idf]) #sinpi(2*a)
  X_t <- matrix(MeanX, ncol=t, nrow=n, byrow = T) + DistUw(distFUe, SigX)  #rbind(mvtnorm::rmvnorm(n=n*.5,mean = MeanX, sigma = SigX), mvtnorm::rmvnorm(n=n*.5,mean = MeanX2, sigma = SigX))
  plot(a,MeanX, type="l",lwd=3)
  
  # We generate the basis functions from bsplines 
  a   <- seq(0, 1,length.out=t)
  bs2 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0,attr(bs20,"knots"), 1) ))
  #bs2 <- B.basis(a, knots= seq(0,1, length.out = pn-1))
  
  # We generate Yi the scalar response with error terms from Normal (0,.05) (Equation (1)) 
  #if(distF == "Norm"){
  pi.g <- .5
  Prob0 <- rbinom(n=n,size=1,prob = pi.g)
  
  Z <- matrix(rnorm(n=n*j0, sd = 1), ncol = j0,); Z <- scale(Z); 
  #Z[,2] <- c(mapply(rnorm, n=1, mean = crossprod(t(X_t),Betafunc(a,met[idf]))/length(a))); Z <- scale(Z); 
  
  Betaz <- rnorm(n = j0+1); 
  Betaz <- c( -1.05, .57)*.35 #c(0., -1.05, .57) # c(1.87, -1.05, .57)
  Zmod <- Z #cbind(1, Z)
  Val =1 #<- caseN %in%paste("case",1:4,"a",sep="")
  #*(1-Val)
  ei <- DistEps(n, distFep, sig.e)
  BetatF <- (Betafunc(a,met[idf]) - mean(Betafunc(a,met[idf])))*2
  fx0 = crossprod(t(X_t),BetatF)/length(a)
  Ei = ei*fx0
  Y <- Zmod%*%Betaz*Val + fx0  + .5*Ei  #*(crossprod(t(X_t),Betafunc(a,met[idf]))/length(a)) #(ei-mean(ei))  
  # #d0*rnorm(n, sd = sig.e) + (1-d0)*(rgamma(n=n,shape=1,scale=1.5) - 1.5) # (n,1) Y matrix
  
  
  #f0 <- function(t) { X_t[1,]*Betafunc(t,met[idf])}
  par(mfrow=c(2,2))
  MASS::truehist(Y)
  MASS::truehist(.5*ei*fx0)
  MASS::truehist(c(Zmod%*%Betaz))
  MASS::truehist(fx0)
  
  #print(mean(Zmod%*%Betaz*Val + crossprod(t(X_t),Betafunc(a,met[idf]))/length(a))/sd((ei-mean(ei))))
  print(mean(Zmod%*%Betaz*Val + fx0)/sd((ei-mean(ei))))
  
  
  #Y <- 0 + crossprod(t(X_t),Betafunc(a,met[idf]))/length(a) +  rgamma(n=n,shape=1,scale=1.5) - 1.5 # (n,1) Y matrix
  #---------------------------------------------------------------------
  # We generate the observed surrogates values W with errors from the Skew Normal distribution (Equation (2))
  #---------------------------------------------------------------------
  SigU <- CovMat(t, CovMet,sig.u,rho.u)#ones(t,t)*rho.u*sig.u*sig.u; diag(SigU) <- (sig.u^2) 
  MeanU <- numeric(t)
  U <- DistUw(distFUe, SigU)#mvtnorm::rmvnorm(n=n,mean = MeanU, sigma = SigU)
  W = X_t + U  # (n,t) W matrix of surrogate values
  Wn <- (W%*%bs2)/length(a)
  
  #---------------------------------------------------------------------
  # We generate the IV with errors from the Normal distribution (Equation (3))
  #---------------------------------------------------------------------
  SigW <- ones(t,t)*rho.m*sig.w*sig.w; diag(SigW) <- (sig.w^2) 
  MeanW <- numeric(t)
  omega <- DistUw(distFUe, SigU) #mvtnorm::rmvnorm(n=n,mean = MeanW, sigma = SigW)
  
  M <- X_t%*%diag(Delt) + omega  # (n,t) M matrix
  # Mn <- (M%*%bs2)/length(a)
  #Delthat <- rep(1, length(Delt))#Delt # lowess(a, (colMeans(M)/colMeans(W)), f=1/6)$y # c(abs(colMeans(M)/(colMeans(W)))) #   lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) # 
  #plot(a, Delthat, type="l")
  Mn_Mod <- M%*%diag(1/Delt)
  #bs2 <- bs(a, df = pn, intercept = T)
  Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a) 
  Wn <- (W%*%bs2)/length(a)
  
  Data <- list(a = a, bs2 = bs2, Z = Z,  Y = Y, X = X_t, DataN=list(Mn = Mn, Wn = Wn, Xn = (X_t%*%bs2)/length(a) ), Data=list(M= M, W = W, X = X_t), Delt = Delt, BetaZ = Betaz, Betat = BetatF)
  Data
}


#--- Dataset with replicated obs
#-- Simulate data with constant variance

# Z <- matrix(rnorm(n=n*(j0-1), sd = 1), ncol = c(j0-1)); Z <- scale(Z); 
# Z <- cbind(Z, rbinom(n = n, prob=pi.g, size=1))

DataSimReplicate <- function(pn, n, t, a, j0 = 2, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep=c("Rst", "Norm","MixNorm"), distFUe = c("Rmst","Mvnorm"), distFepWe=c("Rmst","Mvnorm"), CovMet =c("CS", "Ar1"), sig.e, idf=c(1:4), nrep, Wdist,Z){
  #@parms Wdist = "norm" or "count"  
  #Nbasis0 =  c(5, 6, 15, 15) # pn = [which(n == c(100, 200,500,1000))]
  #J = K = k  = pn = pn00 =  Nbasis0[which(c(100,200,500, 1000) == n)]
  a <- seq(0, 1, length.out=t) ## time points at which obs. are takes
  
  met <- c("f1","f2","f3","f4","f5")
  
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
  
  
  P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D 
    return(P)
  }
  
  #Deltafun <- function(a, scale, loc){ (.25 - (a - .5)^2)*scale + rep(loc,length(a))}
  Deltafun <- function(a, scale, loc){  (1.25 + Betafunc(a, met[idf]))*scale + rep(loc,length(a))}
  
  Delt <- Deltafun(a, scale=1.,loc=.5)
  # P.mat <- function(K){
  #   return(ppls::Penalty.matrix(K,order=2))
  # }
  par(mfrow=c(2,2))
  plot(a,Betafunc(a,  met[idf]), type="l")
  plot(a, Delt, type="l")
  
  #P.mat(3)
  
  #---------------------------------------------------------------------
  #---- Simulate X
  #---------------------------------------------------------------------
  #sig.x = .5
  SigX <- CovMat(t, CovMet,sig.x,rhox) #ones(t,t)*rhox*sig.x*sig.x; diag(SigX) <- (sig.x^2) 
  MeanX <- Betafunc(a, met[1]) #Betafunc(a, met[idf]) #sinpi(2*a)
  MeanX2 <- Betafunc(a, met[1]) #Betafunc(a, met[idf]) #sinpi(2*a)
  X_t <- matrix(MeanX, ncol=t, nrow=n, byrow = T) + DistUw(n,distFUe, SigX)  #rbind(mvtnorm::rmvnorm(n=n*.5,mean = MeanX, sigma = SigX), mvtnorm::rmvnorm(n=n*.5,mean = MeanX2, sigma = SigX))
  plot(a,MeanX, type="l",lwd=3)
  
  # We generate the basis functions from bsplines 
  a   <- seq(0, 1,length.out=t)
  bs2 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0,attr(bs20,"knots"), 1) ))
  #bs2 <- B.basis(a, knots= seq(0,1, length.out = pn-1))
  
  # We generate Yi the scalar response with error terms from Normal (0,.05) (Equation (1)) 
  #if(distF == "Norm"){
  pi.g <- .65
  Prob0 <- rbinom(n=n,size=1,prob = pi.g)
  
  # Z <- matrix(rnorm(n=n*(j0-1), sd = 1), ncol = c(j0-1)); Z <- scale(Z); 
  # Z <- cbind(Z, rbinom(n = n, prob=pi.g, size=1))
  #Z[,2] <- c(mapply(rnorm, n=1, mean = crossprod(t(X_t),Betafunc(a,met[idf]))/length(a))); Z <- scale(Z); 
  
  Betaz <- rnorm(n = j0+1); 
  Betaz <- c(-1.05, .57)*c(1, .35,.35)[idf] #c(0., -1.05, .57) # c(1.87, -1.05, .57)
  Betaz <- c(-1.65,  0.91, -0.84)  #rnorm(n = j0) + .2; #c(0., -1.05, .57) # c(1.87, -1.05, .57)
  Zmod <- Z #cbind(1, Z)
  Val =1 #<- caseN %in%paste("case",1:4,"a",sep="")
  #*(1-Val)
  if(distFep != "Gam"){
  ei <- DistEps(n, distFep, sig.e)*c(1,1,.5, 1)[idf]*3
  BetatF <- (Betafunc(a,met[idf]) - mean(Betafunc(a,met[idf])))*c(1,2,2, 2)[idf]
  fx0 = crossprod(t(X_t),BetatF)/length(a)
  }else{ 
  ei <- DistEps(n, distFep, sig.e)*1
  BetatF <- 1.5*(Betafunc(a,met[idf]) - mean(Betafunc(a,met[idf])))*c(1,2,2, 2)[idf]
  fx0 = crossprod(t(X_t),BetatF)/length(a)
  }
  
  #---- Here we simulate Y with non-canstant variance (dependent on the functional covariate X(t))
  
  Y <- c(Zmod%*%Betaz*Val) + c(fx0)  + (ei - mean(ei))*c(fx0 + c(Zmod%*%Betaz*Val))  #*(crossprod(t(X_t),Betafunc(a,met[idf]))/length(a)) #(ei-mean(ei))  
  # #d0*rnorm(n, sd = sig.e) + (1-d0)*(rgamma(n=n,shape=1,scale=1.5) - 1.5) # (n,1) Y matrix
  
  
  #f0 <- function(t) { X_t[1,]*Betafunc(t,met[idf])}
  par(mfrow=c(2,2))
  MASS::truehist(Y)
  MASS::truehist(ei)
  MASS::truehist(c(Zmod%*%Betaz))
  MASS::truehist(fx0)
  
  #print(mean(Zmod%*%Betaz*Val + crossprod(t(X_t),Betafunc(a,met[idf]))/length(a))/sd((ei-mean(ei))))
  print(mean(Zmod%*%Betaz*Val + fx0)/sd((ei-mean(ei))))
  
  
  #Y <- 0 + crossprod(t(X_t),Betafunc(a,met[idf]))/length(a) +  rgamma(n=n,shape=1,scale=1.5) - 1.5 # (n,1) Y matrix
  #---------------------------------------------------------------------
  # We generate the observed surrogates values W with errors from the Skew Normal distribution (Equation (2))
  #---------------------------------------------------------------------
  SigW <- ones(t,t)*rho.m*sig.w*sig.w; diag(SigW) <- (sig.w^2) 
  MeanW <- numeric(t)
  SigU <- CovMat(t, CovMet,sig.u,rho.u) #ones(t,t)*rho.u*sig.u*sig.u; diag(SigU) <- (sig.u^2) 
  MeanU <- numeric(t)
  W_array <- array(NA,c(n, t, nrep))
  M_array <- array(NA,c(n, t, nrep))
  
  if(Wdist == "norm"){
  for(li in 1:nrep){
  U <- DistUw(n,distFUe, SigU) #mvtnorm::rmvnorm(n=n,mean = MeanU, sigma = SigU)
  W_array[,,li] = X_t + U  # (n,t) W matrix of surrogate values
  
  omega <- DistUw(n,distFUe, SigW) #mvtnorm::rmvnorm(n=n,mean = MeanW, sigma = SigW)
  M_array[,,li] <- X_t%*%diag(Delt) + omega  # (n,t) M matrix
  # Me <- DistUw(n,distFUe, Sig)#mvtnorm::rmvnorm(n=n,mean = MeanU, sigma = SigU)
  # M_array[,,li] = X_t + U  # (n,t) W matrix of surrogate values
  
  }} else{
    temp = array((X_t) , dim=c(n,t,m_w))
    W_array= apply(temp, c(1,2,3), function(s) {rpois(1,exp(s))})
  }
  
  Wn <- (W_array[,,1]%*%bs2)/length(a)
  
  #---------------------------------------------------------------------
  # We generate the IV with errors from the Normal distribution (Equation (3))
  #---------------------------------------------------------------------
  # SigW <- ones(t,t)*rho.m*sig.w*sig.w; diag(SigW) <- (sig.w^2) 
  # MeanW <- numeric(t)
  # omega <- DistUw(n,distFUe, SigW) #mvtnorm::rmvnorm(n=n,mean = MeanW, sigma = SigW)
  # 
  # M <- X_t%*%diag(Delt) + omega  # (n,t) M matrix
  # # Mn <- (M%*%bs2)/length(a)
  # #Delthat <- rep(1, length(Delt))#Delt # lowess(a, (colMeans(M)/colMeans(W)), f=1/6)$y # c(abs(colMeans(M)/(colMeans(W)))) #   lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) # 
  # #plot(a, Delthat, type="l")
  # Mn_Mod <- M%*%diag(1/Delt)
  # #bs2 <- bs(a, df = pn, intercept = T)
  # Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a) 
  # Wn <- (W%*%bs2)/length(a)
  Xn <- (X_t%*%bs2)/length(a)
  
  Wrep = array(NA,c(nrep,t,n))
  for(i in 1:n){
    Wrep[,,i] <- t(W_array[i,,])
    
  }
  
  Data <- list(a = a, bs2 = bs2, Z = Z,  Y = Y, X = X_t, Xn = Xn, 
               Data=list(W = W_array, Wrep = Wrep, M= W_array[,,1], Mall = M_array, X = X_t), Delt = Delt, BetaZ = Betaz, Betat = BetatF)
  Data
}


#--- FUI appaproach

FUI = function(model="gaussian",smooth=FALSE, data,silent = FALSE){
  
  ##' @param model is the model for apporixation 
  ##   model = gaussian, poisson 
  ##' @param data is the observed measurement of X(t) with repeated measures, which is in 3 dimension [n, t, m_w]
  ##' @param smooth whether conduct the smoothing step or not (Step II)
  ##' @param silent whether to show descriptions of each step
  
  ## The output of this function is the approximation of the true function covariate X(t),
  ##    where each column represent each time point and each row represent each subject
  
  
  
  library(lme4) ## mixed models
  #library(refund) ## fpca.face
  library(dplyr) ## organize lapply results
  #library(progress) ## display progress bar
  library(mgcv) ## smoothing in step 2
  #library(mvtnorm) ## joint CI
  #library(parallel) ## mcapply
  
  ### create a dataframe for analysis 
  m_w = dim(data)[3] ## number of repeated measures
  t = dim(data)[2] ## number of observations on the functional domain (time points)
  n = dim(data)[1] ## number of subjects 
  M = NULL
  i=1
  while(i <= m_w){
    M = rbind(M,data[,,i])
    i=i+1
  } ### convert the 3d array to 2d matrix 
  
  data.M = data.frame(M = I(M),seqn= rep(1:n, m_w) )
  
  ##########################################################################################
  ## Step 1 (Massive Univariate analysis)
  ##########################################################################################
  if(silent == FALSE) print("Step 1: Massive Univariate Mixed Models")
  
  if(model =="gaussian"){
    fit = data.M %>%
      dplyr::select(M) %>%
      apply( 2, function(s) {
        temp= data.frame(M=s, seqn=data.M$seqn)
        mod =  suppressMessages(lmer(M ~ (1 | seqn), data = temp))
        return(predict(mod))
      })
  }else{
    fit = data.M %>%
      dplyr::select(M) %>%
      apply( 2, function(s) {
        temp= data.frame(M=s, seqn=data.M$seqn)
        mod =  suppressMessages(glmer(M ~ (1 | seqn), data = temp, family = model))
        return(predict(mod))
      })
    
  }
  
  approx= fit[!duplicated(data.M$seqn),] ## each column represents one time point 
  
  ##########################################################################################
  ## Step 2 (Smoothing)
  ##########################################################################################
  
  if(smooth==TRUE){
    if(silent == FALSE) print("Step 2: Smoothing")
    nknots <- min(round(t/4), 35) ## number of knots for penalized splines smoothing
    argvals = seq(0,1, length.out = t) ## locations of observations on the functional domain
    approx= t(apply(approx, 1, function(x) gam(x ~ s(argvals, bs = "cr", k = (nknots + 1)), method = "REML")$fitted.values))
  }
  
  return(approx)
}



### Fast short multivariate interference(FSMI) function ###

FSMI = function(model="gaussian", cov.model ="us", data){
  ## @t is the total time points
  ## @model is the model for apporixation
  ##   model = gaussian, poisson
  ## @cov.model is the covaraince structure for random effect
  ##   cov.model = us, id, cs
  ## @data is the observed measurement of X(t) with repeated measures, in 3 dimension [n,t,m_w]
  ## The output of this function is the approximation of the true function covariate X(t),
  ##    where each column represent each time point and each row represent each subject
  
  m_w = dim(data)[3] ## number of repeated measures
  t = dim(data)[2] ## number of observations on the functional domain (time points)
  n = dim(data)[1] ## number of subjects 
  data.M = matrix(data, prod(dim(data)[1:3])) %>% as.data.frame()### convert the 3d array to 2d matrix
  data.M$seqn = rep(1:n, prod(t,m_w)) ## when convert into matrix, each column comes after the first colum and then the third demension
  data.M$time = rep(sapply(1:t, function(s) rep(s, n)),m_w)
  data.M$day = lapply(1:m_w, function(s) rep(s, n*t)) %>% unlist()
  colnames(data.M)[1] ="M"
  
  ind.time = 1:t ### number of time point
  fit = lapply(ind.time, 
               function(s){
    #print(s)
    
    ### select the interval of time for analysis
    if ((s-1)<1){
      ind.interval = 1:3
    }else if((s+1)>t){
      ind.interval = (t-2):t
    }else {
      ind.interval = (s-1):(s+1)
    }
    
    temp = data.M %>%
      filter(time %in% ind.interval) %>% # only select the data of the selected time points
      mutate(time.cat = as.character(time))
    
    if(model=="gaussian"){
      if(cov.model =="us"){ ## unstructured
        mod =  suppressMessages(lmer(M ~ (1| seqn)+(0 + time.cat | seqn), data = temp)) ## default setting for lmer is unstructured
      }else if (cov.model =="id"){ ## independent
        mod = suppressMessages(lmer(M ~ (1 | seqn)+(1|time.cat), data = temp))
      }  # https://stats.stackexchange.com/questions/86958/variance-covariance-structure-for-random-effects-in-lme4
      # https://stats.stackexchange.com/questions/49775/variance-covariance-matrix-in-lmer
      else if (cov.model =="cs"){ ## compound symmetric
        mod = suppressMessages(lme(M ~1, random = list(seqn= ~1, seqn =pdCompSymm(~ time.cat-1)),
                                   data = temp))
      }else {
        stop ("wrong covariance matrix of random effect")
      }
    }else{
      if(cov.model =="us"){ ## unstructured
        mod =  suppressMessages(glmer(M ~ (1| seqn)+(0 + time.cat | seqn), data = temp, family = model)) ## default setting for lmer is unstructured
      }else if (cov.model =="id"){ ## independent
        mod = suppressMessages(glmer(M ~ (1 | seqn)+(1|time.cat), data = temp, family = model))
      }  # https://stats.stackexchange.com/questions/86958/variance-covariance-structure-for-random-effects-in-lme4
      # https://stats.stackexchange.com/questions/49775/variance-covariance-matrix-in-lmer
      else if (cov.model =="cs"){ ## compound symmetric
        mod = suppressMessages(glme(M ~1, random = list(seqn= ~1, seqn =pdCompSymm(~ time.cat-1)),
                                    data = temp, family = model))
      }else {
        stop ("wrong covariance matrix of random effect")
      }
    }
    # # Different variances
    # ## seperate variance per time level 
    # mod =  lme(M ~time.cat, random = list(seqn= ~1, seqn =pdDiag(~ time.cat-1)),data = temp)
    # 
    # ## unstructured at time level 
    # mod =  lme(M ~time.cat, random = list(seqn= ~1, seqn =pdSymm(~ time.cat-1)),data = temp)
    # 
    # 
    # ## Compound symmetric at time level 
    # mod =  lme(M ~time.cat, random = list(seqn= ~1, seqn =pdCompSymm(~ time.cat-1)),
    #            data = temp)
    # ## https://rpubs.com/samuelkn/CovarianceStructuresInR
    # 
    
    #getVarCov(mod, type="marginal") ## get covariance matrix of y
    #VarCorr(mod, type="marginal") ## get the G matrix
    
    ## us = unstructured, id = independent, cs = compound symmetric, ar1 = autoregressive (1)
    
    ## us = unstructured, id = independent, cs = compound symmetric, ar1 = autoregressive (1)
    pred = predict(mod, newdata = data.frame(time.cat = as.character(s), seqn = unique(temp$seqn)))
    return(pred)
  }) ### end of lapply()
  
  
  approx= matrix(unlist(fit), nrow= n, ncol = t) ## each column represents one time point
  
  return(approx)
}




#--- Find prior parameters for the inversge Gamma prior

SolParm <-function(parm){
  val = c(.2, 4)
  (val[1] - nimble::qinvgamma(0.0001,shape=parm[1],scale=parm[2]))^2 + (val[2] - nimble::qinvgamma(0.9999,shape=parm[1],scale=parm[2]))^2
}

#--------------------------------------------------------
#--- Other aproaches
#--------------------------------------------------------

library(stats)
library(sn)
library(splines)
library(ald)
library(mvtnorm)
library(boot)
library(fda)
#library(ggplot2)
library(lattice)
library(readxl)
library(plyr)
library(magrittr)
library(Matrix)
library(SparseM)
library(quantreg)
#library(corpcor)
library(MASS)
library(reshape)
library(grid)
library(gridExtra)
#library(knitr)
#library(doParallel)
#library(foreach)


### extrapolation function 
extra_quad = function(k,beta_hat_lambda,lambda,lambda_app){
  ## k is the number of basis functions for the functional variable
  ## beta_hat_lambda is the estimated beta(t) from SIMEX simulation step
  ## lambda is the squence of lambda 
  ## lambda_app = -1 for SIMEX estimators 
  gamma = rep(0,k)  
  lambda2 = lambda*lambda
  for(m in 1:k)
  {
    testr<-beta_hat_lambda[m,]
    lr<-lm(testr~lambda+lambda2)
    coeff<-lr$coefficients
    gamma[m]<- coeff[1]+lambda_app*coeff[2]+lambda_app*lambda_app*coeff[3]
  }
  
  return(gamma)
}

#### functions used to generate data 
met = c("f1","f2","f3","f4","f5", "f6", "f7")
Betafuncoo = function(t, met){
  switch(met,
         f1 = sin(2*pi*t),  ## f1 = sin(2*pi*t),
         f2 = 1/(1+exp(4*2*(t-.5))),
         f3 = sin(pi*(8*(t-.5))/2)/ (1 + (2*(8*(t-.5))^2)*(sign(8*(t-.5))+1)),
         f4 = sin(pi*(16*(t-.5))/2)/ (1 + (2*(16*(t-.5))^2)*(.5*sin(8*(t-.5))+1)),
         f5 = sin(pi*(16*(t-.5))/2)/ (1 + (2*(16*(t-.5))^2)*(sin(8*(t-.5))+1)),
         f6 = 1/(1+exp((t))),
         f7 = 1/(1+exp(4*2*(t-.5))) +1
  )
}


######## function for selecting number of basis ###
### based on BIC 
### the output of this function is a BIC value
L_BIC=function(k, Y, W, tau, EF=EF,time_interval){
  ### Y is response, W is observed matrix of W(t); 
  ### k is the number of basis and should be at least 4; 
  ##  tau is quantile level; 
  ## time_interval is interval of observed time
  cubic_bs = bs(time_interval, df=k, degree=3, intercept = TRUE)
  pred = t(t(W)-colMeans(W))%*%cubic_bs/length(time_interval)
  
  coef_quant = rq(Y ~ pred +EF, tau=tau)$coefficients
  n = length(Y)
  eps = Y-cbind(rep(1,n),pred, EF)%*%coef_quant
  loss = eps*ifelse(eps<0, tau-1, tau) 
  
  re=log(mean(loss))+(k+1)*log(n)/n
  return(re)
  
}



#main.delta=function(Y, W, M, Z, n,c,sd_x,sd_w,p_sim,fi,fx,seeds,sim_liter, deltat){
main.delta=function(Y_norm, W_t, M_t, X_t, Z, delta_t,a, seeds=123,sim_liter=1, p_sim = 0.5){
  require(fda)
  ## @seeds is the the seeding basis
  ## @n is the size of simulated data
  ## @c is the tuning parameter for generating delta(t) (delta(t) = c*sin(2*pi*t) +1)
  ## @sd_x is the sd of X(t)
  ## @sd_w is the sd of W(t)
  ## @p_sim is the quantile number (0.25, 0.5, 0.75, 0.95)
  ## @fi is the function for generating Y
  ## @fx is the function for generating X(t)
  ## @sim_lier is the number of replications we do for one simulation
  
  set.seed(seeds)
  t = ncol(W_t)  ## Number of time points
  k_max = 7 ## The maximum number of basis functions to be considered
  #a  = seq(0, 1, length.out=t) #create a sequence between 0,1 of length = t
  n = length(Y)
  
  # sd_m=0.25 ## standard deviation of instrumental variable 
  # sigma_y = .1 ###standard deviation of Y
  # sigma_ef = 0.5 ###standard deviation of EF covariate
  # p = 0.6 ## prob for the EF binary predictor
  
  ############ create covariance matrix for error terms #################
  # rho_x=0.5 ## correlation coefficient of X(t)
  # rho_w=0.5 ## correlation coefficient of W(t)
  # rho_m=0.5 ## correlation coefficient of M(t)
  # 
  # ###compound symmetric covariance matrix ###
  # Sigma_m=sd_m^2*(matrix(rho_m,nrow = t, ncol = t)+diag(rep(1-rho_m,t)))
  # Sigma_x=sd_x^2*(matrix(rho_x,nrow = t, ncol = t)+diag(rep(1-rho_x,t)))
  # Sigma_w=sd_w^2*(matrix(rho_w,nrow = t, ncol = t)+diag(rep(1-rho_w,t)))
  # 
  zero = rep(0,times=t)
  
  ############### create some matrices to store results #############
  
  #### functional variable #####
  beta_X = matrix(nrow=t,ncol=sim_liter) #results matrix from the quantile regression based on the true X
  beta_simex = matrix(nrow=t,ncol=sim_liter)  #SIMEX estimator when assuming delta(t) is 1
  beta_naive = matrix(nrow=t,ncol=sim_liter)  ## naive estimator
  beta_simex.delta = matrix(nrow=t,ncol=sim_liter)  ## SIMEX estimator when delta(t) is estimated as raw ratio
  beta_simex.delta.known = matrix(nrow=t,ncol=sim_liter)  ## SIMEX estimator when delta(t) is known
  beta_simex.deltas = matrix(nrow=t,ncol=sim_liter)  ## SIMEX estimator when delta(t) is estimated as smoothed ratio
  
  ###### continuous error-free covariate ####
  beta_X_EF = matrix(nrow=1,ncol=sim_liter) ## ## benchmark estimator 
  beta_naive_EF = matrix(nrow=1,ncol=sim_liter) # SIMEX estimator when assuming delta(t) is 1
  beta_W_EF = matrix(nrow=1,ncol=sim_liter)  ### EF estimator when ignore ME in functional varibale 
  beta_naive_EF.delta = matrix(nrow=1,ncol=sim_liter)
  beta_naive_EF.deltas = matrix(nrow=1,ncol=sim_liter)
  beta_naive_EF.delta.known = matrix(nrow=1,ncol=sim_liter)
  ###### binary error-free covariate ####
  beta_X_EF.b = matrix(nrow=1,ncol=sim_liter) ## benchmark estimator 
  beta_naive_EF.b = matrix(nrow=1,ncol=sim_liter) ## SIMEX estimator when assuming delta(t) is 1
  beta_W_EF.b = matrix(nrow=1,ncol=sim_liter)  ### EF estimator when ignore ME in functional variable 
  beta_naive_EF.delta.b = matrix(nrow=1,ncol=sim_liter)
  beta_naive_EF.deltas.b = matrix(nrow=1,ncol=sim_liter)
  beta_naive_EF.delta.known.b = matrix(nrow=1,ncol=sim_liter)
  
  
  #### number of basis functions k #### 
  selected_k_naive=c() ## k based on BIC value of naive model 
  selected_k_bench=c() ## k based on BIC value of benchmark model (with X(t))
  selected_k_n=c() ## k based on sample size 
  
  #######  estimated delta(t) #####
  delta_t.hat = matrix(nrow=t, ncol = sim_liter) ## raw delta(t)
  delta_t.hat.smooth = matrix(nrow=t, ncol = sim_liter) ## smoothed delta(t)
  
  
  
  for(iter in 1:sim_liter){
    
    #-------------------------------------------------------------#
    ################### Generate X, W, M, Y,and EF ################
    #-------------------------------------------------------------#
    
    ######### generate error terms #####
    # err_x = mvrnorm(n, zero, Sigma_x) ### error term of X(t)
    # err_w = mvrnorm(n, zero, Sigma_w)### error term of W(t)
    # err_m = mvrnorm(n, zero, Sigma_m)### error term of M(t)
    # err_ef = rnorm(n, 0, sigma_ef) ### error term of continuous error-free covariate
    # 
    
    ### generate delta(t)
    # delta_t =c*Betafunc(a,"f1") +1 # c is the constant might change for different delta(t)
    # 
    # X_t = matrix(rep(Betafunc(a,fx),n),nrow=n,byrow=TRUE) + err_x 
    # M_t = t(apply(X_t, 1, function(s) delta_t*s)) + err_m 
    # W_t = X_t + err_w
    # 
    # EF = rnorm(n, 0, sigma_ef)  ## EF predictor that is NOT correlated with X(t)
    # EF.b = rbinom(n, 1,p) ## binary predictor 
    # 
    # #### generate Y ### 
    # gamma.1 = 2 ## true association of Y and EF
    # gamma.3 = 0.6 ## true association of Y and biarny EF
    # 
    # Y_norm = crossprod(t(X_t),Betafunc(a,fi))/length(a)+ ## true association of Y and x(t)
    #   crossprod(t(EF), gamma.1)+ ## true association of Y and EF
    #   crossprod(t(EF.b), gamma.3)+
    #   rnorm(n,0,sd=sigma_y) 
    # 
    #-------------------------------------------------------------#
    ######################### estimate delta_t   #################
    #-------------------------------------------------------------#
    ### delta(t) is estimated as the ratio of M(t) and W(t) based on the assumption 
    ### that M(t) = delta(t)*W(t)
    delta_t.hat.smooth = lowess(a, (colMeans(M_t))/(colMeans(W_t)), f=1/3)$y ## smoothed delta(t)
    delta_t.hat = (colMeans(M_t))/(colMeans(W_t)) ## non-smoothed delta(t) 
    
    
    ## create new M_t by dividing the old one with estimated delta_t
     M_t.new = M_t%*%diag(1 / c(delta_t.hat))
    M_t.news = M_t%*%diag(1 / c(delta_t.hat.smooth))
   M_t.known = M_t%*%diag(1 / c(delta_t)) 
    
    #-------------------------------------------------------------#
    ############################# Select k  #######################
    #-------------------------------------------------------------#
    
    BIC_value_naive = sapply(5:k_max, L_BIC, Y=Y_norm, W = W_t, tau = p_sim, EF=Z, time_interval = a)
    k = (5:k_max)[which.min(BIC_value_naive)]
    selected_k_naive[iter]=k
    
    BIC_value_bench = sapply(5:k_max, L_BIC, Y=Y_norm, W = X_t, tau = p_sim,  EF=Z,time_interval = a)
    selected_k_bench[iter]=(5:k_max)[which.min(BIC_value_bench)]
    
    k = ceiling(n^(1/5))+2
    selected_k_n[iter] = k
    bs2 = bs(a, df = k, degree = 3, intercept = TRUE)
    
    #-------------------------------------------------------------#
    #################### basis expansion of M_t,W_t,X_t  ##########
    #-------------------------------------------------------------#
    
    W_i = t(t(W_t)-colMeans(W_t))%*%bs2/length(a)   #(n,k) matrix inner product
    X_i = t(t(X_t)-colMeans(X_t))%*%bs2/length(a)   #(n,k) matrix
    M_i = t(t(M_t)-colMeans(M_t))%*%bs2/length(a)   #(n,k) matrix
    M_i.new = t(t(M_t.new)-colMeans(M_t.new))%*%bs2/length(a)   #(n,k) matrix
    M_i.news = t(t(M_t.news)-colMeans(M_t.news))%*%bs2/length(a)   #(n,k) matrix
    M_i.known = t(t(M_t.known)-colMeans(M_t.known))%*%bs2/length(a)   #(n,k) matrix
    
    #-------------------------------------------------------------#
    #################### Benchmark regression model  ##############
    #-------------------------------------------------------------#
    
    #model = rq(Y_norm ~ X_i +EF + EF.b, tau=p_sim)
    model = rq(Y_norm ~ X_i + Z, tau=p_sim)
    quantreg_X = model$coefficients[2:(k+1)]
    beta_X[,iter] = crossprod(t(bs2),quantreg_X)
    beta_X_EF[,iter] = model$coefficients[(k+2)]  ## estimate of EF when functional variable is true measurement
    beta_X_EF.b[,iter] = model$coefficients[(k+3)] 
    
    #-------------------------------------------------------------#
    #################### Naive regression model  ##############
    #-------------------------------------------------------------#
   # model.W = rq(Y_norm ~ W_i +EF+ EF.b, tau=p_sim)
    model.W = rq(Y_norm ~ W_i + Z, tau=p_sim)
    quantreg_W = model.W$coefficients[2:(k+1)]
    beta_naive[,iter] = crossprod(t(bs2),quantreg_W)
    beta_W_EF[,iter] = model.W$coefficients[(k+2)] ## estimate of EF when ME in functional variable is ignored
    beta_W_EF.b[,iter] = model.W$coefficients[(k+3)]
    
    #-------------------------------------------------------------#
    ############ Get a reasonable estimate for Sigma_uu  #########
    #-------------------------------------------------------------#
    
    ##### estimated covariance matrix of X(t) ####
    #######by assumption cov(W(t),M(t))/delta= Sigma_xx
    Sigma_xx =try( cov(W_i, M_i),T)     
    Sigma_xx.delta = try( cov(W_i, M_i.new),T)  
    Sigma_xx.deltas = try( cov(W_i, M_i.news),T) 
    Sigma_xx.delta.known = try( cov(W_i, M_i.known),T) 
    
    ### estimated covariance matrix of W(t) ####
    Sigma_ww = var(W_i)  ########covariance matrix of observed surrogate var(W_ic) (centered)
    
    ###### estimate covariance matrix of measurement error U(t) ###
    Sigma_uu.delta = try(Sigma_ww - Sigma_xx.delta,T) ######covariance matrix of W|x from W=X+U
    Sigma_uu.delta = try(make.positive.definite(as.matrix(forceSymmetric(Sigma_uu.delta))) ,T)
    
    Sigma_uu = try(Sigma_ww - Sigma_xx,T)
    Sigma_uu = try(make.positive.definite(as.matrix(forceSymmetric(Sigma_uu))) ,T)
    
    Sigma_uu.deltas = try(Sigma_ww - Sigma_xx.deltas,T)     #######covariance matrix of W|x from W=X+U
    Sigma_uu.deltas = try(make.positive.definite(as.matrix(forceSymmetric(Sigma_uu.deltas))) ,T)
    
    Sigma_uu.delta.known = try(Sigma_ww - Sigma_xx.delta.known,T)     #######covariance matrix of W|x from W=X+U
    Sigma_uu.delta.known = try(make.positive.definite(as.matrix(forceSymmetric(Sigma_uu.delta.known))) ,T)
    
    #---------------------------------------------------------------#
    ######################### SIME simulation step  #################
    #---------------------------------------------------------------# 
    EF = Z[,1]
    EF.b = Z[,2]
    ##########Simulation step of the SIMEX procedure, see page 101 of Ray's book##############################
    B = 100                        ####number of repliates     
    lambda = seq(0.0001,2.0001,.05)              ###get a set of monotonically increasing small numbers
    
    #### when delta(t) is assumed to be 1 #####
    gamma.simex = lapply(seq(1:B), function(b){
      sapply(lambda, function(s) {
        set.seed(b+iter)
        U_b = try(mvrnorm(n, rep(0, ncol(W_i)), Sigma_uu, empirical = TRUE),T)
        W_lambda =try( W_i + (sqrt(s)*U_b) ,T)
        
        model =  try(rq(Y_norm ~ W_lambda+ EF+ EF.b, tau=p_sim),T)
        try(model$coefficients ,T)
      })
    } )
    
    #### when delta(t) is estimated as raw ratio #####
    gamma_simex.delta = lapply(seq(1:B), function(b){
      sapply(lambda, function(s) {
        set.seed(b+iter)
        U_b = try(mvrnorm(n, rep(0, ncol(W_i)), Sigma_uu.delta, empirical = TRUE),T)
        W_lambda =try( W_i + (sqrt(s)*U_b) ,T)
        
        model=  try(rq(Y_norm ~ W_lambda+ EF+ EF.b, tau=p_sim),T)
        try(model$coefficients ,T)
      })
    } )
    
    #### when delta(t) is estimated as smoothed ratio #####
    gamma_simex.deltas = lapply(seq(1:B), function(b){
      sapply(lambda, function(s) {
        set.seed(b+iter)
        U_b.deltas = try(mvrnorm(n, rep(0, ncol(W_i)), Sigma_uu.deltas, empirical = TRUE),T)
        W_lambda.deltas = try(W_i + (sqrt(s)*U_b.deltas) ,T)
        
        model =  try(rq(Y_norm ~ W_lambda.deltas+ EF+ EF.b, tau=p_sim),T)
        try(model$coefficients ,T)
      })
    } )
    
    #### when delta(t) is known #####
    gamma_simex.delta.known = lapply(seq(1:B), function(b){
      sapply(lambda, function(s) {
        set.seed(b+iter)
        U_b.delta.known = try(mvrnorm(n, rep(0, ncol(W_i)), Sigma_uu.delta.known, empirical = TRUE),T)
        W_lambda.delta.known = try(W_i + (sqrt(s)*U_b.delta.known) ,T)
        
        model =  try(rq(Y_norm ~ W_lambda.delta.known+ EF+ EF.b, tau=p_sim),T)
        try(model$coefficients ,T)
      })
    } )
    
    #-------------------------------------------------------------#
    ######################### extrapolation step  #################
    #-------------------------------------------------------------#
    #### get average accross B ###
    gamma_simex.ave = try(Reduce("+", gamma.simex)/B,T)
    gamma_simex.delta.ave = try(Reduce("+", gamma_simex.delta)/B,T)
    gamma_simex.deltas.ave = try(Reduce("+", gamma_simex.deltas)/B,T)
    gamma_simex.delta.known.ave = try(Reduce("+", gamma_simex.delta.known)/B,T)
    
    
    gamma.t.simex = (try(extra_quad((k+3), gamma_simex.ave,lambda,-1),T)) ## all regression coefficients
    gamma.t.simex.delta = (try(extra_quad((k+3), gamma_simex.delta.ave,lambda,-1),T))
    gamma.t.simex.deltas = (try(extra_quad((k+3), gamma_simex.deltas.ave,lambda,-1),T))
    gamma.t.simex.delta.known = (try(extra_quad((k+3), gamma_simex.delta.known.ave,lambda,-1),T))
    
    
    
    #-------------------------------------------------------------#
    ######################### assume delta is 1  #################
    #-------------------------------------------------------------#
    beta_simex[,iter] = try(crossprod(t(bs2),(gamma.t.simex[2:(k+1)])),T)
    beta_naive_EF[,iter]=try((gamma.t.simex[k+2]),T)
    beta_naive_EF.b[,iter]=try((gamma.t.simex[k+3]),T)
    
    
    #-------------------------------------------------------------#
    ######################### delta estimated  ###############
    #-------------------------------------------------------------#
    
    beta_simex.delta[,iter] = try(crossprod(t(bs2),(gamma.t.simex.delta[2:(k+1)])),T)
    beta_naive_EF.delta[,iter]=try((gamma.t.simex.delta[k+2]),T)
    beta_naive_EF.delta.b[,iter]=try((gamma.t.simex.delta[k+3]),T)
    
    
    #-------------------------------------------------------------#
    ######################### delta estimated smoothed ############
    #-------------------------------------------------------------#
    
    
    beta_simex.deltas[,iter] = try(crossprod(t(bs2),(gamma.t.simex.deltas[2:(k+1)])),T)
    beta_naive_EF.deltas[,iter]=try((gamma.t.simex.deltas[k+2]),T)
    beta_naive_EF.deltas.b[,iter]=try((gamma.t.simex.deltas[k+3]),T)
    
    
    #-------------------------------------------------------------#
    ######################### delta known #######################
    #-------------------------------------------------------------#
    beta_simex.delta.known[,iter] = try(crossprod(t(bs2),(gamma.t.simex.delta.known[2:(k+1)])),T)
    beta_naive_EF.delta.known[,iter]=try((gamma.t.simex.delta.known[k+2]),T)
    beta_naive_EF.delta.known.b[,iter]=try((gamma.t.simex.delta.known[k+3]),T)
    
    
    print(iter)
    
  } ### end of simulation iterations
  
  
  #-------------------------------------------------------------#
  ####################### output of the function ################
  #-------------------------------------------------------------#
  
  re=list(beta_X=beta_X,beta_simex=beta_simex,beta_naive=beta_naive,
          selected_k_naive=selected_k_naive,selected_k_bench=selected_k_bench,
          beta_simex.delta = beta_simex.delta, ## SIMEX estimator when delta is assumed to be 1
          beta_simex.delta.known = beta_simex.delta.known, ### SIMEX estimator of functional variable when delta is known
          beta_simex.deltas = beta_simex.deltas, ### SIMEX estimator of functional variable when delta is estimated 
          beta_X_EF=beta_X_EF,beta_W_EF=beta_W_EF,
          beta_X_EF.b=beta_X_EF.b,beta_W_EF.b=beta_W_EF.b,
          beta_naive_EF=beta_naive_EF, #### Estimator of EF covariate when ME in functional variable is corrected by assume delta is 1 
          beta_naive_EF.delta= beta_naive_EF.delta,
          beta_naive_EF.delta.known= beta_naive_EF.delta.known,
          beta_naive_EF.deltas= beta_naive_EF.deltas,
          beta_naive_EF.b=beta_naive_EF.b, #### Estimator of the binary EF covariate when ME in functional variable is corrected by assume delta is 1 
          beta_naive_EF.delta.b= beta_naive_EF.delta.b,
          beta_naive_EF.delta.known.b= beta_naive_EF.delta.known.b,
          beta_naive_EF.deltas.b= beta_naive_EF.deltas.b,
          delta_t = delta_t.hat,
          delta_t.smooth= delta_t.hat.smooth) #### Estimator of EF covariate when ME in functional variable is corrected by estimating delta 
  
  return(re)
  
} ### end of simulation function 

#-------------------------------------------------------------#
########################### Example ###########################
#-------------------------------------------------------------#


### run a 500-replications simulation 
#res = main.delta(n=200,c= 0,sd_x=1.5,sd_w=0.75,p_sim=0.5,fi = "f1",fx = "f7",seeds=123,sim_liter=500)







cat("\n\n All the functions have been imported ------\n")