#!/usr/bin/Rscript
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Cobined update update of te beta(t) and betaz using Gibbs
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
 #!/usr/bin/Rscript
#.libPaths(c( "/geode2/home/u070/rszoh/Quartz/R/x86_64-pc-linux-gnu-library/3.6", "/geode2/soft/hps/rhel8/r/3.6.0/lib64/R/library" ))
#  libs <- c(.libPaths(), c("/geode2/home/u070/rszoh/Quartz/R/x86_64-pc-linux-gnu-library/4.0", "/geode2/soft/hps/rhel8/r/4.0.4/lib64/R/library" ))
# .libPaths(libs)
#  #.libPaths(c("/geode2/home/u070/rszoh/BigRed3/R/x86_64-pc-linux-gnu-library/3.6", "/geode2/soft/hps/cle6/r/3.6.0/lib64/R/library"))
# # #[1] "/geode2/home/u070/rszoh/BigRed3/R/x86_64-pc-linux-gnu-library/3.6"
# # #[2] "/geode2/soft/hps/cle6/r/3.6.0/lib64/R/library"   
#-- Add your personal R library path to the libray path

userid="rszoh" ## <---------------- change this to your id

  libs <- c(.libPaths(), c(paste0("/geode2/home/u070/",userid,"/Quartz/R/x86_64-pc-linux-gnu-library/4.1"), "/geode2/soft/hps/rhel8/r/4.1.1/lib64/R/library",
                           "/geode2/soft/hps/rhel7/r/4.1.1/lib64/R/library"))
 .libPaths(libs)
 .libPaths()
 

  rm(list= ls())
  cmdArgs = commandArgs()
  print(cmdArgs)
# # ##cpchar = abs(as.integer(cmdArgs[length(cmdArgs)]))
  cpchar = cmdArgs[length(cmdArgs)]
  
   cp <- as.numeric(cpchar); cp #(gsub("--","",cpchar)); cp
 Meth <- as.character(cmdArgs[length(cmdArgs)-1]); Meth
 tau0 <- as.numeric(as.character(cmdArgs[length(cmdArgs)-2])); tau0
 case <- as.character(cmdArgs[length(cmdArgs)-3]); case
 
 
 
 #stop()
# print(cp)
# print(case)
# cp =1
# case = 'case3a'
# # Functions for the Gibbs updates
# # Bayesian SoFR
# #****************************************
# #OrgPath <- "/N/dc2/scratch/rszoh/SoFROutputBetaZVer3Final/"
# #OrgPath <- "/N/slate/rszoh/SoFROutputBetaZVer3Final120220/"
# OrgPath <- "/N/slate/rszoh/SoFROutputBetaZVer3Final122120/"
#OrgPath <- "/N/slate/rszoh/SoFROut102421/"
#OrgPath <- paste(OrgPath,case,"/",sep="")
#dir.create(OrgPath)
# #OrgPath <- paste(OrgPath,case,"/",sep="")
# dir.create(OrgPath)
# #cp=1
#dir.create(OrgPath)
#--- import useful libraries

library(LaplacesDemon)
library(GeneralizedHyperbolic)
library(mvtnorm)
library(splines)
library(nimble)
library(Matrix)
library(pracma)
library(splines2)
library(truncnorm)
#library(tmvtnorm)
library(emulator)
library(mclust)
library(Rcpp)
library(compiler)
library(TruncatedNormal)
library(inline)
#library(caret)
library(data.table)
#library(RcppFunc)
#library(ppls)
library(refund)
library(mixtools)
library(KScorrect)
#library(heavy)
#library(cascsim)
#library(RcppFuncVec)
#library(RcppFunc)
library(Rmpfr)  
library(quantreg)
library(QSOFRfunc)  #-- This package will be available on the cluster
cat("\n **** Done loading all libraries ***--- > \n")
#set.seed(2021000)
#----- Put Rcpp functions
#---------------------------------------
userid="rszoh"       ## <---------------- change this to your id


func <- function(Pik, n=1, size=1){
  which.max(rmultinom(n=1,size=1,prob = Pik))
}
funcCmp <- cmpfun(func)

F2 <- function(Pik, n=1, size=1){
  
  apply(Pik,1,funcCmp)
}
F2Cmp <- cmpfun(F2)

{
  # # 
  # {Code <-
  #   '
  # arma::mat CikW(arma::mat W, arma::mat X, arma::mat Mu, arma::mat Sig, arma::vec Pk){
  # 
  # const double log2pi = std::log(2.0 * M_PI);
  # 
  # int n = W.n_rows;
  # int K = Mu.n_rows;
  # int P = Mu.n_cols;
  # double constants = -(static_cast<double>(P)/2.0) * log2pi;
  # arma::mat sigma(P, P);
  # arma::mat pik(n, K);
  # arma::mat rooti(P, P);
  # 
  # for(int k=0; k < K; k++){
  # sigma = arma::reshape(Sig.row(k), P, P);
  # rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
  # double rootisum = arma::sum(log(rooti.diag()));
  # for(int i=0; i < n; i++){
  # arma::rowvec xtp = W.row(i) - X.row(i);
  # arma::vec z = rooti * arma::trans( xtp - Mu.row(k));
  # pik(i,k) = Pk(k)*exp(constants - 0.5 * arma::sum(z%z) + rootisum);
  # }
  # }
  # return(pik);
  # }
  # '
  # }
  # 
  # {Code <-
  #     "
  #     #include <RcppArmadillo.h>
  #    // [[Rcpp::depends(RcppArmadillo)]]
  # 
  #    // [[Rcpp::export]]
  # arma::mat CikW(arma::mat W, arma::mat X, arma::mat Mu, arma::mat Sig, arma::vec Pk){
  # 
  # const double log2pi = std::log(2.0 * M_PI);
  # 
  # int n = W.n_rows;
  # int K = Mu.n_rows;
  # int P = Mu.n_cols;
  # double constants = -(static_cast<double>(P)/2.0) * log2pi;
  # arma::mat sigma(P, P);
  # arma::mat pik(n, K);
  # arma::mat rooti(P, P);
  # 
  # for(int k=0; k < K; k++){
  # sigma = arma::reshape(Sig.row(k), P, P);
  # rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
  # double rootisum = arma::sum(log(rooti.diag()));
  # for(int i=0; i < n; i++){
  # arma::rowvec xtp = W.row(i) - X.row(i);
  # arma::vec z = rooti * arma::trans( xtp - Mu.row(k));
  # pik(i,k) = Pk(k)*exp(constants - 0.5 * arma::sum(z%z) + rootisum);
  # }
  # }
  # return(pik);
  # }
  # "
  # }
  # sourceCpp(code = Code)
  # 
  # {Code1 <- '
  # arma::mat CikMvec(arma::mat M, arma::mat X, arma::mat Mu, arma::mat Sig, arma::vec Pk, arma::vec Delt){
  # 
  # const double log2pi = std::log(2.0 * M_PI);
  # 
  # int n = M.n_rows;
  # int K = Mu.n_rows;
  # int P = Mu.n_cols;
  # double constants = -(static_cast<double>(P)/2.0) * log2pi;
  # arma::mat sigma(P, P);
  # arma::mat pik(n, K);
  # arma::mat rooti(P, P);
  # arma::mat diDelt = diagmat(Delt);
  # 
  # for(int k=0; k < K; k++){
  # sigma = arma::reshape(Sig.row(k), P, P);
  # rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
  # double rootisum = arma::sum(log(rooti.diag()));
  # for(int i=0; i < n; i++){
  # arma::rowvec xtp = M.row(i) - X.row(i)*diDelt;
  # arma::vec z = rooti * arma::trans( xtp - Mu.row(k));
  # pik(i,k) = Pk(k)*exp(constants - 0.5 * arma::sum(z%z) + rootisum);
  # }
  # }
  # return(pik);
  # }
  # '
  # }
  # {Code2 <- '
  # arma::mat CikX(arma::mat X, arma::mat Mu, arma::mat Sig, arma::vec Pk){
  # 
  # const double log2pi = std::log(2.0 * M_PI);
  # 
  # int n = X.n_rows;
  # int K = Mu.n_rows;
  # int P = Mu.n_cols;
  # double constants = -(static_cast<double>(P)/2.0) * log2pi;
  # arma::mat sigma(P, P);
  # arma::mat pik(n, K);
  # arma::mat rooti(P, P);
  # 
  # for(int k=0; k < K; k++){
  # sigma = arma::reshape(Sig.row(k), P, P);
  # rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
  # double rootisum = arma::sum(log(rooti.diag()));
  # for(int i=0; i < n; i++){
  # arma::rowvec xtp = X.row(i);
  # arma::vec z = rooti * arma::trans( xtp - Mu.row(k));
  # pik(i,k) = Pk(k)*exp(constants - 0.5 * arma::sum(z%z) + rootisum);
  # }
  # }
  # return(pik);
  # }
  # '
  # }
  # # # 
  # {Code3 <- '
  # 
  # arma::colvec Y = Rcpp::as<arma::vec>(Ys);
  # arma::mat X = Rcpp::as<arma::mat>(Xs);
  # arma::colvec  Mu = Rcpp::as<arma::vec>(Mus);
  # arma::colvec Sig = Rcpp::as<arma::vec>(Sigs);
  # arma::colvec Pk  = Rcpp::as<arma::vec>(Pks);
  # arma::colvec Gam = Rcpp::as<arma::vec>(Gams);
  # 
  # 
  # int n = X.n_rows;
  # int K = Mu.n_elem;
  # 
  # arma::mat pik = arma::zeros<arma::mat>(n, K);
  # arma::colvec Ytld = Y - X*Gam;
  # 
  # for(int k=0; k < K; k++){
  # for(int i=0; i < n; i++){
  # pik(i,k) = Pk(k)*R::dnorm(Ytld(i), Mu(k), sqrt(Sig(k)), 0);
  # }
  # }
  # return Rcpp::wrap(pik);
  # '
  # }
  # #
  # Cike <- cxxfunction(signature(Ys="numeric", Xs="numeric",Mus="numeric",Sigs="numeric",Pks="numeric",Gams="numeric"),
  #                     Code3, plugin="RcppArmadillo")
  # 
  # #cppFunction(Code,depends = "RcppArmadillo")
  # cppFunction(Code1,depends = "RcppArmadillo")
  # cppFunction(Code2,depends = "RcppArmadillo")
  # 
  # {
  #   #   Code4 <- '
  #   # 
  #   # arma::mat UpdateX(arma::vec Yd, arma::mat Wd, arma::mat Md, arma::mat Mukx, arma::vec Sige, arma::mat Sigw, arma::mat Sigm, arma::mat Sigx,  
  #   # arma::vec Cike, arma::vec Cikw, arma::vec Cikm, arma::vec Cikx,arma::vec Gamma, double Delta, arma::vec A, arma::vec B){
  #   # int n = Yd.n_elem;
  #   # int J = Wd.n_cols;
  #   # arma::mat Sigt(J,J);
  #   # arma::vec Mut;
  #   # arma::mat Gamsq = Gamma*Gamma.t();
  #   # double Deltsq = Delta*Delta;
  #   # arma::mat Res(n,J);  
  #   # 
  #   # // Obtaining namespace of Matrix package
  #   # Environment pkg1 = Environment::namespace_env("LaplacesDemon");
  #   # Function F1 = pkg1["as.symmetric.matrix"];
  #   # Function F2 = pkg1["as.inverse"];
  #   # Environment pkg2 = Environment::namespace_env("TruncatedNormal");
  #   # Function F3 = pkg2["rtmvnorm"];
  #   # 
  #   # for(int i=0; i < n; i++){
  #   # int wi = Cikw(i)-1;
  #   # int mi = Cikm(i)-1;
  #   # int xi = Cikx(i)-1;
  #   # int ei = Cike(i)-1;
  #   # arma::mat Sigw0 = as<arma::mat>(F2(F1(reshape(Sigw.row(wi), J, J))));
  #   # arma::mat Sigm0 = as<arma::mat>(F2(F1(reshape(Sigm.row(mi), J, J))));
  #   # arma::mat Sigx0 = as<arma::mat>(F2(F1(reshape(Sigx.row(xi), J, J))));
  #   # 
  #   # Sigt = as<arma::mat>(F2(F1((Gamsq/Sige(ei) + Sigw0 + Sigx0 + Gamsq*Sigm0))));  
  #   # Mut = Sigt*((Gamma*Yd[i]/Sige(ei)) + Sigw0*Wd.row(i).t() + Sigm0*Md.row(i).t() + Sigx0*Mukx.row(xi).t());
  #   # 
  #   # Res.row(i) = as<arma::rowvec>(F3(1, Mut, Sigt, A, B));
  #   # 
  #   # }
  #   # 
  #   # return Res;
  #   # 
  #   # }
  #   # '
  # }
  # 
  # #    //arma::mat UpdateX(arma::vec Yd, arma::mat Wd, arma::mat Md, arma::mat Sigw, arma::mat Sigm, arma::mat Sigx, arma::vec Cikw, arma::vec Cikm, arma::vec Cikx, arma::vec A, arma::vec B, arma::mat Mukx ){
  # {
  #   Code4 <- '
  #   arma::mat UpdateX(arma::mat Wd, arma::mat Md, arma::mat Sigw, arma::mat Sigm, arma::mat Sigx, arma::vec A, arma::vec B, arma::mat Mukx ){
  #   int n = Wd.n_rows;
  #   int J = Wd.n_cols;
  #   arma::mat Sigt(J,J);
  #   arma::mat Sigtp(J,J);
  #   arma::vec Mut(J);
  #   //arma::mat Gamsq = Gamma*Gamma.t();
  #   //double Deltsq = Delta*Delta;
  #   arma::mat Res(n,J);
  # 
  #   // Obtaining namespace of Matrix package
  #   Environment pkg1 = Environment::namespace_env("LaplacesDemon");
  #   Function F1 = pkg1["as.symmetric.matrix"];
  #   Function F2 = pkg1["as.inverse"];
  #   Function F2a = pkg1["rmvnp"];
  #   Function F2b = pkg1["rmvn"];
  #   Environment pkg2 = Environment::namespace_env("TruncatedNormal");
  #   Function F3 = pkg2["rtmvnorm"];
  # 
  #   for(int i=0; i < n; i++){
  #   //int wi = Cikw(i)-1;
  #   //int mi = Cikm(i)-1;
  #   //int xi = Cikx(i)-1;
  #   //int ei = Cike(i)-1;
  #   //arma::mat Sigw0 = as<arma::mat>(F2(F1(reshape(Sigw.row(wi), J, J))));
  #   //arma::mat Sigm0 = as<arma::mat>(F2(F1(reshape(Sigm.row(mi), J, J))));
  #   //arma::mat Sigx0 = as<arma::mat>(F2(F1(reshape(Sigx.row(xi), J, J))));
  #   
  #   arma::mat Sigw0 = as<arma::mat>(F1(reshape(Sigw.row(i), J, J)));
  #   arma::mat Sigm0 = as<arma::mat>(F1(reshape(Sigm.row(i), J, J)));
  #   arma::mat Sigx0 = as<arma::mat>(F1(reshape(Sigx.row(i), J, J)));
  # 
  #   //Sigt = as<arma::mat>(F2(F1((Gamsq/Sige(ei) + Sigw0 + Sigx0 + Gamsq*Sigm0))));
  #    Sigt = as<arma::mat>(F2(Sigw0 + Sigx0 + Sigm0));
  #    Sigtp = Sigw0 + Sigx0 + Sigm0;
  #   Mut =Sigt*(Sigw0*Wd.row(i).t() + Sigm0*Md.row(i).t() + Sigx0*Mukx.row(i).t());
  # 
  #   Res.row(i) = as<arma::rowvec>(F3(1, Mut, Sigt, A, B));
  #   //Res.row(i) = as<arma::rowvec>(F2a(1, Mut, Sigtp));
  # 
  #   }
  # 
  #   return Res;
  # 
  #   }
  #   '
  # }
  # # UpdateX <- cxxfunction(signature(Ys="numeric", Xs="numeric",Mus="numeric",Sigs="numeric",Pks="numeric",Gams="numeric"),
  # #                     Code3, plugin="RcppArmadillo")
  # 
  # cppFunction(Code4, depends = "RcppArmadillo")
  # # 

  
  }
#sourceCpp("QSOFR_Rcpp.cpp")

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
#---------------------------------------------------------------------
# Put useful functions here
#------------------------------------------------------------------------------ 
#--- Case1  - n=100, n = 200; n = 500
case = "case3"
{
  switch (case, 
          
          #case 1 - asymptotic verification
          "case1a" = {caseN ="case1a"; n = 200; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0.; rho.u = 0.; rho.m = 0; Delta=1; t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]},# normal Error / beta(t) = met[1]
          "case2a" = {caseN ="case2a"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          "case3a" = {caseN ="case3a"; n = 1000; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          #"case4a" = {caseN ="case4a"; n = 1000; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          #"case4a" = {caseN ="case4a"; n = 1000; sig.e=.4; sig.x = .5; sig.u = .5; sig.w=.4; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[1]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          
          #case 2 - Correlated errors
          "case1b" = {caseN ="case1b"; n = 200; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]},# normal Error / beta(t) = met[1]
          "case2b" = {caseN ="case2b"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          "case3b" = {caseN ="case3b"; n = 1000; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          
          #"case4" = {caseN ="case4"; n = 1000; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          #"case32" = {caseN ="case32"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=4; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"},
          
          #case 3 - Various distribution for the responses
          "case1c" = {caseN ="case1c"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[1]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]},# normal Error / beta(t) = met[1]
          "case2c" = {caseN ="case2c"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[4]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          "case3c" = {caseN ="case3c"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[3]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          #"case4c" = {caseN ="case4c"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[4]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          #"case32" = {caseN ="case32"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=4;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"},
          
          
          #case 4 - Varying error variance  sig.e
          "case1d" = {caseN ="case1d"; n = 500; sig.e=4; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]},# normal Error / beta(t) = met[1]
          "case2d" = {caseN ="case2d"; n = 500; sig.e=16; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          
          #case 5 - Varying error variance  sig.x
          "case1e" = {caseN ="case1e"; n = 500; sig.e=1; sig.x = 2; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          "case2e" = {caseN ="case2e"; n = 500; sig.e=1; sig.x = 16; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          #"case32" = {caseN ="case32"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=4;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"},
          
          #case 6 - Varying error variance  sig.u
          "case1f" = {caseN ="case1f"; n = 500; sig.e=1; sig.x = 4; sig.u = 1; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          "case2f" = {caseN ="case2f"; n = 500; sig.e=1; sig.x = 4; sig.u = 16; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          
          #case g - Varying error variance  rho.u
          "case1g" = {caseN ="case1g"; n = 500; sig.e=1; sig.x = 4; sig.u = 1; sig.w=1; rhox = 0.5; rho.u = 0.25; rho.m = 0.25; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          "case2g" = {caseN ="case2g"; n = 500; sig.e=1; sig.x = 4; sig.u = 16; sig.w=1; rhox = 0.5; rho.u = 0.75; rho.m = 0.75; Delta=1;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          "case3g" = {caseN ="case3g"; n = 500; sig.e=1; sig.x = 4; sig.u = 1; sig.w=1; rhox = 0.5; rho.u = 0.25; rho.m = 0.25; Delta=1;t=50; nrep=2; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          "case4g" = {caseN ="case4g"; n = 500; sig.e=1; sig.x = 4; sig.u = 16; sig.w=1; rhox = 0.5; rho.u = 0.75; rho.m = 0.75; Delta=1;t=50; nrep=2; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[2]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          
          #case g - Varying number of replicates
          
          
          { 
          # #case 1 - Changing error variance sig.e
          # "case1b" = {caseN ="case1b"; n = 500; sig.e=.5; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[1]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]},# normal Error / beta(t) = met[1]
          # "case2b" = {caseN ="case2b"; n = 500; sig.e=4; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[1]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          # "case3b" = {caseN ="case3b"; n = 500; sig.e=16; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[1]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          # 
          # #case 1 - Changing error u sig.x
          # "case1b" = {caseN ="case1b"; n = 500; sig.e=1; sig.x = 1; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[1]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]},# normal Error / beta(t) = met[1]
          # "case2b" = {caseN ="case2b"; n = 500; sig.e=1; sig.x = 16; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[1]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          # "case3b" = {caseN ="case3b"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0.5; rho.u = 0.5; rho.m = 0.5; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distFep=c("Rst", "Norm","MixNorm","Gam")[1]; distFUe = c("Rmst","Mvnorm")[2]; distFepWe=c("Rmst","Mvnorm")[2]; CovMet =c("CS", "Ar1")[2]}, # normal Error/ beta(t) = met[1]
          # 
          # 
          # 
          # #-- different Delta
          # "case5" = {caseN ="case5"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=.1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # 
          # #-- Skew istributions
          # "case6" = {caseN ="case6"; n = 100; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Gam"}, # Gamma(1,1.5) Error/ beta(t) = met[1]
          # "case7" = {caseN ="case7"; n = 200; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Gam"}, # Gamma(1,1.5) Error/ beta(t) = met[1]
          # "case8" = {caseN ="case8"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Gam"}, # Gamma(1,1.5) Error/ beta(t) = met[1]
          # "case8a" = {caseN ="case8a"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "MixNorm"}, # Gamma(1,1.5) Error/ beta(t) = met[1]
          # 
          # "case9" = {caseN ="case9"; n = 1000; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Gam"}, # Gamma(1,1.5) Error/ beta(t) = met[1]
          # 
          # #-- Function 
          # "case10" = {caseN ="case10"; n = 100; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 3;distF <- "Norm"}, # Gamma(1,1.5) Error/ beta(t) = met[1]
          # "case10a" = {caseN ="case10a"; n = 200; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 3;distF <- "Norm"}, # 
          # "case10b" = {caseN ="case10b"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 3;distF <- "Norm"},
          # "case10c" = {caseN ="case10c"; n = 1000; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 3;distF <- "Norm"},
          # 
          # # Varying sig.u 
          # "case11" = {caseN ="case11"; n = 500; sig.e=1; sig.x = 4; sig.u = 0.5; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error / beta(t) = met[1]
          # "case12" = {caseN ="case12"; n = 500; sig.e=1; sig.x = 4; sig.u = 1; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # "case13" = {caseN ="case13"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, #normal Error/ beta(t) = met[1]
          # "case14" = {caseN ="case14"; n = 500; sig.e=1; sig.x = 4; sig.u = 16; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # 
          # # Varying sig.w 
          # "case15" = {caseN ="case15"; n = 500; sig.e=1; sig.x = 4; sig.u = 1; sig.w=0.5; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error / beta(t) = met[1]
          # "case16" = {caseN ="case16"; n = 500; sig.e=1; sig.x = 4; sig.u = 1;   sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # "case17" = {caseN ="case17"; n = 500; sig.e=1; sig.x = 4; sig.u = 1;  sig.w=4; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, #normal Error/ beta(t) = met[1]
          # "case18" = {caseN ="case18"; n = 500; sig.e=1; sig.x = 4; sig.u = 1; sig.w=16; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # 
          # #varying rhox   
          # "case19" = {caseN ="case19"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=.1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # "case20" = {caseN ="case20"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = .25; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error / beta(t) = met[1]
          # "case21" = {caseN ="case21"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = .5; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # "case22" = {caseN ="case22"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = .75; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, #normal Error/ beta(t) = met[1]
          # 
          # #varying rho.u
          # "case23" = {caseN ="case23"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = .25; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # "case24" = {caseN ="case24"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = .5; rho.m = 0; Delta=.1; t = 50; pn = ceiling(n^{1/3.8}) + 1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # "case25" = {caseN ="case25"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = .75; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error / beta(t) = met[1]
          # 
          # #varying rhox.m 
          # "case26" = {caseN ="case26"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = .0; rho.u = 0; rho.m = 0.25; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # "case27" = {caseN ="case27"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = .0; rho.u = 0; rho.m = 0.50; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, #normal Error/ beta(t) = met[1]
          # "case28" = {caseN ="case28"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = .0; rho.u = 0; rho.m = 0.75; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # 
          # #various Delta
          # "case29" = {caseN ="case29"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=.25; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # "case30" = {caseN ="case30"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=.5; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # "case31" = {caseN ="case31"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=2; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # "case32" = {caseN ="case32"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=4; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # 
          # 
          # #Sigx
          # "case33" = {caseN ="case33"; n = 500; sig.e=1; sig.x = .5; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error / beta(t) = met[1]
          # "case34" = {caseN ="case34"; n = 500; sig.e=1; sig.x = 1; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          # "case35" = {caseN ="case35"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, #normal Error/ beta(t) = met[1]
          # "case36" = {caseN ="case36"; n = 500; sig.e=1; sig.x = 16; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"} # normal Error/ beta(t) = met[1]
          # 
          
          
          #"case29" = {caseN ="case29"; n = 100; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 2; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          #"case30" = {caseN ="case30"; n = 200; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 2; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          #"case31" = {caseN ="case31"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 2; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          #"case32" = {caseN ="case32"; n = 1000; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 2; distF <- "Norm"} # normal Error/ beta(t) = met[1]
          }
  )
}


#--- Source importan functions
#source("Core_Function.R")
MainPath <- paste0("/geode2/home/u070/",userid,"/Quartz/BSOFR/Scripts/ReducCore_Function.R")
source(MainPath)
pn=20
print("Simulate data")
#nrep = 5; 
Wdist = "norm"
print("Simulate data")

SeedVal <- runif(n=1, min=1000, max=100000)

set.seed(20004)
#DataSim <- DataSimReplicate(pn, n, t, a, j0=2, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep=c("Rst", "Norm","MixNorm")[2], distFUe = c("Rmst","Mvnorm")[2], distFepWe=c("Rmst","Mvnorm")[1], CovMet =c("CS", "Ar1")[2], sig.e, idf=c(1:4)[3], nrep, Wdist)
#DataSim <- DataSimReplicate(pn, n, t, a, j0=2, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep=c("Rst", "Norm","MixNorm")[1], distFUe = c("Rmst","Mvnorm")[1], distFepWe=c("Rmst","Mvnorm")[1], CovMet =c("CS", "Ar1")[2], sig.e, idf=c(1:4)[3], nrep, Wdist)

#DataSim <- DataSimReplicate(pn, n, t, a, j0=3, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep=c("Rst", "Norm","MixNorm","Gam")[1], distFUe = c("Rmst","Mvnorm")[2], distFepWe=c("Rmst","Mvnorm")[2], CovMet =c("CS", "Ar1")[2], sig.e, idf=c(1:4)[3], nrep, Wdist)
#DataSim
distFep=c("Rst", "Norm","MixNorm","Gam")[4]
#distFUe = c("Rmst","Mvnorm")[2]
#distFepWe=c("Rmst","Mvnorm")[2]
#CovMet =c("CS", "Ar1")[2]
j0 = 3
pi.g <- .65
Prob0 <- rbinom(n=n,size=1,prob = pi.g)
Z <- matrix(rnorm(n=n*(j0-1), sd = 1), ncol = c(j0-1)); Z <- scale(Z); 
Z <- cbind(Z, rbinom(n = n, prob=pi.g, size=1))
DataSim <- DataSimReplicate(pn, n, t, a, j0=3, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep, distFUe, distFepWe, CovMet=c("CS", "Ar1")[2], sig.e, idf=c(1:4)[3], nrep, Wdist, Z)
#DataSim
#DataSim <- DataSimReplicate(pn, n, t, a, j0=2, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep=c("Rst", "Norm","MixNorm")[1], distFUe = c("Rmst","Mvnorm")[1], distFepWe=c("Rmst","Mvnorm")[1], CovMet =c("CS", "Ar1")[2], sig.e, idf=c(1:4)[3], nrep, Wdist)
#DataSim <- DataSimReplicate(pn, n, t, a, j0=2, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep=c("Rst", "Norm","MixNorm")[1], distFUe = c("Rmst","Mvnorm")[2], distFepWe=c("Rmst","Mvnorm")[2], CovMet =c("CS", "Ar1")[2], sig.e, idf=c(1:4)[3], nrep, Wdist)
#DataSim
#DataSim <- DataSim(pn, n, t, a, j0=2, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep=c("Rst", "Norm","MixNorm")[2], distFUe = c("Rmst","Mvnorm")[2], distFepWe=c("Rmst","Mvnorm")[1], CovMet =c("CS", "Ar1")[2], sig.e, idf=c(1:4)[1])
#DataSim <- DataSimHeter(pn, n, t, a, j0=2, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep=c("Rst", "Norm","MixNorm")[2], distFUe = c("Rmst","Mvnorm")[2], distFepWe=c("Rmst","Mvnorm")[2], CovMet =c("CS", "Ar1")[2], sig.e, idf=c(1:4)[3])

#DataSim <- DataSimHeter(pn, n, t, a, j0=2, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep=c("Rst", "Norm","MixNorm")[2], distFUe = c("Rmst","Mvnorm")[2], distFepWe=c("Rmst","Mvnorm")[1], CovMet =c("CS", "Ar1")[2], sig.e, idf=c(1:4)[3])
#Meth = c("Wrep","FUI","Wnaiv")[2]

#--- Assign data 
Y <- DataSim$Y

W <-  DataSim$Data$W
M <-  DataSim$Data$M
X <- DataSim$Data$X
a <- DataSim$a
Delt <- DataSim$Delt
Z <- DataSim$Z
Xn <- DataSim$Xn #(X%*%bs2)/length(a)

if(Meth=="Wrep"){
  
  W = DataSim$Data$Wrep
  M = apply(DataSim$Data$Mall, c(1,2), mean)
}

cat("\n #--- MCMC samples ----\n")
set.seed(round(tau0*cp) + SeedVal)
tau0 = tau0
tunprop = 0.75 ## acceptance rate around 48%
tunprop = 1.5 ## acceptance rate around 42%
tunprop = 3.5 ## acceptance rate around 42%
g0 = 15
Nsim = 10000
#MCMCSample01 <- BQBayes.ME_SoFRFuncSimFxFUI(Y, W, M, Z, a, Nsim=7500, sig02.del = 2, tauval = 1,  Delt, alpx=1, tau0 = tau0,X, g0=15,tunprop)

#Meth = c("Wrep","FUI","Wnaiv")[2]

MCMCSample01 <- FinalMeth(Meth, Y, W, M, Z, a, Nsim = Nsim, sig02.del = 0.1, tauval = .1,  Delt, alpx=1, tau0 , X, g0,tunprop)
#BQBayes.ME_SoFRFuncSimFxFUI


cat("\n #---Save all the output to the ---- \n")
OrgPath <- paste0("/N/scratch/",userid,"/BSOFR",Meth,"/quant_",tau0,"/")

#OrgPath <- "/N/slate/rszoh/SoFROut05042022/"
#OrgPath <- paste(OrgPath,case,"/",sep="")
#dir.create(OrgPath, recursive = T)
#OrgPath <- paste(OrgPath,case,"/",sep="")
#dir.create(OrgPath)


DataPath <-  paste0(OrgPath,"case_",caseN,"/")

ifelse(!dir.exists(DataPath), dir.create(DataPath, recursive = T), FALSE)
### Save the posterior Draws
#DataPath <- OrgPath  #"/N/dc2/scratch/rszoh/SoFROutput/"
cat("\n ----- Save output ---- \n")
#-----------------------------------------------------------------
## Look at the posterior
Path_file <- DataPath

#--- Save gamma values
Path_file <- paste(DataPath, "Gamma_",cp,".csv", sep="")
write.csv(x = MCMCSample01$Gamma,file = Path_file)

#--- Save betaZ values
Path_file <- paste(DataPath, "Betaz_",cp,".csv", sep="")
write.csv(x = MCMCSample01$BetaZ,file = Path_file)

cat("\n ***** End of code ***** \n")