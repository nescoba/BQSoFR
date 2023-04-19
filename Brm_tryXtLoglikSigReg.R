#---- This will fit the model using STAN
#-------------------------------------------------------------------------------
#-- Implementing Model in Rstan
library(brms)
library(rstan)

library(LaplacesDemon)
library(GeneralizedHyperbolic)
library(mvtnorm)
library(splines)
library(nimble)
library(Matrix)
library(pracma)
library(splines2)
library(truncnorm)
library(tmvtnorm)
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
#-------------------------------------------------------------------------------
#--- Import Data
#-------------------------------------------------------------------------------
#source("~/Library/CloudStorage/OneDrive-IndianaUniversity/Join_folder_Annie_DrZoh/NHANES Dataset/2013-2014/BQReg_Nhanes2012_13Application.R")
#load("/Users/rszoh/Library/CloudStorage/OneDrive-IndianaUniversity/Join_folder_Annie_DrZoh/NHANES Dataset/2013-2014/FinalDataV2.RData")

source("~/Library/CloudStorage/OneDrive-IndianaUniversity/GitHub/BQSoFR/ReducCore_Function.R")
#-------------------------------------------------------------------------------
#--- Set up the data
#-------------------------------------------------------------------------------
case = "case3a"
{
  switch (case, 
          
          #case 1 - asymptotic verification
          "case1a" = {caseN ="case1a"; n = 100; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"},# normal Error / beta(t) = met[1]
          "case2a" = {caseN ="case2a"; n = 200; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          "case3a" = {caseN ="case3a"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          #"case4a" = {caseN ="case4a"; n = 1000; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          "case4a" = {caseN ="case4a"; n = 1000; sig.e=.4; sig.x = .5; sig.u = .5; sig.w=.4; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          
          #case 1 - asymptotic verification
          "case1" = {caseN ="case1"; n = 100; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"},# normal Error / beta(t) = met[1]
          "case2" = {caseN ="case2"; n = 200; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          "case3" = {caseN ="case3"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          "case4" = {caseN ="case4"; n = 1000; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          #"case32" = {caseN ="case32"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=4; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"},
          
          #-- different Delta
          "case5" = {caseN ="case5"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=.1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          
          #-- Skew istributions
          "case6" = {caseN ="case6"; n = 100; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Gam"}, # Gamma(1,1.5) Error/ beta(t) = met[1]
          "case7" = {caseN ="case7"; n = 200; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Gam"}, # Gamma(1,1.5) Error/ beta(t) = met[1]
          "case8" = {caseN ="case8"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Gam"}, # Gamma(1,1.5) Error/ beta(t) = met[1]
          "case8a" = {caseN ="case8a"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "MixNorm"}, # Gamma(1,1.5) Error/ beta(t) = met[1]
          
          "case9" = {caseN ="case9"; n = 1000; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Gam"}, # Gamma(1,1.5) Error/ beta(t) = met[1]
          
          #-- Function 
          "case10" = {caseN ="case10"; n = 100; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 3;distF <- "Norm"}, # Gamma(1,1.5) Error/ beta(t) = met[1]
          "case10a" = {caseN ="case10a"; n = 200; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 3;distF <- "Norm"}, # 
          "case10b" = {caseN ="case10b"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 3;distF <- "Norm"},
          "case10c" = {caseN ="case10c"; n = 1000; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 3;distF <- "Norm"},
          
          # Varying sig.u 
          "case11" = {caseN ="case11"; n = 500; sig.e=1; sig.x = 4; sig.u = 0.5; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error / beta(t) = met[1]
          "case12" = {caseN ="case12"; n = 500; sig.e=1; sig.x = 4; sig.u = 1; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          "case13" = {caseN ="case13"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, #normal Error/ beta(t) = met[1]
          "case14" = {caseN ="case14"; n = 500; sig.e=1; sig.x = 4; sig.u = 16; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          
          # Varying sig.w 
          "case15" = {caseN ="case15"; n = 500; sig.e=1; sig.x = 4; sig.u = 1; sig.w=0.5; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error / beta(t) = met[1]
          "case16" = {caseN ="case16"; n = 500; sig.e=1; sig.x = 4; sig.u = 1;   sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          "case17" = {caseN ="case17"; n = 500; sig.e=1; sig.x = 4; sig.u = 1;  sig.w=4; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, #normal Error/ beta(t) = met[1]
          "case18" = {caseN ="case18"; n = 500; sig.e=1; sig.x = 4; sig.u = 1; sig.w=16; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          
          #varying rhox   
          "case19" = {caseN ="case19"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=.1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          "case20" = {caseN ="case20"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = .25; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error / beta(t) = met[1]
          "case21" = {caseN ="case21"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = .5; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          "case22" = {caseN ="case22"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = .75; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, #normal Error/ beta(t) = met[1]
          
          #varying rho.u
          "case23" = {caseN ="case23"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = .25; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          "case24" = {caseN ="case24"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = .5; rho.m = 0; Delta=.1; t = 50; pn = ceiling(n^{1/3.8}) + 1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          "case25" = {caseN ="case25"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = .75; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error / beta(t) = met[1]
          
          #varying rhox.m 
          "case26" = {caseN ="case26"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = .0; rho.u = 0; rho.m = 0.25; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          "case27" = {caseN ="case27"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = .0; rho.u = 0; rho.m = 0.50; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, #normal Error/ beta(t) = met[1]
          "case28" = {caseN ="case28"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = .0; rho.u = 0; rho.m = 0.75; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          
          #various Delta
          "case29" = {caseN ="case29"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=.25; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          "case30" = {caseN ="case30"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=.5; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          "case31" = {caseN ="case31"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=2; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          "case32" = {caseN ="case32"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=4; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          
          
          #Sigx
          "case33" = {caseN ="case33"; n = 500; sig.e=1; sig.x = .5; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error / beta(t) = met[1]
          "case34" = {caseN ="case34"; n = 500; sig.e=1; sig.x = 1; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          "case35" = {caseN ="case35"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"}, #normal Error/ beta(t) = met[1]
          "case36" = {caseN ="case36"; n = 500; sig.e=1; sig.x = 16; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"} # normal Error/ beta(t) = met[1]
          
          
          
          #"case29" = {caseN ="case29"; n = 100; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 2; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          #"case30" = {caseN ="case30"; n = 200; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 2; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          #"case31" = {caseN ="case31"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 2; distF <- "Norm"}, # normal Error/ beta(t) = met[1]
          #"case32" = {caseN ="case32"; n = 1000; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=1; t = 50; pn = ceiling(n^{1/3.8})+2; idf = 2; distF <- "Norm"} # normal Error/ beta(t) = met[1]
          
  )
}


#--- Source importan functions
#source("Core_Function.R")
pn = 10
nrep = 5
#DataSim <- DataSim(pn, n, t, a, j0=2, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep=c("Rst", "Norm","MixNorm")[2], distFUe = c("Rmst","Mvnorm")[2], distFepWe=c("Rmst","Mvnorm")[1], CovMet =c("CS", "Ar1")[2], sig.e, idf=c(1:4)[3])
Wdist = "norm"
DataSim <- DataSimReplicate(pn, n, t, a, j0 = 2, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep=c("Rst", "Norm","MixNorm")[2], distFUe = c("Rmst","Mvnorm")[2], distFepWe=c("Rmst","Mvnorm")[2], CovMet =c("CS", "Ar1")[1], sig.e, idf=c(1:4)[1], nrep, Wdist,Z)

#--- Assign data 
Y <- DataSim$Y
W <- DataSim$Data$W
M <- DataSim$Data$W
X <- DataSim$Data$X
a <- DataSim$a
Delt <- DataSim$Delt
Z <- DataSim$Z


#-------------------------------------
# You need Warray (n x T x nRep)
#
Warray = array(NA, dim = dim(DataSim$Data$Wrep)[c(3,2,1)])

for(j in 1:dim(Warray)[3]){
  Warray[,,j] = t(DataSim$Data$Wrep[j,,])
}

pn = 10
a <- seq(0,1, length = dim(Warray)[2])
bs2 <- bs(a, df = pn, intercept = T)


#Xt <- apply(Warray, c(1,2), mean)
Xt =  as.matrix(FUI(model="gaussian", smooth=T, Warray ,silent = FALSE)) #} #- (abs(range(W)[1]) +1)
#Xt <- as.matrix(FUI(model="gaussian",smooth=T, Warray,silent = FALSE))
Wt <- (Xt%*%bs2)/length(a)


#---- Transform the data
# df_sample$AgeYRTru <- df_sample$AgeYR
# #df_sample$AgeYR <- scale(df_sample$AgeYR, center = T, scale = T)
# df_sampleOld <- df_sample
# df_sample$AgeYR <- df_sample$AgeYR/100
# df_sample$MnASTP <- df_sample$MnASTP*10
# df_sample$MnMVPA <-  df_sample$MnMVPA/100


#--- df_sample is a dataframe with the Y (response), Z (covariates); Wt (n x pn) is the functional data ()

df_sampleWt <- data.frame(Y, Z, W = Wt) #[1:200,]



#*******************************************************************************
# trial code (Cases we need to look at)
# 
#-------------------------------------------------------------------------------
#case1 (Fit the model in brms ignoring MVPA and ASTP)
#case2 (Fit the model in STAN with smoothing prior and ignoring MVPA and ASTP)
#case3 (Fit the model in brms with MVPA and ASTP)
#Case4 (Fit the model in STAN with MVPA and ASTP and smoothing prior)

#-------------------------------------------------------------------------------
#---- Data set prep. 
#-------------------------------------------------------------------------------
tau0 = .1

#form <- paste0("Y ~ ", paste(c("Gender","Race","HealthCondt2","AgeYR", colnames(df_sampleWt)[grepl("W",colnames(df_sampleWt))]), collapse = " + "))
#Dt0 <- make_standata(bf(as.formula(form), quantile = tau0), data = df_sampleWt[c(1:nrow(df_sampleWt)),], family = asym_laplace())
#, stanvars = stanvars2, prior = bprior,chains = 2,iter = 500, control = list(adapt_delta = 0.97), data2 = list(M = rep(0, 29), V = SigbPrio))


fitCase1  <- list()
fitCase1b <- list()
fitCase2  <- list()
fitCase2b <- list()
fitCase3  <- list()
fitCase4  <- list()
fitCase5  <- list()
tauVal <- c(.2,.5,.8) # (quantile values to consider)
Niter = 5000
Nchain = 2


for(l in 1:length(tauVal)){ 
  
  tau0 <- tauVal[l]
  Bd = GamBnd(tau0)[1:2]
  
  Dt0$pn = pn
  Dt0$Xold <- Dt0$X
  Dt0$X <- Dt0$Xold[,!grepl("W.[0-9]", colnames(Dt0$Xold))]
  Dt0$Wn <- Dt0$Xold[,grepl("W.[0-9]", colnames(Dt0$Xold))]; dim(Dt0$Wn)
  Dt0$M <- numeric(Dt0$pn)
  Dt0$V <- as.positive.definite(as.inverse(P.mat(pn) + diag(rep(.01, pn))))*.1
  Dt0$K <- ncol(Dt0$X)
  Bd <- GamBnd(tau0)[1:2]
  Dt0$Bd <- Bd*.99
  Dt0$tau0 <- tau0
  
  varX <- c("MnASTP", "MnMVPA")
  
  # #-------------------------------------------------------------------------------
  # # Case 1 (no MPVA and ASTP)
  # #-------------------------------------------------------------------------------
  # #@' This function is the stan_funs2 for the GAL distribution
  # 
  # Bd = GamBnd(tau0)[1:2] 
  # 
  # {
  #   GAL2 <- custom_family(
  #     "GAL2", dpars = c("mu", "sigma","gam", "tau"), links=c("identity","log","identity","identity"),
  #     lb=c(NA,0, Bd[1]*.9,0), ub=c(NA,NA, Bd[2]*.9,1), type="real") #, vars = "vint1[n]"
  #   
  #   
  #   stan_funs2 <- "
  # /*
  # A = -est*p_neg + .5*pow(gam, 2)*pow(p_neg/p_pos, 2) + log(Phi_approx(a2-a3)) + log1m_exp(fabs(log(Phi_approx(a2-a3)) - log(Phi_approx(a2)))); 
  # gam = (gamU - gamL) * ligam + gamL;
  # real gam = (gamU - gamL) * ligam + gamL;
  # real GAL2_lpdf(real y, real mu, real sigma, real ligam, real tau, real gamL, real gamU){
  # real GAL2_rng(real mu, real sigma, real ligam, real tau, real gamL, real gamU){
  #  */
  #    /* helper function for asym_laplace_lpdf
  # * Args:
  #   *   y: the response value
  # *   tau: quantile parameter in (0, 1)
  # */
  #   real rho_quantile(real y, real tau) {
  #     if (y < 0) {
  #       return y * (tau - 1);
  #     } else {
  #       return y * tau;
  #     }
  #   }
  # /* asymmetric laplace log-PDF for a single response
  # * Args:
  #   *   y: the response value
  # *   mu: location parameter
  # *   sigma: positive scale parameter
  # *   tau: quantile parameter in (0, 1)
  # * Returns:
  #   *   a scalar to be added to the log posterior
  # */
  #   real asym_laplace_lpdf(real y, real mu, real sigma, real tau) {
  #     return log(tau * (1 - tau)) -
  #       log(sigma) -
  #       rho_quantile((y - mu) / sigma, tau);
  #   }
  # /* asymmetric laplace log-CDF for a single quantile
  # * Args:
  #   *   y: a quantile
  # *   mu: location parameter
  # *   sigma: positive scale parameter
  # *   tau: quantile parameter in (0, 1)
  # * Returns:
  #   *   a scalar to be added to the log posterior
  # */
  #   real asym_laplace_lcdf(real y, real mu, real sigma, real tau) {
  #     if (y < mu) {
  #       return log(tau) + (1 - tau) * (y - mu) / sigma;
  #     } else {
  #       return log1m((1 - tau) * exp(-tau * (y - mu) / sigma));
  #     }
  #   }
  # /* asymmetric laplace log-CCDF for a single quantile
  # * Args:
  #   *   y: a quantile
  # *   mu: location parameter
  # *   sigma: positive scale parameter
  # *   tau: quantile parameter in (0, 1)
  # * Returns:
  #   *   a scalar to be added to the log posterior
  # */
  #   real asym_laplace_lccdf(real y, real mu, real sigma, real tau) {
  #     if (y < mu) {
  #       return log1m(tau * exp((1 - tau) * (y - mu) / sigma));
  #     } else {
  #       return log1m(tau) - tau * (y - mu) / sigma;
  #     }
  #   }
  #  
  #  real GAL2_lpdf(real y, real mu, real sigma, real gam, real tau){
  #  
  #  real p_pos;
  #  real p_neg;
  #  real a3;
  #  real a2;
  #  real p;
  #  real est;
  #  real A;
  #  real B;
  #  real Res = 0;
  #  //real gam = ligam;
  #   p = 1 * (gam < 0) + (tau - 1 * (gam < 0))/(2*Phi(-fabs(gam)) * exp(.5 * pow(gam, 2)));
  #   p_pos = p - 1 * (gam > 0);
  #   p_neg = p -  1* (gam < 0);  
  #   est = (y - mu) / sigma; 
  #   
  #   if(fabs(gam) > 0){
  #   a3 = p_pos * (est / fabs(gam));
  #   a2 = fabs(gam) * (p_neg / p_pos);
  #   
  #   
  #   if(est/gam > 0){
  #     A =  0.5 * pow(gam, 2) * pow(p_neg/p_pos, 2) - est * p_neg + log_diff_exp(log(Phi_approx(a2-a3)), log(Phi_approx(a2)) ); 
  #     B =  0.5 * pow(gam, 2) - p_pos * est + log(Phi_approx(-fabs(gam) + a3));
  #     Res = log(2*p*(1-p)) - log(sigma) +  log_sum_exp(A, B);
  #   }else{
  #     Res =  log(2*p*(1-p)) - log(sigma) - p_pos * est + 0.5 * pow(gam, 2) + log(Phi_approx(-fabs(gam) ));
  #   }
  #   }else{
  #   Res = asym_laplace_lpdf( y | mu, sigma, tau); 
  #   }
  #    
  #   return Res;
  #  }
  # 
  # real GAL2_rng(real mu, real sigma, real ligam, real tau){
  # 
  #    real A;
  #    real B;
  #    real C;
  #    real p;
  #    real hi;
  #    real nui;
  #    real mui=0;
  #    real Up = uniform_rng(.5, 1.0);
  #    
  #    real gam = ligam;
  #    p = (gam < 0) + (tau - (gam < 0))/(2*Phi_approx(-fabs(gam))*exp(.5*pow(gam, 2)));
  #    A = (1 - 2*p)/(p - pow(p,2));
  #    B = 2/(p - pow(p,2));
  #    C = 1/((gam > 0) - p);
  #    
  #     hi = sigma * inv_Phi(Up);
  #    nui = sigma * exponential_rng(1);
  #    mui += mu + A * nui + C * fabs(gam) * hi;
  # 
  #    return normal_rng(mui, sqrt(sigma*B*nui));
  # }
  # "
  # }  
  # 
  # #--- Now define all of these here
  # form <- paste0("Y ~ ", paste(c("Gender","Race","HealthCondt2","AgeYR", colnames(df_sampleWt)[grepl("W",colnames(df_sampleWt))]), collapse = " + "))
  # 
  # stanvars2 <- stanvar(scode = stan_funs2, block = "functions")
  # fitCase1[[l]] <- brms::brm(bf(as.formula(form), tau = tau0), data = df_sampleWt, family = GAL2, stanvars = stanvars2,
  #                            chains = Nchain,iter = Niter, control = list(adapt_delta = 0.99), cores = 2, seed = 1123) # init = 0.1,
  # 
  # 
  # #-------------------------------------------------------------------------------
  # # Case 1b (with MPVA and ASTP) - Wt
  # #-------------------------------------------------------------------------------
  # #@' This function is the stan_funs2 for the GAL distribution
  # 
  # Bd = GamBnd(tau0)[1:2]
  # 
  # {
  #   GAL2 <- custom_family(
  #     "GAL2", dpars = c("mu", "sigma","gam", "tau"), links=c("identity","log","identity","identity"),
  #     lb=c(NA,0, Bd[1]*.9,0), ub=c(NA,NA, Bd[2]*.9,1), type="real") #, vars = "vint1[n]"
  #   
  #   
  #   stan_funs2 <- "
  # /*
  # A = -est*p_neg + .5*pow(gam, 2)*pow(p_neg/p_pos, 2) + log(Phi_approx(a2-a3)) + log1m_exp(fabs(log(Phi_approx(a2-a3)) - log(Phi_approx(a2)))); 
  # gam = (gamU - gamL) * ligam + gamL;
  # real gam = (gamU - gamL) * ligam + gamL;
  # real GAL2_lpdf(real y, real mu, real sigma, real ligam, real tau, real gamL, real gamU){
  # real GAL2_rng(real mu, real sigma, real ligam, real tau, real gamL, real gamU){
  #  */
  #    /* helper function for asym_laplace_lpdf
  # * Args:
  #   *   y: the response value
  # *   tau: quantile parameter in (0, 1)
  # */
  #   real rho_quantile(real y, real tau) {
  #     if (y < 0) {
  #       return y * (tau - 1);
  #     } else {
  #       return y * tau;
  #     }
  #   }
  # /* asymmetric laplace log-PDF for a single response
  # * Args:
  #   *   y: the response value
  # *   mu: location parameter
  # *   sigma: positive scale parameter
  # *   tau: quantile parameter in (0, 1)
  # * Returns:
  #   *   a scalar to be added to the log posterior
  # */
  #   real asym_laplace_lpdf(real y, real mu, real sigma, real tau) {
  #     return log(tau * (1 - tau)) -
  #       log(sigma) -
  #       rho_quantile((y - mu) / sigma, tau);
  #   }
  # /* asymmetric laplace log-CDF for a single quantile
  # * Args:
  #   *   y: a quantile
  # *   mu: location parameter
  # *   sigma: positive scale parameter
  # *   tau: quantile parameter in (0, 1)
  # * Returns:
  #   *   a scalar to be added to the log posterior
  # */
  #   real asym_laplace_lcdf(real y, real mu, real sigma, real tau) {
  #     if (y < mu) {
  #       return log(tau) + (1 - tau) * (y - mu) / sigma;
  #     } else {
  #       return log1m((1 - tau) * exp(-tau * (y - mu) / sigma));
  #     }
  #   }
  # /* asymmetric laplace log-CCDF for a single quantile
  # * Args:
  #   *   y: a quantile
  # *   mu: location parameter
  # *   sigma: positive scale parameter
  # *   tau: quantile parameter in (0, 1)
  # * Returns:
  #   *   a scalar to be added to the log posterior
  # */
  #   real asym_laplace_lccdf(real y, real mu, real sigma, real tau) {
  #     if (y < mu) {
  #       return log1m(tau * exp((1 - tau) * (y - mu) / sigma));
  #     } else {
  #       return log1m(tau) - tau * (y - mu) / sigma;
  #     }
  #   }
  #  
  #  real GAL2_lpdf(real y, real mu, real sigma, real gam, real tau){
  #  
  #  real p_pos;
  #  real p_neg;
  #  real a3;
  #  real a2;
  #  real p;
  #  real est;
  #  real A;
  #  real B;
  #  real Res = 0;
  #  //real gam = ligam;
  #   p = 1 * (gam < 0) + (tau - 1 * (gam < 0))/(2*Phi(-fabs(gam)) * exp(.5 * pow(gam, 2)));
  #   p_pos = p - 1 * (gam > 0);
  #   p_neg = p -  1* (gam < 0);  
  #   est = (y - mu) / sigma; 
  #   
  #   if(fabs(gam) > 0){
  #   a3 = p_pos * (est / fabs(gam));
  #   a2 = fabs(gam) * (p_neg / p_pos);
  #   
  #   
  #   if(est/gam > 0){
  #     A =  0.5 * pow(gam, 2) * pow(p_neg/p_pos, 2) - est * p_neg + log_diff_exp(log(Phi_approx(a2-a3)), log(Phi_approx(a2)) ); 
  #     B =  0.5 * pow(gam, 2) - p_pos * est + log(Phi_approx(-fabs(gam) + a3));
  #     Res = log(2*p*(1-p)) - log(sigma) +  log_sum_exp(A, B);
  #   }else{
  #     Res =  log(2*p*(1-p)) - log(sigma) - p_pos * est + 0.5 * pow(gam, 2) + log(Phi_approx(-fabs(gam) ));
  #   }
  #   }else{
  #   Res = asym_laplace_lpdf( y | mu, sigma, tau); 
  #   }
  #    
  #   return Res;
  #  }
  # 
  # real GAL2_rng(real mu, real sigma, real ligam, real tau){
  # 
  #    real A;
  #    real B;
  #    real C;
  #    real p;
  #    real hi;
  #    real nui;
  #    real mui=0;
  #    real Up = uniform_rng(.5, 1.0);
  #    
  #    real gam = ligam;
  #    p = (gam < 0) + (tau - (gam < 0))/(2*Phi_approx(-fabs(gam))*exp(.5*pow(gam, 2)));
  #    A = (1 - 2*p)/(p - pow(p,2));
  #    B = 2/(p - pow(p,2));
  #    C = 1/((gam > 0) - p);
  #    
  #     hi = sigma * inv_Phi(Up);
  #    nui = sigma * exponential_rng(1);
  #    mui += mu + A * nui + C * fabs(gam) * hi;
  # 
  #    return normal_rng(mui, sqrt(sigma*B*nui));
  # }
  # "
  # }  
  # 
  # #--- Now define all of these here
  # formCase1b <- paste0("Y ~ ", paste(c("Gender","Race","HealthCondt2","AgeYR"), collapse = " + "))
  # for(pa in varX){
  #   formCase1b <- paste0(formCase1b, " + s(", pa, ", bs='ps', k=11)")
  # }
  # 
  # stanvars2 <- stanvar(scode = stan_funs2, block = "functions")
  # fitCase1b[[l]] <- brms::brm(bf(as.formula(formCase1b), tau = tau0), data = df_sampleWt, family = GAL2, stanvars = stanvars2,
  #                             chains = Nchain,iter = Niter, control = list(adapt_delta = 0.99), cores = 2,seed = 1123) # init = 0.1,
  # 
  # 
  # 
  #-------------------------------------------------------------------------------
  # Case 2 - Without MVPA and ASTP (smooth Xt)
  #-------------------------------------------------------------------------------
  {
    StandCodeVGalLin <- "
      // > fit_fakeqGal3a.5$model
      // generated with brms 2.18.0
      functions {
        
        /*
        A = -est*p_neg + .5*pow(gam, 2)*pow(p_neg/p_pos, 2) + log(Phi_approx(a2-a3)) + log1m_exp(fabs(log(Phi_approx(a2-a3)) - log(Phi_approx(a2)))); 
        gam = (gamU - gamL) * ligam + gamL;
        real gam = (gamU - gamL) * ligam + gamL;
        real GAL2_lpdf(real y, real mu, real sigma, real ligam, real tau, real gamL, real gamU){
        real GAL2_rng(real mu, real sigma, real ligam, real tau, real gamL, real gamU){
         */
           /* helper function for asym_laplace_lpdf
        * Args:
          *   y: the response value
        *   tau: quantile parameter in (0, 1)
        */
          real rho_quantile(real y, real tau) {
            if (y < 0) {
              return y * (tau - 1);
            } else {
              return y * tau;
            }
          }
        /* asymmetric laplace log-PDF for a single response
        * Args:
          *   y: the response value
        *   mu: location parameter
        *   sigma: positive scale parameter
        *   tau: quantile parameter in (0, 1)
        * Returns:
          *   a scalar to be added to the log posterior
        */
          real asym_laplace_lpdf(real y, real mu, real sigma, real tau) {
            return log(tau * (1 - tau)) -
              log(sigma) -
              rho_quantile((y - mu) / sigma, tau);
          }
        /* asymmetric laplace log-CDF for a single quantile
        * Args:
          *   y: a quantile
        *   mu: location parameter
        *   sigma: positive scale parameter
        *   tau: quantile parameter in (0, 1)
        * Returns:
          *   a scalar to be added to the log posterior
        */
          real asym_laplace_lcdf(real y, real mu, real sigma, real tau) {
            if (y < mu) {
              return log(tau) + (1 - tau) * (y - mu) / sigma;
            } else {
              return log1m((1 - tau) * exp(-tau * (y - mu) / sigma));
            }
          }
        /* asymmetric laplace log-CCDF for a single quantile
        * Args:
          *   y: a quantile
        *   mu: location parameter
        *   sigma: positive scale parameter
        *   tau: quantile parameter in (0, 1)
        * Returns:
          *   a scalar to be added to the log posterior
        */
          real asym_laplace_lccdf(real y, real mu, real sigma, real tau) {
            if (y < mu) {
              return log1m(tau * exp((1 - tau) * (y - mu) / sigma));
            } else {
              return log1m(tau) - tau * (y - mu) / sigma;
            }
          }
         
         real GAL2_lpdf(real y, real mu, real sigma, real gam, real tau){
         
         real p_pos;
         real p_neg;
         real a3;
         real a2;
         real p;
         real est;
         real A;
         real B;
         real Res = 0;
         //real gam = ligam;
          p = 1 * (gam < 0) + (tau - 1 * (gam < 0))/(2*Phi(-fabs(gam))*exp(.5*pow(gam, 2)));
          p_pos = p -  1 * (gam > 0);
          p_neg = p -  1 * (gam < 0);  
          est = (y - mu) / sigma; 
          
          if(fabs(gam) > 0){
          a3 = p_pos * (est / fabs(gam));
          a2 = fabs(gam) * (p_neg / p_pos);
          
          
          if(est/gam > 0){
            A =  0.5*pow(gam, 2)*pow(p_neg/p_pos, 2) - est*p_neg + log_diff_exp(log(Phi_approx(a2-a3)), log(Phi_approx(a2)) ); 
            B =  0.5*pow(gam, 2) - p_pos*est + log(Phi_approx(-fabs(gam) + a3));
            Res = log(2*p*(1-p)) - log(sigma) +  log_sum_exp(A, B);
          }else{
            Res =  log(2*p*(1-p)) - log(sigma) - p_pos * est + 0.5 * pow(gam, 2) + log(Phi_approx(-fabs(gam) ));
          }
          }else{
          Res = asym_laplace_lpdf( y | mu, sigma, tau); 
          }
           
          return Res;
         }
        
        real GAL2_rng(real mu, real sigma, real gam, real tau){
        
           real A;
           real B;
           real C;
           real p;
           real hi;
           real nui;
           real mui=0;
           real Up = uniform_rng(.5, 1.0);
           
          // real gam = ligam;
           p = (gam < 0) + (tau - (gam < 0))/(2*Phi_approx(-fabs(gam))*exp(.5*pow(gam, 2)));
           A = (1 - 2*p)/(p - pow(p,2));
           B = 2/(p - pow(p,2));
           C = 1/((gam > 0) - p);
           
            hi = sigma * inv_Phi(Up);
           nui = sigma * exponential_rng(1);
           mui += mu + A * nui + C * fabs(gam) * hi;
        
           return normal_rng(mui, sqrt(sigma*B*nui));
        }
        
      }
      data {
        int<lower=1> pn; //Number of basis;
        int<lower=1> N;  // total number of observations
        vector[N] Y;  // response variable
        int<lower=1> K;  // number of population-level effects
        matrix[N, K] X;  // population-level design matrix
        int prior_only;  // should the likelihood be ignored?
        matrix[N,pn] Wn;
          vector[pn] M;
         matrix[pn, pn] V;
         vector[2] Bd;
         real<lower=0,upper=1> tau0;
      }
      transformed data {
        int Kc = K - 1;
        matrix[N, Kc] Xc;  // centered version of X without an intercept
        vector[Kc] means_X;  // column means of X before centering
        for (i in 2:K) {
          means_X[i - 1] = mean(X[, i]);
          Xc[, i - 1] = X[, i] - means_X[i - 1];
        }
      }
      parameters {
        vector[Kc] b;  // population-level effects
        real Intercept;  // temporary intercept for centered predictors
        vector[pn] bw;
        real<lower=0.01,upper=.99> psi;
         
        // standarized spline coefficients
        //real<lower=0> sigma;  // dispersion parameter
        real<lower=Bd[1],upper=Bd[2]> gam;
        real<lower=0> teta;
      }
      transformed parameters {
         //bwa ar(1) stationary prior
         vector[pn] bwa;
        real tau = tau0;
        real lprior = 0;  // prior contributions to the log posterior
        // compute bwa
        bwa[1] = bw[1];
        for(i in 2:pn)
          bwa[i] = psi*bwa[i-1] + bw[i]*teta; 
          
        lprior += normal_lpdf(b | 0.0, 10);
        lprior += std_normal_lpdf(bw);
        lprior += uniform_lpdf(psi | 0.01, .99);
        lprior += uniform_lpdf(gam | Bd[1], Bd[2]);
        //lprior += cauchy_lpdf(teta| 0, 1) - 1* cauchy_lccdf(0| 0, 1);
        //lprior += gamma_lpdf(teta| 1, 1);
        //lprior += weibull_lpdf(teta| 1, 2.5);
        lprior += std_normal_lpdf(teta) - 1* log(0.5);
        lprior += student_t_lpdf(Intercept | 3, 28.7, 5.6);
        //lprior += normal_lpdf(bs | 0, 5);
        // lprior += student_t_lpdf(sds_1_1 | 3, 0, 5.6)
        //   - 1 * student_t_lccdf(0 | 3, 0, 5.6);
        // lprior += student_t_lpdf(sds_2_1 | 3, 0, 5.6)
        //   - 1 * student_t_lccdf(0 | 3, 0, 5.6);
        lprior += student_t_lpdf(sigma | 3, 0, 5.6)
          - 1 * student_t_lccdf(0 | 3, 0, 5.6);
        lprior += student_t_lpdf(Intercept_sigma | 3, 0, 2.5);
       lprior += student_t_lpdf(sds_sigma_1_1 | 3, 0, 2.7)
          - 1 * student_t_lccdf(0 | 3, 0, 2.7);
      }
      model {
        // likelihood including constants
        if (!prior_only) {
          // initialize linear predictor term + Xs * bs
          vector[N] mu = rep_vector(0.0, N);
          vector[N] sigma = rep_vector(0.0, N);
          mu += Intercept + Xc * b + Wn * bwa;
       sigma += Intercept_sigma + Xs_sigma * bs_sigma + Zs_sigma_1_1 * s_sigma_1_1;
        sigma = exp(sigma);
          for (n in 1:N) {
          //  target += GAL2_lpdf(Y[n] | mu[n], sigma, ligam, tau0);
              target += GAL2_lpdf(Y[n] | mu[n], sigma, gam, tau0);
          }
        }
        // priors including constants
        target += lprior;
        target += std_normal_lpdf(zs_sigma_1_1);
        //target += std_normal_lpdf(zs_1_1);
        //target += std_normal_lpdf(zs_2_1);
      }
      generated quantities {
        // actual population-level intercept
        real b_Intercept = Intercept - dot_product(means_X, b);
        vector[N] mu = rep_vector(0.0, N);
        vector[N] log_lik = rep_vector(0.0, N);
          mu += Intercept + Xc * b + Wn * bwa;
        for(n in 1:N)
              log_lik[n] +=  GAL2_lpdf(Y[n] | mu[n], sigma, gam, tau0);

        //Vector[N] ASTP = Xs[1:N,1] * bs[1] + Zs_1_1 * s_1_1;
        //vector[N] MVPA = Xs[1:N,2] * bs[2] + Zs_2_1 * s_2_1;
      }
      "
  }
  
  #--- Below is what you will use - discard the stan code above!
  {
    StandCodeVGalLin <- "
    
    
functions {
  
    /*
    A = -est*p_neg + .5*pow(gam, 2)*pow(p_neg/p_pos, 2) + log(Phi_approx(a2-a3)) + log1m_exp(fabs(log(Phi_approx(a2-a3)) - log(Phi_approx(a2))));
  gam = (gamU - gamL) * ligam + gamL;
  real gam = (gamU - gamL) * ligam + gamL;
  real GAL2_lpdf(real y, real mu, real sigma, real ligam, real tau, real gamL, real gamU){
  real GAL2_rng(real mu, real sigma, real ligam, real tau, real gamL, real gamU){
   */
     /* helper function for asym_laplace_lpdf
  * Args:
    *   y: the response value
  *   tau: quantile parameter in (0, 1)
  */
    real rho_quantile(real y, real tau) {
      if (y < 0) {
        return y * (tau - 1);
      } else {
        return y * tau;
      }
    }
  /* asymmetric laplace log-PDF for a single response
  * Args:
    *   y: the response value
  *   mu: location parameter
  *   sigma: positive scale parameter
  *   tau: quantile parameter in (0, 1)
  * Returns:
    *   a scalar to be added to the log posterior
  */
    real asym_laplace_lpdf(real y, real mu, real sigma, real tau) {
      return log(tau * (1 - tau)) -
        log(sigma) -
        rho_quantile((y - mu) / sigma, tau);
    }
  /* asymmetric laplace log-CDF for a single quantile
  * Args:
    *   y: a quantile
  *   mu: location parameter
  *   sigma: positive scale parameter
  *   tau: quantile parameter in (0, 1)
  * Returns:
    *   a scalar to be added to the log posterior
  */
    real asym_laplace_lcdf(real y, real mu, real sigma, real tau) {
      if (y < mu) {
        return log(tau) + (1 - tau) * (y - mu) / sigma;
      } else {
        return log1m((1 - tau) * exp(-tau * (y - mu) / sigma));
      }
    }
  /* asymmetric laplace log-CCDF for a single quantile
  * Args:
    *   y: a quantile
  *   mu: location parameter
  *   sigma: positive scale parameter
  *   tau: quantile parameter in (0, 1)
  * Returns:
    *   a scalar to be added to the log posterior
  */
    real asym_laplace_lccdf(real y, real mu, real sigma, real tau) {
      if (y < mu) {
        return log1m(tau * exp((1 - tau) * (y - mu) / sigma));
      } else {
        return log1m(tau) - tau * (y - mu) / sigma;
      }
    }

   real GAL2_lpdf(real y, real mu, real sigma, real ligam, real tau){

   real p_pos;
   real p_neg;
   real a3;
   real a2;
   real p;
   real est;
   real A;
   real B;
   real Res = 0;
   real gam = ligam;
    p = (gam < 0) + (tau - (gam < 0))/(2*Phi(-fabs(gam))*exp(.5*pow(gam, 2)));
    p_pos = p -  (gam > 0);
    p_neg = p -  (gam < 0);
    est = (y - mu) / sigma;

    if(fabs(gam) > 0){
    a3 = p_pos * (est / fabs(gam));
    a2 = fabs(gam) * (p_neg / p_pos);


    if(est/gam > 0){
      A =  0.5*pow(gam, 2)*pow(p_neg/p_pos, 2) - est*p_neg + log_diff_exp(log(Phi_approx(a2-a3)), log(Phi_approx(a2)) );
      B =  0.5*pow(gam, 2) - p_pos*est + log(Phi_approx(-fabs(gam) + a3));
      Res = log(2*p*(1-p)) - log(sigma) +  log_sum_exp(A, B);
    }else{
      Res =  log(2*p*(1-p)) - log(sigma) - p_pos * est + 0.5 * pow(gam, 2) + log(Phi_approx(-fabs(gam) ));
    }
    }else{
    Res = asym_laplace_lpdf( y | mu, sigma, tau);
    }

    return Res;
   }

  real GAL2_rng(real mu, real sigma, real ligam, real tau){

     real A;
     real B;
     real C;
     real p;
     real hi;
     real nui;
     real mui=0;
     real Up = uniform_rng(.5, 1.0);

     real gam = ligam;
     p = (gam < 0) + (tau - (gam < 0))/(2*Phi_approx(-fabs(gam))*exp(.5*pow(gam, 2)));
     A = (1 - 2*p)/(p - pow(p,2));
     B = 2/(p - pow(p,2));
     C = 1/((gam > 0) - p);

      hi = sigma * inv_Phi(Up);
     nui = sigma * exponential_rng(1);
     mui += mu + A * nui + C * fabs(gam) * hi;

     return normal_rng(mui, sqrt(sigma*B*nui));
  }
  
}
data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  // data for splines
  int Ks;  // number of linear effects
  matrix[N, Ks] Xs;  // design matrix for the linear effects
  // data for spline s(x)
  int nb_1;  // number of bases
  int knots_1[nb_1];  // number of knots
  // basis function matrices
  matrix[N, knots_1[1]] Zs_1_1;
  // data for splines
  int Ks_sigma;  // number of linear effects
  matrix[N, Ks_sigma] Xs_sigma;  // design matrix for the linear effects
  // data for spline s(x)
  int nb_sigma_1;  // number of bases
  int knots_sigma_1[nb_sigma_1];  // number of knots
  // basis function matrices
  matrix[N, knots_sigma_1[1]] Zs_sigma_1_1;
  int prior_only;  // should the likelihood be ignored?
  
  // Below are the values you need to append to the data created by Brms
  int pn; // Number of basis function and the is the number of columns for Wn
  vector[2] Bd; // the boundary values for Bd
  matrix[N,pn] Wn; // the projected values of Wn
  real<lower=0,upper=1> tau0;
}
transformed data {
}
parameters {
  real Intercept;  // temporary intercept for centered predictors
  vector[Ks] bs;  // spline coefficients
  // parameters for spline s(x)
  // standarized spline coefficients
  // Prior specification for the Ar prior on the the coefficents for Wn
  vector[pn] bw;
  real<lower=0.01,upper=.99> psi;
  real<lower=0> teta;
  vector[knots_1[1]] zs_1_1;
  real<lower=0> sds_1_1;  // standard deviations of spline coefficients
  real Intercept_sigma;  // temporary intercept for centered predictors
  vector[Ks_sigma] bs_sigma;  // spline coefficients
  // parameters for spline s(x)
  // standarized spline coefficients
  vector[knots_sigma_1[1]] zs_sigma_1_1;
  real<lower=0> sds_sigma_1_1;  // standard deviations of spline coefficients
  real<lower=Bd[1],upper=Bd[2]> gam;
}
transformed parameters {
  // actual spline coefficients
  vector[knots_1[1]] s_1_1;
  // actual spline coefficients
  vector[knots_sigma_1[1]] s_sigma_1_1;
  real tau = 0.25;
  real lprior = 0;  // prior contributions to the log posterior
  //bwa ar(1) stationary prior
  vector[pn] bwa;
  //real tau = tau0;
        // compute bwa
  bwa[1] = bw[1];
  for(i in 2:pn)
  bwa[i] = psi*bwa[i-1] + bw[i]*teta;
  
  
  lprior += std_normal_lpdf(bw);
  lprior += uniform_lpdf(psi | 0.01, .99);
  lprior += uniform_lpdf(gam | Bd[1], Bd[2]);
  // compute actual spline coefficients
  s_1_1 = sds_1_1 * zs_1_1;
  // compute actual spline coefficients
  s_sigma_1_1 = sds_sigma_1_1 * zs_sigma_1_1;
  lprior += student_t_lpdf(Intercept | 3, 0.1, 2.7);
  lprior += student_t_lpdf(sds_1_1 | 3, 0, 2.7)
    - 1 * student_t_lccdf(0 | 3, 0, 2.7);
  lprior += student_t_lpdf(Intercept_sigma | 3, 0, 2.5);
  lprior += student_t_lpdf(sds_sigma_1_1 | 3, 0, 2.7)
    - 1 * student_t_lccdf(0 | 3, 0, 2.7);
}
model {
  // likelihood including constants
  if (!prior_only) {
    // initialize linear predictor term
    vector[N] mu = rep_vector(0.0, N);
    // initialize linear predictor term
    vector[N] sigma = rep_vector(0.0, N);
    // note I have added the term Wn * bwa 
    mu += Intercept + Xs * bs + Zs_1_1 * s_1_1 + Wn * bwa;
    sigma += Intercept_sigma + Xs_sigma * bs_sigma + Zs_sigma_1_1 * s_sigma_1_1 + Wn * bwa;
    sigma = exp(sigma);
    for (n in 1:N) {
      target += GAL2_lpdf(Y[n] | mu[n], sigma[n], gam, tau0);
    }
  }
  // priors including constants
  target += lprior;
  target += std_normal_lpdf(zs_1_1);
  target += std_normal_lpdf(zs_sigma_1_1);
}
generated quantities {
  // actual population-level intercept
  real b_Intercept = Intercept;
  // actual population-level intercept
  real b_sigma_Intercept = Intercept_sigma;
 // I added this part so swe cna compare model performance  
 vector[N] mu = rep_vector(0.0, N);
 vector[N] sigma = rep_vector(0.0, N);
 vector[N] log_lik = rep_vector(0.0, N);
 
 mu += Intercept + Xs * bs + Zs_1_1 * s_1_1 + Wn * bwa;
sigma += Intercept_sigma + Xs_sigma * bs_sigma + Zs_sigma_1_1 * s_sigma_1_1 + Wn * bwa;
    sigma = exp(sigma);
  for(n in 1:N)
    log_lik[n] +=  GAL2_lpdf(Y[n] | mu[n], sigma[n], gam, tau0);

}
    
    
    
    "
    
  }
  
  df2 <- df_sampleWt
  
  #form2 <- paste0("Y ~ ", paste(c("Gender","Race","HealthCondt2","AgeYR", colnames(df2)[grepl("W",colnames(df2))]), collapse = " + "))
  
  #--- You need to have two formula  - one for the mean and the other for the sigma
  #-- Linear effects 
  form_mu = paste0("Y ~", paste(c(colnames(df_sampleWt)[grepl("X",colnames(df_sampleWt))], colnames(df_sampleWt)[grepl("W",colnames(df_sampleWt))]), collapse = " + ")) # formula for the response
  form_mu
  
  #--- Non linear effect
  form_mu = paste0("Y ~ X1 + X2 + s(X3) + ", paste(c(colnames(df_sampleWt)[grepl("W",colnames(df_sampleWt))]), collapse = " + ")) # formula for the response
  form_mu  
  
   
  #--- Linear effect on sigma
  form_sigma = paste0("sigma ~", paste(c(colnames(df_sampleWt)[grepl("X",colnames(df_sampleWt))], colnames(df_sampleWt)[grepl("W",colnames(df_sampleWt))]), collapse = " + ")) # formula for the response  # formula for log(sigma)
  
  #--- Non-linear effect
  form_sigma = paste0("sigma ~ X1 + X2 + s(X3) + ", paste(c(colnames(df_sampleWt)[grepl("W",colnames(df_sampleWt))]), collapse = " + ")) # formula for the response
  
  
  
  #--- Replace here with a model formula so that brms can create the data
    #-- The line below will create the data you need but you ned to trick it a littiel bit
  Dt2 <- make_standata(bf(as.formula(form_mu), as.formula(form_sigma), quantile = tau0), data = df2[c(1:nrow(df2)),], family = asym_laplace())

  
  Dt2$pn = pn
  Dt2$Xold <- Dt2$X
  Dt2$X <- Dt2$Xold[,!grepl("W.[0-9]", colnames(Dt2$Xold))]
  Dt2$Wn <- Dt2$Xold[,grepl("W.[0-9]", colnames(Dt2$Xold))]; dim(Dt2$Wn)
  Dt2$M <- numeric(Dt2$pn)
  Dt2$K <- ncol(Dt2$X)
  Bd <- GamBnd(tau0)[1:2]
  Dt2$Bd <- Bd*.9
  Dt2$tau0 <- tau0
  #Dt2$X[,ncol(Dt2$X)] <- Dt2$X[,ncol(Dt2$X)]/100
  
  
  fitCase2[[l]] <- stan(model_code = StandCodeVGalLin,
                        data = Dt2, iter = Niter, chains = Nchain, verbose = TRUE,control = list(adapt_delta = 0.99,max_treedepth=11), cores = 2, seed = 1123)
  
  
  #--- Posterior analysis
  
  #--- Takes 2 output
  PostAnalyStan <- function(fit,  Bx=bs2, tau0, df0=10, smoothT = F){
    
    #----------------------------------------------------
    met <- c("f1","f2","f3","f4","f5")
    #---  Beta function
    Betafunc <- function(t, met){
      switch(met,
             f1 = 1.*sin(2*pi*t) + 1.5 ,
             f2 = 1/(1+exp(4*2*(t-.5))),
             f3 = 1.*sin(pi*(8*(t-.5))/2)/ (1 + (2*(8*(t-.5))^2)*(sign(8*(t-.5))+1)) + 1.5,
             f4 = sin(pi*(16*(t-.5))/2)/ (1 + (2*(16*(t-.5))^2)*(.5*sin(8*(t-.5))+1)),
             f5 = sin(pi*(16*(t-.5))/2)/ (1 + (2*(16*(t-.5))^2)*(sin(8*(t-.5))+1))
      )
    }
    idf = 1
    
    
    #----------------------------------------------------
    
    PostMat <- as.matrix(fit)
    tau <-  PostMat[1, grepl("tau",colnames(PostMat))]
    b <- PostMat[, c("Intercept","b_Intercept", paste0("bs[",1,"]"))]
    Gam <- PostMat[, grepl("bwa", colnames(PostMat))] # 
    #Gamx <- PostMatx[, grepl("bwa", colnames(PostMatx))] #
    
    
    #--- Format time to hh:MM
    timeHM_formatter <- function(x) {
      h <- floor(x/60)
      m <- floor(x %% 60)
      lab <- sprintf("%d:%02d", h, m) # Format the strings as HH:MM
      return(lab)
    }
    
    
    
    if(smoothT){
      betastp <- PostMat[,c("bs[1]", grepl("s_1_1", colnames(PostMat)))] 
      betmvpa <- PostMat[,c("bs[2]", grepl("s_2_1", colnames(PostMat)))]
      
      betastpx <- PostMat[,c("bs[1]", grepl("s_1_1", colnames(PostMatx)))]
      betmvpax <- PostMat[,c("bs[2]", grepl("s_2_1", colnames(PostMatx)))]
      
      #--- Now collect basis and all
      Zastp <- cbind(Dt0$Xs[,1], Dt0$Zs_1_1)
      Zmvpa <- cbind(Dt0$Xs[,2], Dt0$Zs_2_1)
    }
    # bs <- PostMat[, grepl("bs", colnames(PostMat))] # 
    #Gam <- PostMat[, grepl("b_W", colnames(PostMat))] #
    #---------------------------------------------------------------------------
    #--- Plot the estimates
    #---------------------------------------------------------------------------
    Fw <- Gam %*% t(Bx)
    #Fwx <- Gamx %*% t(Bx)
    
    if(smoothT){
      AstpF <- betastp %*% t(Zastp)
      MvpaF <- betmvpa %*% t(Zmvpa)
      
      AstpFx <- betastpx %*% t(Zastp)
      MvpaFx <- betmvpax %*% t(Zmvpa)
    }
    
    FwSmth <- matrix(NA, ncol=ncol(Fw), nrow=nrow(Fw))
    FwSmthx <- matrix(NA, ncol=ncol(Fw),nrow=nrow(Fw))
    
    #--- Smooth these results
    if(smoothT){
      AstpFsm <- matrix(NA, ncol=ncol(AstpF), nrow=nrow(AstpF))
      MvpaFsm <- matrix(NA, ncol=ncol(MvpaF), nrow=nrow(MvpaF))
      
      AstpFxsm <- matrix(NA, ncol=ncol(AstpFx), nrow=nrow(AstpFx))
      MvpaFxsm <- matrix(NA, ncol=ncol(MvpaFx), nrow=nrow(MvpaFx))
    }
    
    x = seq(-.1,1.1,length = ncol(Fw))
    for(i in 1:nrow(Fw)){
      FwSmth[i,] <- smooth.spline(x,Fw[i,], df=df0)$y
     FwSmthx[i,] <- 1.5*((Betafunc(x,met[idf]) - mean(Betafunc(x,met[idf])))*c(1,2,2, 2)[idf] + qnorm(tau)*((Betafunc(x,met[idf]) - mean(Betafunc(x,met[idf])))*c(1,2,2, 2)[idf]))
      
      if(smoothT){
        AstpFsm[i,] <- smooth.spline(Zastp[,1],AstpF[i,], df=df0)$y
        MvpaFsm[i,] <- smooth.spline(Zmvpa[,1],MvpaF[i,], df=df0)$y
        
        AstpFxsm[i,] <- smooth.spline(Zastp[,1],AstpFx[i,], df=df0)$y
        MvpaFxsm[i,] <- smooth.spline(Zmvpa[,1],MvpaFx[i,], df=df0)$y
      }
    }
    id <- which(x > 0 & x < 1)
    x <- x[id]
    Fw <- Fw[,id]
    FwSmth <- FwSmth[,id]
    #Fwx <- Fwx[,id]
    FwSmthx <- FwSmthx[,id]
    
    #FwSmth <- apply(Fw, 1, smooth.spline)
    # fAstp <- tcrossprod(bs[,1], Dt0$Xs[,1]) + tcrossprod(PostMat[, grepl("^*s_1_1", colnames(PostMat))], Dt0$Zs_1_1)
    # fMvpa <- tcrossprod(bs[,2], Dt0$Xs[,2]) + tcrossprod(PostMat[, grepl("^*s_2_1", colnames(PostMat))], Dt0$Zs_2_1)
    # 
    # list(Fw =Fw, fAstp = fAstp, fMvpa = fMvpa)
    
    # od1 <- order(Dt0$Xs[,1])
    # od2 <- order(Dt0$Xs[,2])
    if(smoothT){
      #par(mfrow=c(2,2))
      #conditional_smooths(fit,ask = F)
      #plot(conditional_effects(fit,effects = c("MnASTP", "MnMVPA"))) 
      dat1 <- as.matrix(cbind(t(apply(AstpFxsm, 2, quantile, probs=c(0.025,.975))), colMeans(AstpFxsm), colMeans(AstpFsm) ) )
      p1 <- ggmatplot(x = Df$MnASTP, dat1 ,
                      plot_type = "line", color = c("black","black","red","green"),linetype=1,xlab="ASTP",ylab="", main = paste(tau)) + geom_hline(yintercept = 0, colour="grey",size=2,linetype = 2) 
      
      
      dat2 <- as.matrix(cbind(t(apply(MvpaFxsm, 2, quantile, probs=c(0.025,.975))), colMeans(MvpaFxsm), colMeans(MvpaFsm) ) )
      p2 <- ggmatplot(x = px[[2]]$data$MnMVPA, dat2 ,
                      plot_type = "line", color = c("black","black","red","green"),linetype=1,xlab="MVPA",ylab="", main = paste(tau)) + geom_hline(yintercept = 0, colour="grey",size=2,linetype = 2) 
      
      # p1 = plot(conditional_smooths(fit,"s(MnASTP, bs='ps', k=11)"), ask = F) #s(MnASTP, bs='ps', k=11)
      # p2 = plot(conditional_smooths(fit,"s(MnMVPA, bs='ps', k=11)"), ask = F) # s(MnMVPA)
      #p1 = plot(conditional_smooths(fit,"s(MnASTP)"), ask = F) #s(MnASTP, bs='ps', k=11)
      #p2 = plot(conditional_smooths(fit,"s(MnMVPA)"), ask = F) # s(MnMVPA)
      Reso <- cbind(t(apply(FwSmthx,2,quantile, probs=c(0.025, 0.975))), colMeans(FwSmthx), colMeans(FwSmth))
      p3 <- ggmatplot(x, Reso ,
                      plot_type = "line", color = c("black","black","red","green"),linetype=1,xlab="Time",ylab="", main = paste(tau)) + geom_hline(yintercept = 0, colour="grey",size=2,linetype = 2) 
      
      #print(p3)
      #print(wrap_plots(p[[1]], p[[2]], p3))
      pf3 =  (p1 | p2) / p3 
      
      
      
      
    }else{
      #par(mfrow=c(2,2))}
      # matplot((Dt0$Xs[,1])[od1], (t(rbind(apply(fAstp,2,quantile, probs=c(0.025, 0.975)), colMeans(fAstp))))[od1,], type="l", lwd=3, xlab= "ASTP",ylab="")
      # matplot((Dt0$Xs[,2])[od2], (t(rbind(apply(fMvpa,2,quantile, probs=c(0.025, 0.975)), colMeans(fMvpa))))[od2,], type="l", lwd=3, xlab= "MVPA",ylab="")
      # matplot(x, t(rbind(apply(Fw,2,quantile, probs=c(0.025, 0.975)), colMeans(Fw))), type="l", lty=1,lwd=3, xlab= "Time",ylab="Gamma", main = paste(tau))
      # abline(h=0,lwd=3,col="lightgrey",lty=2)
      # matplot(x, t(rbind(apply(FwSmth,2,quantile, probs=c(0.025, 0.975)), colMeans(FwSmth))), type="l", lty=1, lwd=3, xlab= "Time",ylab="Gamma", main = paste(tau))
      # abline(h=0,lwd=3,col="lightgrey",lty=2)
      Reso <- cbind(t(rbind(apply(FwSmth,2,quantile, probs=c(0.025, 0.975)), colMeans(FwSmthx), colMeans(FwSmth))))
      p3 <- ggmatplot(x*1439, Reso,
                      plot_type = "line", color = c("black","black","red","green"),linetype=c(1,1,2,4), linewidth=2,xlab="Time",ylab="", main = paste(tau)) + geom_hline(yintercept = 0, colour="grey",size=2,linetype = 2) 
      
      p3.1 <- p3 + theme(axis.text = element_text(face="bold"),
                         axis.text.x = element_text(size=12, face="bold", colour = "black"), 
                         # axis.text.y = element_text(size=12,  colour = "black"), # unbold
                         axis.title.x = element_text(size=14, face="bold", colour = "black"),
                         axis.text.y = element_text(size=12, face="bold", colour = "black")) + scale_fill_discrete(name = " ", labels = c("", "2.5%-97.5%", "ME Correct.","Naive")) + scale_x_continuous(
                           name = "Time",
                           breaks = seq(0, 1439, by = 120),
                           labels = timeHM_formatter, guide = guide_axis(angle = 45)) + labs(fill=' ') + theme(legend.position="none")
      print(p3.1)
    }
  }
  
  #--- Takes 2 output
  PostAnalyStanV3 <- function(fit, fitx,fitx2 , Bx=bs2, Dt0, Df, df0=10, smoothT = F, sig=0.1){
    
    PostMat <- as.matrix(fit)
    PostMatx <- as.matrix(fitx)
    PostMatx2 <- as.matrix(fitx2)
    
    tau <-  PostMat[1, grepl("tau",colnames(PostMat))]
    b <- PostMat[, c("Intercept","b_Intercept", paste0("b[",1:7,"]"))]
    Gam <- PostMat[, grepl("bwa", colnames(PostMat))] # 
    Gamx <- PostMatx[, grepl("bwa", colnames(PostMatx))] #
    Gamx2 <- PostMatx2[, grepl("bwa", colnames(PostMatx2))] #
    
    #--- Format time to hh:MM
    timeHM_formatter <- function(x) {
      h <- floor(x/60)
      m <- floor(x %% 60)
      lab <- sprintf("%d:%02d", h, m) # Format the strings as HH:MM
      return(lab)
    }
    
    
    
    if(smoothT){
      betastp <- PostMat[,c("bs[1]", grepl("s_1_1", colnames(PostMat)))] 
      betmvpa <- PostMat[,c("bs[2]", grepl("s_2_1", colnames(PostMat)))]
      
      betastpx <- PostMat[,c("bs[1]", grepl("s_1_1", colnames(PostMatx)))]
      betmvpax <- PostMat[,c("bs[2]", grepl("s_2_1", colnames(PostMatx)))]
      
      #--- Now collect basis and all
      Zastp <- cbind(Dt0$Xs[,1], Dt0$Zs_1_1)
      Zmvpa <- cbind(Dt0$Xs[,2], Dt0$Zs_2_1)
    }
    # bs <- PostMat[, grepl("bs", colnames(PostMat))] # 
    #Gam <- PostMat[, grepl("b_W", colnames(PostMat))] #
    #---------------------------------------------------------------------------
    #--- Plot the estimates
    #---------------------------------------------------------------------------
    Fw <- Gam %*% t(Bx)
    Fwx <- Gamx %*% t(Bx)
    Fwx2 <- Gamx2 %*% t(Bx)
    
    if(smoothT){
      AstpF <- betastp %*% t(Zastp)
      MvpaF <- betmvpa %*% t(Zmvpa)
      
      AstpFx <- betastpx %*% t(Zastp)
      MvpaFx <- betmvpax %*% t(Zmvpa)
    }
    
    FwSmth <- matrix(NA, ncol=ncol(Fw), nrow=nrow(Fw))
    FwSmthx <- matrix(NA, ncol=ncol(Fwx),nrow=nrow(Fwx))
    FwSmthx2 <- matrix(NA, ncol=ncol(Fwx2),nrow=nrow(Fwx2))
    
    #--- Smooth these results
    if(smoothT){
      AstpFsm <- matrix(NA, ncol=ncol(AstpF), nrow=nrow(AstpF))
      MvpaFsm <- matrix(NA, ncol=ncol(MvpaF), nrow=nrow(MvpaF))
      
      AstpFxsm <- matrix(NA, ncol=ncol(AstpFx), nrow=nrow(AstpFx))
      MvpaFxsm <- matrix(NA, ncol=ncol(MvpaFx), nrow=nrow(MvpaFx))
    }
    
    x = seq(-.1,1.1,length = ncol(Fw))
    for(i in 1:nrow(Fw)){
      FwSmth[i,] <- smooth.spline(x,Fw[i,], df=df0)$y
      FwSmthx[i,] <- smooth.spline(x,Fwx[i,], df=df0)$y
      FwSmthx2[i,] <- smooth.spline(x,Fwx2[i,], df=df0)$y
      
      if(smoothT){
        AstpFsm[i,] <- smooth.spline(Zastp[,1],AstpF[i,], df=df0)$y
        MvpaFsm[i,] <- smooth.spline(Zmvpa[,1],MvpaF[i,], df=df0)$y
        
        AstpFxsm[i,] <- smooth.spline(Zastp[,1],AstpFx[i,], df=df0)$y
        MvpaFxsm[i,] <- smooth.spline(Zmvpa[,1],MvpaFx[i,], df=df0)$y
      }
    }
    id <- which(x > 0 & x < 1)
    x <- x[id]
    Fw <- Fw[,id]
    FwSmth <- FwSmth[,id]
    Fwx <- Fwx[,id]
    FwSmthx <- FwSmthx[,id]
    FwSmthx2 <- FwSmthx2[,id]
    #FwSmth <- apply(Fw, 1, smooth.spline)
    # fAstp <- tcrossprod(bs[,1], Dt0$Xs[,1]) + tcrossprod(PostMat[, grepl("^*s_1_1", colnames(PostMat))], Dt0$Zs_1_1)
    # fMvpa <- tcrossprod(bs[,2], Dt0$Xs[,2]) + tcrossprod(PostMat[, grepl("^*s_2_1", colnames(PostMat))], Dt0$Zs_2_1)
    # 
    # list(Fw =Fw, fAstp = fAstp, fMvpa = fMvpa)
    
    # od1 <- order(Dt0$Xs[,1])
    # od2 <- order(Dt0$Xs[,2])
    if(smoothT){
      #par(mfrow=c(2,2))
      #conditional_smooths(fit,ask = F)
      #plot(conditional_effects(fit,effects = c("MnASTP", "MnMVPA"))) 
      dat1 <- as.matrix(cbind(t(apply(AstpFxsm, 2, quantile, probs=c(0.025,.975))), colMeans(AstpFxsm), colMeans(AstpFsm) ) )
      p1 <- ggmatplot(x = Df$MnASTP, dat1 ,
                      plot_type = "line", color = c("black","black","red","green"),linetype=1,xlab="ASTP",ylab="", main = paste(tau)) + geom_hline(yintercept = 0, colour="grey",size=2,linetype = 2) 
      
      
      dat2 <- as.matrix(cbind(t(apply(MvpaFxsm, 2, quantile, probs=c(0.025,.975))), colMeans(MvpaFxsm), colMeans(MvpaFsm) ) )
      p2 <- ggmatplot(x = px[[2]]$data$MnMVPA, dat2 ,
                      plot_type = "line", color = c("black","black","red","green"),linetype=1,xlab="MVPA",ylab="", main = paste(tau)) + geom_hline(yintercept = 0, colour="grey",size=2,linetype = 2) 
      
      # p1 = plot(conditional_smooths(fit,"s(MnASTP, bs='ps', k=11)"), ask = F) #s(MnASTP, bs='ps', k=11)
      # p2 = plot(conditional_smooths(fit,"s(MnMVPA, bs='ps', k=11)"), ask = F) # s(MnMVPA)
      #p1 = plot(conditional_smooths(fit,"s(MnASTP)"), ask = F) #s(MnASTP, bs='ps', k=11)
      #p2 = plot(conditional_smooths(fit,"s(MnMVPA)"), ask = F) # s(MnMVPA)
      Reso <- cbind(t(apply(FwSmthx,2,quantile, probs=c(0.025, 0.975))), colMeans(FwSmthx), colMeans(FwSmth))
      p3 <- ggmatplot(x, Reso ,
                      plot_type = "line", color = c("black","black","red","green"),linetype=1,xlab="Time",ylab="", main = paste(tau)) + geom_hline(yintercept = 0, colour="grey",size=2,linetype = 2) 
      
      #print(p3)
      #print(wrap_plots(p[[1]], p[[2]], p3))
      pf3 =  (p1 | p2) / p3 
      
      
      
      
    }else{
      #par(mfrow=c(2,2))}
      # matplot((Dt0$Xs[,1])[od1], (t(rbind(apply(fAstp,2,quantile, probs=c(0.025, 0.975)), colMeans(fAstp))))[od1,], type="l", lwd=3, xlab= "ASTP",ylab="")
      # matplot((Dt0$Xs[,2])[od2], (t(rbind(apply(fMvpa,2,quantile, probs=c(0.025, 0.975)), colMeans(fMvpa))))[od2,], type="l", lwd=3, xlab= "MVPA",ylab="")
      # matplot(x, t(rbind(apply(Fw,2,quantile, probs=c(0.025, 0.975)), colMeans(Fw))), type="l", lty=1,lwd=3, xlab= "Time",ylab="Gamma", main = paste(tau))
      # abline(h=0,lwd=3,col="lightgrey",lty=2)
      # matplot(x, t(rbind(apply(FwSmth,2,quantile, probs=c(0.025, 0.975)), colMeans(FwSmth))), type="l", lty=1, lwd=3, xlab= "Time",ylab="Gamma", main = paste(tau))
      # abline(h=0,lwd=3,col="lightgrey",lty=2)
      Reso <- cbind(t(rbind(apply(FwSmthx,2,quantile, probs=c(sig/2, 1 - sig/2)), colMeans(FwSmthx))), colMeans(FwSmth), colMeans(FwSmthx2))
      #Reso <- cbind(t(rbind(apply(FwSmthx,2,quantile, probs=c(0.025, 0.975)), colMeans(FwSmthx))), colMeans(FwSmth), colMeans(FwSmthx2))
      p3 <- ggmatplot(x*1439, Reso,
                      plot_type = "line", color = c("black","black","red","green","coral"),linetype=c(1,1,5,3, 6), linewidth=2,xlab="Time",ylab="", main = paste(tau)) + geom_hline(yintercept = 0, colour="grey",size=2,linetype = 2) 
      
      p3.1 <- p3 + theme(axis.text = element_text(face="bold"),
                         axis.text.x = element_text(size=12, face="bold", colour = "black"), 
                         # axis.text.y = element_text(size=12,  colour = "black"), # unbold
                         axis.title.x = element_text(size=14, face="bold", colour = "black"),
                         axis.text.y = element_text(size=12, face="bold", colour = "black")) + scale_fill_discrete(name = " ", labels = c("", "2.5%-97.5%", "ME Correct.","Naive")) + scale_x_continuous(
                           name = "Time",
                           breaks = seq(0, 1439, by = 120),
                           labels = timeHM_formatter, guide = guide_axis(angle = 45)) + labs(fill=' ') + theme(legend.position="none")
      print(p3.1)
    }
  }
  
  
  #---- Plot ehf function data
  PostAnalyStan(fitCase2[[l]],  bs2, tau0, df0=10, smoothT = F)
  
  
  #-------------------------------------------------------------------------------
  # Case 2b - Without MVPA and ASTP (smooth Wt)  (with mixture distribution)
  #-------------------------------------------------------------------------------
  {
    StandCodeVGalLin <- "
      // > fit_fakeqGal3a.5$model
      // generated with brms 2.18.0
      functions {
        
        /*
        A = -est*p_neg + .5*pow(gam, 2)*pow(p_neg/p_pos, 2) + log(Phi_approx(a2-a3)) + log1m_exp(fabs(log(Phi_approx(a2-a3)) - log(Phi_approx(a2)))); 
        gam = (gamU - gamL) * ligam + gamL;
        real gam = (gamU - gamL) * ligam + gamL;
        real GAL2_lpdf(real y, real mu, real sigma, real ligam, real tau, real gamL, real gamU){
        real GAL2_rng(real mu, real sigma, real ligam, real tau, real gamL, real gamU){
         */
           /* helper function for asym_laplace_lpdf
        * Args:
          *   y: the response value
        *   tau: quantile parameter in (0, 1)
        */
          real rho_quantile(real y, real tau) {
            if (y < 0) {
              return y * (tau - 1);
            } else {
              return y * tau;
            }
          }
        /* asymmetric laplace log-PDF for a single response
        * Args:
          *   y: the response value
        *   mu: location parameter
        *   sigma: positive scale parameter
        *   tau: quantile parameter in (0, 1)
        * Returns:
          *   a scalar to be added to the log posterior
        */
          real asym_laplace_lpdf(real y, real mu, real sigma, real tau) {
            return log(tau * (1 - tau)) -
              log(sigma) -
              rho_quantile((y - mu) / sigma, tau);
          }
        /* asymmetric laplace log-CDF for a single quantile
        * Args:
          *   y: a quantile
        *   mu: location parameter
        *   sigma: positive scale parameter
        *   tau: quantile parameter in (0, 1)
        * Returns:
          *   a scalar to be added to the log posterior
        */
          real asym_laplace_lcdf(real y, real mu, real sigma, real tau) {
            if (y < mu) {
              return log(tau) + (1 - tau) * (y - mu) / sigma;
            } else {
              return log1m((1 - tau) * exp(-tau * (y - mu) / sigma));
            }
          }
        /* asymmetric laplace log-CCDF for a single quantile
        * Args:
          *   y: a quantile
        *   mu: location parameter
        *   sigma: positive scale parameter
        *   tau: quantile parameter in (0, 1)
        * Returns:
          *   a scalar to be added to the log posterior
        */
          real asym_laplace_lccdf(real y, real mu, real sigma, real tau) {
            if (y < mu) {
              return log1m(tau * exp((1 - tau) * (y - mu) / sigma));
            } else {
              return log1m(tau) - tau * (y - mu) / sigma;
            }
          }
         
         real GAL2_lpdf(real y, real mu, real sigma, real gam, real tau){
         
         real p_pos;
         real p_neg;
         real a3;
         real a2;
         real p;
         real est;
         real A;
         real B;
         real Res = 0;
         //real gam = ligam;
          p = 1 * (gam < 0) + (tau - 1 * (gam < 0))/(2*Phi(-fabs(gam))*exp(.5*pow(gam, 2)));
          p_pos = p -  1 * (gam > 0);
          p_neg = p -  1 * (gam < 0);  
          est = (y - mu) / sigma; 
          
          if(fabs(gam) > 0){
          a3 = p_pos * (est / fabs(gam));
          a2 = fabs(gam) * (p_neg / p_pos);
          
          if(est/gam > 0){
            A =  0.5*pow(gam, 2)*pow(p_neg/p_pos, 2) - est*p_neg + log_diff_exp(log(Phi_approx(a2-a3)), log(Phi_approx(a2)) ); 
            B =  0.5*pow(gam, 2) - p_pos*est + log(Phi_approx(-fabs(gam) + a3));
            Res = log(2*p*(1-p)) - log(sigma) +  log_sum_exp(A, B);
          }else{
            Res =  log(2*p*(1-p)) - log(sigma) - p_pos * est + 0.5 * pow(gam, 2) + log(Phi_approx(-fabs(gam) ));
          }
          }else{
          Res = asym_laplace_lpdf( y | mu, sigma, tau); 
          }
           
          return Res;
         }
        
        real GAL2_rng(real mu, real sigma, real gam, real tau){
        
           real A;
           real B;
           real C;
           real p;
           real hi;
           real nui;
           real mui=0;
           real Up = uniform_rng(.5, 1.0);
           
          // real gam = ligam;
           p = (gam < 0) + (tau - (gam < 0))/(2*Phi_approx(-fabs(gam))*exp(.5*pow(gam, 2)));
           A = (1 - 2*p)/(p - pow(p,2));
           B = 2/(p - pow(p,2));
           C = 1/((gam > 0) - p);
           
            hi = sigma * inv_Phi(Up);
           nui = sigma * exponential_rng(1);
           mui += mu + A * nui + C * fabs(gam) * hi;
        
           return normal_rng(mui, sqrt(sigma*B*nui));
        }
        
      }
      data {
        int<lower=1> pn; //Number of basis;
        int<lower=1> N;  // total number of observations
        vector[N] Y;  // response variable
        int<lower=1> K;  // number of population-level effects
        matrix[N, K] X;  // population-level design matrix
        int prior_only;  // should the likelihood be ignored?
        matrix[N,pn] Wn;
          vector[pn] M;
         matrix[pn, pn] V;
         vector[2] Bd;
         real<lower=0,upper=1> tau0;
         int<lower=1> Kep;
      }
      transformed data {
        int Kc = K - 1;
        matrix[N, Kc] Xc;  // centered version of X without an intercept
        vector[Kc] means_X;  // column means of X before centering
        for (i in 2:K) {
          means_X[i - 1] = mean(X[, i]);
          Xc[, i - 1] = X[, i] - means_X[i - 1];
        }
      }
      parameters {
        simplex[Kep] pis;
        vector[Kc] b;  // population-level effects
        real Intercept;  // temporary intercept for centered predictors
        vector[pn] bw;
        real<lower=0.01,upper=.99> psi;
         
        // standarized spline coefficients
        vector<lower=0>[Kep] sigma;  // dispersion parameter
        //ordered<lower=Bd[1],upper=Bd[2]>[Kep] gam;
        vector<lower=Bd[1],upper=Bd[2]>[Kep] gam;
        real<lower=0> teta;
      }
      transformed parameters {
        //bwa ar(1) stationary prior
         vector[pn] bwa;
        real tau = tau0;
        real lprior = 0;  // prior contributions to the log posterior
        // compute bwa
        bwa[1] = bw[1];
        for(i in 2:pn)
          bwa[i] = psi*bwa[i-1] + bw[i]*teta; 
          
        lprior += normal_lpdf(b | 0.0, 10);
        lprior += std_normal_lpdf(bw);
        lprior += uniform_lpdf(psi | 0.01, .99);
        lprior += uniform_lpdf(gam | Bd[1], Bd[2]);
        //lprior += cauchy_lpdf(teta| 0, 1) - 1* cauchy_lccdf(0| 0, 1);
        //lprior += gamma_lpdf(teta| 1, 1);
        //lprior += weibull_lpdf(teta| 1, 2.5);
        lprior += std_normal_lpdf(teta) - 1* log(0.5);
        lprior += student_t_lpdf(Intercept | 3, 28.7, 5.6);
        //lprior += normal_lpdf(bs | 0, 5);
        // lprior += student_t_lpdf(sds_1_1 | 3, 0, 5.6)
        //   - 1 * student_t_lccdf(0 | 3, 0, 5.6);
        // lprior += student_t_lpdf(sds_2_1 | 3, 0, 5.6)
        //   - 1 * student_t_lccdf(0 | 3, 0, 5.6);
        lprior += student_t_lpdf(sigma | 3, 0, 5.6)
          - 1 * student_t_lccdf(0 | 3, 0, 5.6);
      }
      model {
        // likelihood including constants
        if (!prior_only) {
         // pis
         
          vector[Kep] log_pis = log(pis);
          // initialize linear predictor term + Xs * bs
          vector[N] mu = rep_vector(0.0, N);
          mu += Intercept + Xc * b + Wn * bwa;
          for (n in 1:N) {
          //  target += GAL2_lpdf(Y[n] | mu[n], sigma, ligam, tau0);
            vector[Kep] lps = log_pis;
            for(l in 1:Kep)
              lps[l] +=  GAL2_lpdf(Y[n] | mu[n], sigma[l], gam[l], tau0);
              
          target += log_sum_exp(lps);
          //log_lik[n] = log_sum_exp(lps);
          }
        }
        // priors including constants
        target += lprior;
        //target += std_normal_lpdf(zs_1_1);
        //target += std_normal_lpdf(zs_2_1);
      }
      generated quantities {
        // actual population-level intercept
        real b_Intercept = Intercept - dot_product(means_X, b);
        vector[N] mu = rep_vector(0.0, N);
        vector[N] log_lik = rep_vector(0.0, N);
        vector[Kep] lps0 = log(pis);
        vector[Kep] lps1 = log(pis);
          mu += Intercept + Xc * b + Wn * bwa;
         
        for(n in 1:N) {
             lps1 = log(pis);
            for(l in 1:Kep)
              lps1[l] +=  GAL2_lpdf(Y[n] | mu[n], sigma[l], gam[l], tau0);
              
          log_lik[n] += log_sum_exp(lps1);
          }
        //Vector[N] ASTP = Xs[1:N,1] * bs[1] + Zs_1_1 * s_1_1;
        //vector[N] MVPA = Xs[1:N,2] * bs[2] + Zs_2_1 * s_2_1;
      }
      "
  }
  df2b <- df_sampleWt
  
  form1 <- paste0("Y ~ ", paste(c("Gender","Race","HealthCondt2","AgeYR", colnames(df2b)[grepl("W",colnames(df2b))]), collapse = " + "))
  Dt2b <- make_standata(bf(as.formula(form1), quantile = tau0), data = df2b[c(1:nrow(df2b)),], family = asym_laplace())
  
  Dt2b$pn = pn
  Dt2b$Xold <- Dt2b$X
  Dt2b$X <- Dt2b$Xold[,!grepl("W.[0-9]", colnames(Dt2b$Xold))]
  Dt2b$Wn <- Dt2b$Xold[,grepl("W.[0-9]", colnames(Dt2b$Xold))]; dim(Dt2b$Wn)
  Dt2b$M <- numeric(Dt2b$pn)
  Dt2b$V <- as.positive.definite(as.inverse(P.mat(pn) + diag(rep(.01, pn))))*.1
  Dt2b$K <- ncol(Dt2b$X)
  Bd <- GamBnd(tau0)[1:2]
  Dt2b$Bd <- Bd*.9
  Dt2b$tau0 <- tau0
  #Dt2b$X[,ncol(Dt2b$X)] <- Dt2b$X[,ncol(Dt2b$X)]/100
  Dt2b$Kep <- 2
  
  fitCase2b[[l]] <- stan(model_code = StandCodeVGalLin,
                         data = Dt2b, iter = Niter, chains = Nchain, verbose = TRUE,control = list(adapt_delta = 0.99,max_treedepth=11), cores = 2, seed = 1123)
  
  # loo1 <- loo(fit1, save_psis = TRUE)
  # LogLik2b <- extract_log_lik(fitCase2b[[1]], parameter_name = "log_lik")  
  # loo::waic(LogLik2b) 
  # LooVal2b <- loo::loo(LogLik2b, save_psis = TRUE)
  
  # #-------------------------------------------------------------------------------
  # # Case 3
  # #-------------------------------------------------------------------------------
  # formCase3 <- paste0("Y ~ ", paste(c("Gender","Race","HealthCondt2","AgeYR", colnames(df_sampleWt)[grepl("W",colnames(df_sampleWt))]), collapse = " + "))
  # for(pa in varX){
  #   formCase3 <- paste0(formCase3, " + s(", pa, ", bs='ps', k=11)")
  # }
  # 
  # df_sampleWtSd <- df_sampleWt
  # df_sampleWtSd$AgeYR <- df_sampleWtSd$AgeYR/sd(df_sampleWtSd$AgeYR)
  # 
  # stanvars2 <- stanvar(scode = stan_funs2, block = "functions")
  # fitCase3[[l]] <- brms::brm(bf(as.formula(formCase3), tau = tau0), data = df_sampleWt, family = GAL2, 
  #                            stanvars = stanvars2,chains = Nchain,iter = Niter, control = list(adapt_delta = 0.99),  cores = 2, seed = 1123) # init = 0.1,
  # 
  # 
  # # expose_functions(fitCase3[[l]], vectorize = TRUE)
  # # log_lik_GAL2 <- function(i, prep) {
  # #      mu <- brms::get_dpar(prep, "mu", i = i)
  # #   sigma <- brms::get_dpar(prep, "sigma", i = i)
  # #     gam <- brms::get_dpar(prep, "gam", i = i)
  # #     tau <- brms::get_dpar(prep, "tau", i = i)
  # #   #tau <- prep$data$tau[i]
  # #   y <- prep$data$Y[i]
  # #   GAL2_lpdf(y, mu, sigma, gam, tau)
  # # }
  # # 
  # # LooVal3 <- loo(fitCase3[[1]])
  # # 
  # # loo(fitCase3[[1]], LogLik2b)
  # #-------------------------------------------------------------------------------
  # # Case 4
  # #-------------------------------------------------------------------------------
  # df4 <- df_sample
  # df4$AgeYRTru <- df4$AgeYR
  # #df4$AgeYR <- scale(df4$AgeYR, center = T, scale = T)
  # 
  # df4$AgeYR <- df_sample$AgeYR/100
  # df4$MnASTP <- df_sample$MnASTP*6
  # df4$MnMVPA <-  df_sample$MnMVPA/100
  # 
  # df_sampleWt4 <- data.frame(df4, W = Wt)
  # #tau0 = .1
  # 
  # form4 <- paste0("Y ~ ", paste(c("Gender","Race","HealthCondt2","AgeYR", colnames(df_sampleWt4)[grepl("W",colnames(df_sampleWt4))]), collapse = " + "))
  # for(pa in varX){
  #   form4 <- paste0(form4, " + s(", pa, ", bs='ps', k=11)")
  # }
  # 
  # 
  # Dt4 <- brms::make_standata(bf(as.formula(form4), quantile = tau0), data = df_sampleWt4, family = asym_laplace())
  # #, stanvars = stanvars2, prior = bprior,chains = 2,iter = 500, control = list(adapt_delta = 0.97), data2 = list(M = rep(0, 29), V = SigbPrio))
  # Bd = GamBnd(tau0)[1:2]
  # 
  # Dt4$pn = pn
  # Dt4$Xold <- Dt4$X
  # Dt4$X <- Dt4$Xold[,!grepl("W.[0-9]", colnames(Dt4$Xold))]
  # Dt4$Wn <- Dt4$Xold[,grepl("W.[0-9]", colnames(Dt4$Xold))]; dim(Dt4$Wn)
  # Dt4$M <- numeric(Dt4$pn)
  # Dt4$V <- as.positive.definite(as.inverse(P.mat(pn) + diag(rep(.01, pn))))*.1
  # Dt4$K <- ncol(Dt4$X)
  # Bd <- GamBnd(tau0)[1:2]
  # Dt4$Bd <- Bd*.9
  # Dt4$tau0 <- tau0
  # 
  # #-- Option 2 GAL
  # #fitXt[[l]]
  # {
  #   StandCodeVGal <- "
  # // > fit_fakeqGal3a.5$model
  # // generated with brms 2.18.0
  # functions {
  #   
  #   /*
  #   A = -est*p_neg + .5*pow(gam, 2)*pow(p_neg/p_pos, 2) + log(Phi_approx(a2-a3)) + log1m_exp(fabs(log(Phi_approx(a2-a3)) - log(Phi_approx(a2)))); 
  #   gam = (gamU - gamL) * ligam + gamL;
  #   real gam = (gamU - gamL) * ligam + gamL;
  #   real GAL2_lpdf(real y, real mu, real sigma, real ligam, real tau, real gamL, real gamU){
  #   real GAL2_rng(real mu, real sigma, real ligam, real tau, real gamL, real gamU){
  #    */
  #      /* helper function for asym_laplace_lpdf
  #   * Args:
  #     *   y: the response value
  #   *   tau: quantile parameter in (0, 1)
  #   */
  #     real rho_quantile(real y, real tau) {
  #       if (y < 0) {
  #         return y * (tau - 1);
  #       } else {
  #         return y * tau;
  #       }
  #     }
  #   /* asymmetric laplace log-PDF for a single response
  #   * Args:
  #     *   y: the response value
  #   *   mu: location parameter
  #   *   sigma: positive scale parameter
  #   *   tau: quantile parameter in (0, 1)
  #   * Returns:
  #     *   a scalar to be added to the log posterior
  #   */
  #     real asym_laplace_lpdf(real y, real mu, real sigma, real tau) {
  #       return log(tau * (1 - tau)) -
  #         log(sigma) -
  #         rho_quantile((y - mu) / sigma, tau);
  #     }
  #   /* asymmetric laplace log-CDF for a single quantile
  #   * Args:
  #     *   y: a quantile
  #   *   mu: location parameter
  #   *   sigma: positive scale parameter
  #   *   tau: quantile parameter in (0, 1)
  #   * Returns:
  #     *   a scalar to be added to the log posterior
  #   */
  #     real asym_laplace_lcdf(real y, real mu, real sigma, real tau) {
  #       if (y < mu) {
  #         return log(tau) + (1 - tau) * (y - mu) / sigma;
  #       } else {
  #         return log1m((1 - tau) * exp(-tau * (y - mu) / sigma));
  #       }
  #     }
  #   /* asymmetric laplace log-CCDF for a single quantile
  #   * Args:
  #     *   y: a quantile
  #   *   mu: location parameter
  #   *   sigma: positive scale parameter
  #   *   tau: quantile parameter in (0, 1)
  #   * Returns:
  #     *   a scalar to be added to the log posterior
  #   */
  #     real asym_laplace_lccdf(real y, real mu, real sigma, real tau) {
  #       if (y < mu) {
  #         return log1m(tau * exp((1 - tau) * (y - mu) / sigma));
  #       } else {
  #         return log1m(tau) - tau * (y - mu) / sigma;
  #       }
  #     }
  #    
  #    real GAL2_lpdf(real y, real mu, real sigma, real gam, real tau){
  #    
  #    real p_pos;
  #    real p_neg;
  #    real a3;
  #    real a2;
  #    real p;
  #    real est;
  #    real A;
  #    real B;
  #    real Res = 0;
  #    //real gam = ligam;
  #     p = 1 * (gam < 0) + (tau - 1 * (gam < 0))/(2*Phi(-fabs(gam))*exp(.5*pow(gam, 2)));
  #     p_pos = p -  1 * (gam > 0);
  #     p_neg = p -  1 * (gam < 0);  
  #     est = (y - mu) / sigma; 
  #     
  #     if(fabs(gam) > 0){
  #     a3 = p_pos * (est / fabs(gam));
  #     a2 = fabs(gam) * (p_neg / p_pos);
  #     
  #     
  #     if(est/gam > 0){
  #       A =  0.5*pow(gam, 2)*pow(p_neg/p_pos, 2) - est*p_neg + log_diff_exp(log(Phi_approx(a2-a3)), log(Phi_approx(a2)) ); 
  #       B =  0.5*pow(gam, 2) - p_pos*est + log(Phi_approx(-fabs(gam) + a3));
  #       Res = log(2*p*(1-p)) - log(sigma) +  log_sum_exp(A, B);
  #     }else{
  #       Res =  log(2*p*(1-p)) - log(sigma) - p_pos * est + 0.5 * pow(gam, 2) + log(Phi_approx(-fabs(gam) ));
  #     }
  #     }else{
  #     Res = asym_laplace_lpdf( y | mu, sigma, tau); 
  #     }
  #      
  #     return Res;
  #    }
  #   
  #   real GAL2_rng(real mu, real sigma, real gam, real tau){
  #   
  #      real A;
  #      real B;
  #      real C;
  #      real p;
  #      real hi;
  #      real nui;
  #      real mui=0;
  #      real Up = uniform_rng(.5, 1.0);
  #      
  #     // real gam = ligam;
  #      p = (gam < 0) + (tau - (gam < 0))/(2*Phi_approx(-fabs(gam))*exp(.5*pow(gam, 2)));
  #      A = (1 - 2*p)/(p - pow(p,2));
  #      B = 2/(p - pow(p,2));
  #      C = 1/((gam > 0) - p);
  #      
  #       hi = sigma * inv_Phi(Up);
  #      nui = sigma * exponential_rng(1);
  #      mui += mu + A * nui + C * fabs(gam) * hi;
  #   
  #      return normal_rng(mui, sqrt(sigma*B*nui));
  #   }
  #   
  # }
  # data {
  # int<lower=1> pn; //Number of basis;
  #   int<lower=1> N;  // total number of observations
  #   vector[N] Y;  // response variable
  #   int<lower=1> K;  // number of population-level effects
  #   matrix[N, K] X;  // population-level design matrix
  #   // data for splines
  #   int Ks;  // number of linear effects
  #   matrix[N, Ks] Xs;  // design matrix for the linear effects
  #    
  #   int nb_1;  // number of bases
  #   int knots_1[nb_1];  // number of knots
  #   // basis function matrices
  #   matrix[N, knots_1[1]] Zs_1_1;
  #   
  #   int nb_2;  // number of bases
  #   int knots_2[nb_2];  // number of knots
  #   // basis function matrices
  #   matrix[N, knots_2[1]] Zs_2_1;
  #   int prior_only;  // should the likelihood be ignored?
  #   matrix[N,pn] Wn;
  #     vector[pn] M;
  #    matrix[pn, pn] V;
  #    vector[2] Bd;
  #    real<lower=0,upper=1> tau0;
  # }
  # transformed data {
  #   int Kc = K - 1;
  #   matrix[N, Kc] Xc;  // centered version of X without an intercept
  #   vector[Kc] means_X;  // column means of X before centering
  #   for (i in 2:K) {
  #     means_X[i - 1] = mean(X[, i]);
  #     Xc[, i - 1] = X[, i] - means_X[i - 1];
  #   }
  # }
  # parameters {
  #   vector[Kc] b;  // population-level effects
  #   real Intercept;  // temporary intercept for centered predictors
  #   vector[Ks] bs;  // spline coefficients
  #   vector[pn] bw;
  #   real<lower=0.01,upper=.99> psi;
  #    
  #   // standarized spline coefficients
  #   vector[knots_1[1]] zs_1_1;
  #   real<lower=0> sds_1_1;  // standard deviations of spline coefficients
  #    
  #   // standarized spline coefficients
  #   vector[knots_2[1]] zs_2_1;
  #   real<lower=0> sds_2_1;  // standard deviations of spline coefficients
  #   real<lower=0> sigma;  // dispersion parameter
  #   real<lower=Bd[1],upper=Bd[2]> gam;
  #   real<lower=0> teta;
  # }
  # transformed parameters {
  #    //bwa ar(1) stationary prior
  #    vector[pn] bwa;
  #   // actual spline coefficients
  #   vector[knots_1[1]] s_1_1;
  #   // actual spline coefficients
  #   vector[knots_2[1]] s_2_1;
  #   //matrix[pn,pn] V1;
  #   real tau = tau0;
  #   real lprior = 0;  // prior contributions to the log posterior
  #   // compute actual spline coefficients
  #   s_1_1 = sds_1_1 * zs_1_1;
  #   //V1 = teta*V;
  #   // compute actual spline coefficients
  #   s_2_1 = sds_2_1 * zs_2_1;
  #   // compute bwa
  #   bwa[1] = bw[1];
  #   for(i in 2:pn)
  #     bwa[i] = psi*bwa[i-1] + bw[i]*teta; 
  #     
  #   lprior += normal_lpdf(b | 0.0, 10);
  #   lprior += std_normal_lpdf(bw);
  #   lprior += uniform_lpdf(psi | 0.01, .99);
  #   lprior += uniform_lpdf(gam | Bd[1], Bd[2]);
  #   //lprior += cauchy_lpdf(teta| 0, 1) - 1* cauchy_lccdf(0| 0, 1);
  #   //lprior += gamma_lpdf(teta| 1, 1);
  #   //lprior += weibull_lpdf(teta| 1, 2.5);
  #   lprior += std_normal_lpdf(teta) - 1* log(0.5);
  #   lprior += student_t_lpdf(Intercept | 3, 28.7, 5.6);
  #   lprior += normal_lpdf(bs | 0, 5);
  #   lprior += student_t_lpdf(sds_1_1 | 3, 0, 5.6)
  #     - 1 * student_t_lccdf(0 | 3, 0, 5.6);
  #   lprior += student_t_lpdf(sds_2_1 | 3, 0, 5.6)
  #     - 1 * student_t_lccdf(0 | 3, 0, 5.6);
  #   lprior += student_t_lpdf(sigma | 3, 0, 5.6)
  #     - 1 * student_t_lccdf(0 | 3, 0, 5.6);
  # }
  # model {
  #   // likelihood including constants
  #   if (!prior_only) {
  #     // initialize linear predictor term
  #     vector[N] mu = rep_vector(0.0, N);
  #     mu += Intercept + Xc * b + Wn * bwa + Xs * bs + Zs_1_1 * s_1_1 + Zs_2_1 * s_2_1;
  #     for (n in 1:N) {
  #     //  target += GAL2_lpdf(Y[n] | mu[n], sigma, ligam, tau0);
  #         target += GAL2_lpdf(Y[n] | mu[n], sigma, gam, tau0);
  #     }
  #   }
  #   // priors including constants
  #   target += lprior;
  #   target += std_normal_lpdf(zs_1_1);
  #   target += std_normal_lpdf(zs_2_1);
  # }
  # generated quantities {
  #   // actual population-level intercept
  #   real b_Intercept = Intercept - dot_product(means_X, b);
  #   vector[N] mu = rep_vector(0.0, N);
  #   vector[N] log_lik = rep_vector(0.0, N);
  #         mu += Intercept + Xc * b + Wn * bwa;
  #       for(n in 1:N)
  #             log_lik[n] +=  GAL2_lpdf(Y[n] | mu[n], sigma, gam, tau0);
  # 
  #   //Vector[N] ASTP = Xs[1:N,1] * bs[1] + Zs_1_1 * s_1_1;
  #   //vector[N] MVPA = Xs[1:N,2] * bs[2] + Zs_2_1 * s_2_1;
  # }
  # "
  # }
  # 
  # fitCase4[[l]] <- stan(model_code = StandCodeVGal, 
  #                       data = Dt4, iter = Niter, chains = Nchain, verbose = TRUE,control = list(adapt_delta = 0.99,max_treedepth=11), cores = 2, seed = 1123) 
  # 
  # 
  # 
  # #-------------------------------------------------------------------------------
  # # Case 5
  # #-------------------------------------------------------------------------------
  # df5 <- df_sample
  # df5$AgeYRTru <- df5$AgeYR
  # #df5$AgeYR <- scale(df5$AgeYR, center = T, scale = T)
  # 
  # df5$AgeYR <- df_sample$AgeYR/100
  # df5$MnASTP <- df_sample$MnASTP*6
  # df5$MnMVPA <-  df_sample$MnMVPA/100
  # 
  # df_sampleWt5 <- data.frame(df5, W = Wt)
  # #tau0 = .1
  # 
  # form5 <- paste0("Y ~ ", paste(c("Gender","Race","HealthCondt2","AgeYR", colnames(df_sampleWt5)[grepl("W",colnames(df_sampleWt4))]), collapse = " + "))
  # for(pa in varX){
  #   form5 <- paste0(form5, " + s(", pa, ", bs='ps', k=11)")
  # }
  # 
  # 
  # Dt5 <- brms::make_standata(bf(as.formula(form4), quantile = tau0), data = df_sampleWt5, family = asym_laplace())
  # #, stanvars = stanvars2, prior = bprior,chains = 2,iter = 500, control = list(adapt_delta = 0.97), data2 = list(M = rep(0, 29), V = SigbPrio))
  # Bd = GamBnd(tau0)[1:2]
  # 
  # Dt5$pn = pn
  # Dt5$Xold <- Dt5$X
  # Dt5$X <- Dt5$Xold[,!grepl("W.[0-9]", colnames(Dt5$Xold))]
  # Dt5$Wn <- Dt5$Xold[,grepl("W.[0-9]", colnames(Dt5$Xold))]; dim(Dt5$Wn)
  # Dt5$M <- numeric(Dt5$pn)
  # Dt5$V <- as.positive.definite(as.inverse(P.mat(pn) + diag(rep(.01, pn))))*.1
  # Dt5$K <- ncol(Dt5$X)
  # Bd <- GamBnd(tau0)[1:2]
  # Dt5$Bd <- Bd*.9
  # Dt5$tau0 <- tau0
  # 
  # #-- Option 2 GAL
  # #fitXt[[l]]
  # {
  #   StandCodeVGal5 <- "
  #   // > fit_fakeqGal3a.5$model
  #   // generated with brms 2.18.0
  #   functions {
  #     
  #     /*
  #     A = -est*p_neg + .5*pow(gam, 2)*pow(p_neg/p_pos, 2) + log(Phi_approx(a2-a3)) + log1m_exp(fabs(log(Phi_approx(a2-a3)) - log(Phi_approx(a2)))); 
  #     gam = (gamU - gamL) * ligam + gamL;
  #     real gam = (gamU - gamL) * ligam + gamL;
  #     real GAL2_lpdf(real y, real mu, real sigma, real ligam, real tau, real gamL, real gamU){
  #     real GAL2_rng(real mu, real sigma, real ligam, real tau, real gamL, real gamU){
  #      */
  #        /* helper function for asym_laplace_lpdf
  #     * Args:
  #       *   y: the response value
  #     *   tau: quantile parameter in (0, 1)
  #     */
  #       real rho_quantile(real y, real tau) {
  #         if (y < 0) {
  #           return y * (tau - 1);
  #         } else {
  #           return y * tau;
  #         }
  #       }
  #     /* asymmetric laplace log-PDF for a single response
  #     * Args:
  #       *   y: the response value
  #     *   mu: location parameter
  #     *   sigma: positive scale parameter
  #     *   tau: quantile parameter in (0, 1)
  #     * Returns:
  #       *   a scalar to be added to the log posterior
  #     */
  #       real asym_laplace_lpdf(real y, real mu, real sigma, real tau) {
  #         return log(tau * (1 - tau)) -
  #           log(sigma) -
  #           rho_quantile((y - mu) / sigma, tau);
  #       }
  #     /* asymmetric laplace log-CDF for a single quantile
  #     * Args:
  #       *   y: a quantile
  #     *   mu: location parameter
  #     *   sigma: positive scale parameter
  #     *   tau: quantile parameter in (0, 1)
  #     * Returns:
  #       *   a scalar to be added to the log posterior
  #     */
  #       real asym_laplace_lcdf(real y, real mu, real sigma, real tau) {
  #         if (y < mu) {
  #           return log(tau) + (1 - tau) * (y - mu) / sigma;
  #         } else {
  #           return log1m((1 - tau) * exp(-tau * (y - mu) / sigma));
  #         }
  #       }
  #     /* asymmetric laplace log-CCDF for a single quantile
  #     * Args:
  #       *   y: a quantile
  #     *   mu: location parameter
  #     *   sigma: positive scale parameter
  #     *   tau: quantile parameter in (0, 1)
  #     * Returns:
  #       *   a scalar to be added to the log posterior
  #     */
  #       real asym_laplace_lccdf(real y, real mu, real sigma, real tau) {
  #         if (y < mu) {
  #           return log1m(tau * exp((1 - tau) * (y - mu) / sigma));
  #         } else {
  #           return log1m(tau) - tau * (y - mu) / sigma;
  #         }
  #       }
  #      
  #      real GAL2_lpdf(real y, real mu, real sigma, real gam, real tau){
  #      
  #      real p_pos;
  #      real p_neg;
  #      real a3;
  #      real a2;
  #      real p;
  #      real est;
  #      real A;
  #      real B;
  #      real Res = 0;
  #      //real gam = ligam;
  #       p = 1 * (gam < 0) + (tau - 1 * (gam < 0))/(2*Phi(-fabs(gam))*exp(.5*pow(gam, 2)));
  #       p_pos = p -  1 * (gam > 0);
  #       p_neg = p -  1 * (gam < 0);  
  #       est = (y - mu) / sigma; 
  #       
  #       if(fabs(gam) > 0){
  #       a3 = p_pos * (est / fabs(gam));
  #       a2 = fabs(gam) * (p_neg / p_pos);
  #       
  #       
  #       if(est/gam > 0){
  #         A =  0.5*pow(gam, 2)*pow(p_neg/p_pos, 2) - est*p_neg + log_diff_exp(log(Phi_approx(a2-a3)), log(Phi_approx(a2)) ); 
  #         B =  0.5*pow(gam, 2) - p_pos*est + log(Phi_approx(-fabs(gam) + a3));
  #         Res = log(2*p*(1-p)) - log(sigma) +  log_sum_exp(A, B);
  #       }else{
  #         Res =  log(2*p*(1-p)) - log(sigma) - p_pos * est + 0.5 * pow(gam, 2) + log(Phi_approx(-fabs(gam) ));
  #       }
  #       }else{
  #       Res = asym_laplace_lpdf( y | mu, sigma, tau); 
  #       }
  #        
  #       return Res;
  #      }
  #     
  #     real GAL2_rng(real mu, real sigma, real gam, real tau){
  #     
  #        real A;
  #        real B;
  #        real C;
  #        real p;
  #        real hi;
  #        real nui;
  #        real mui=0;
  #        real Up = uniform_rng(.5, 1.0);
  #        
  #       // real gam = ligam;
  #        p = (gam < 0) + (tau - (gam < 0))/(2*Phi_approx(-fabs(gam))*exp(.5*pow(gam, 2)));
  #        A = (1 - 2*p)/(p - pow(p,2));
  #        B = 2/(p - pow(p,2));
  #        C = 1/((gam > 0) - p);
  #        
  #         hi = sigma * inv_Phi(Up);
  #        nui = sigma * exponential_rng(1);
  #        mui += mu + A * nui + C * fabs(gam) * hi;
  #     
  #        return normal_rng(mui, sqrt(sigma*B*nui));
  #     }
  #     
  #   }
  #   data {
  #   int<lower=1> pn; //Number of basis;
  #     int<lower=1> N;  // total number of observations
  #     vector[N] Y;  // response variable
  #     int<lower=1> K;  // number of population-level effects
  #     matrix[N, K] X;  // population-level design matrix
  #     // data for splines
  #     int Ks;  // number of linear effects
  #     matrix[N, Ks] Xs;  // design matrix for the linear effects
  #      
  #     int nb_1;  // number of bases
  #     int knots_1[nb_1];  // number of knots
  #     // basis function matrices
  #     matrix[N, knots_1[1]] Zs_1_1;
  #     
  #     int nb_2;  // number of bases
  #     int knots_2[nb_2];  // number of knots
  #     // basis function matrices
  #     matrix[N, knots_2[1]] Zs_2_1;
  #     int prior_only;  // should the likelihood be ignored?
  #     matrix[N,pn] Wn;
  #       vector[pn] M;
  #      matrix[pn, pn] V;
  #      vector[2] Bd;
  #      real<lower=0,upper=1> tau0;
  #   }
  #   transformed data {
  #     int Kc = K - 1;
  #     matrix[N, Kc] Xc;  // centered version of X without an intercept
  #     vector[Kc] means_X;  // column means of X before centering
  #     for (i in 2:K) {
  #       means_X[i - 1] = mean(X[, i]);
  #       Xc[, i - 1] = X[, i] - means_X[i - 1];
  #     }
  #   }
  #   parameters {
  #     vector[Kc] b;  // population-level effects
  #     real Intercept;  // temporary intercept for centered predictors
  #     vector[Ks] bs;  // spline coefficients
  #     vector<upper=0>[pn] bw;
  #     real<lower=0.01,upper=.99> psi;
  #      
  #     // standarized spline coefficients
  #     vector[knots_1[1]] zs_1_1;
  #     real<lower=0> sds_1_1;  // standard deviations of spline coefficients
  #      
  #     // standarized spline coefficients
  #     vector[knots_2[1]] zs_2_1;
  #     real<lower=0> sds_2_1;  // standard deviations of spline coefficients
  #     real<lower=0> sigma;  // dispersion parameter
  #     real<lower=Bd[1],upper=Bd[2]> gam;
  #     real<lower=0> teta;
  #   }
  #   transformed parameters {
  #      //bwa ar(1) stationary prior
  #      vector<upper=0>[pn] bwa;
  #     // actual spline coefficients
  #     vector[knots_1[1]] s_1_1;
  #     // actual spline coefficients
  #     vector[knots_2[1]] s_2_1;
  #     //matrix[pn,pn] V1;
  #     real tau = tau0;
  #     real lprior = 0;  // prior contributions to the log posterior
  #     // compute actual spline coefficients
  #     s_1_1 = sds_1_1 * zs_1_1;
  #     //V1 = teta*V;
  #     // compute actual spline coefficients
  #     s_2_1 = sds_2_1 * zs_2_1;
  #     // compute bwa
  #     bwa[1] = bw[1];
  #     for(i in 2:pn)
  #       bwa[i] = psi*bwa[i-1] + bw[i]*teta; 
  #       
  #     lprior += normal_lpdf(b | 0.0, 10);
  #     lprior += std_normal_lpdf(bw) - 1* log(0.5); // imposing bw < 0
  #     lprior += uniform_lpdf(psi | 0.01, .99);
  #     lprior += uniform_lpdf(gam | Bd[1], Bd[2]);
  #     //lprior += cauchy_lpdf(teta| 0, 1) - 1* cauchy_lccdf(0| 0, 1);
  #     //lprior += gamma_lpdf(teta| 1, 1);
  #     //lprior += weibull_lpdf(teta| 1, 2.5);
  #     lprior += std_normal_lpdf(teta) - 1* log(0.5);
  #     lprior += student_t_lpdf(Intercept | 3, 28.7, 5.6);
  #     lprior += normal_lpdf(bs | 0, 5);
  #     lprior += student_t_lpdf(sds_1_1 | 3, 0, 5.6)
  #       - 1 * student_t_lccdf(0 | 3, 0, 5.6);
  #     lprior += student_t_lpdf(sds_2_1 | 3, 0, 5.6)
  #       - 1 * student_t_lccdf(0 | 3, 0, 5.6);
  #     lprior += student_t_lpdf(sigma | 3, 0, 5.6)
  #       - 1 * student_t_lccdf(0 | 3, 0, 5.6);
  #   }
  #   model {
  #     // likelihood including constants
  #     if (!prior_only) {
  #       // initialize linear predictor term
  #       vector[N] mu = rep_vector(0.0, N);
  #       mu += Intercept + Xc * b + Wn * bwa + Xs * bs + Zs_1_1 * s_1_1 + Zs_2_1 * s_2_1;
  #       for (n in 1:N) {
  #       //  target += GAL2_lpdf(Y[n] | mu[n], sigma, ligam, tau0);
  #           target += GAL2_lpdf(Y[n] | mu[n], sigma, gam, tau0);
  #       }
  #     }
  #     // priors including constants
  #     target += lprior;
  #     target += std_normal_lpdf(zs_1_1);
  #     target += std_normal_lpdf(zs_2_1);
  #   }
  #   generated quantities {
  #     // actual population-level intercept
  #     real b_Intercept = Intercept - dot_product(means_X, b);
  #     vector[N] mu = rep_vector(0.0, N);
  #     vector[N] log_lik = rep_vector(0.0, N);
  #         mu += Intercept + Xc * b + Wn * bwa;
  #       for(n in 1:N)
  #             log_lik[n] +=  GAL2_lpdf(Y[n] | mu[n], sigma, gam, tau0);
  # 
  #     //Vector[N] ASTP = Xs[1:N,1] * bs[1] + Zs_1_1 * s_1_1;
  #     //vector[N] MVPA = Xs[1:N,2] * bs[2] + Zs_2_1 * s_2_1;
  #   }
  #   "
  # }
  # 
  # fitCase5[[l]] <- stan(model_code = StandCodeVGal5, 
  #                       data = Dt5, iter = Niter, chains = Nchain, verbose = TRUE,control = list(adapt_delta = 0.99,max_treedepth=11), cores = 2, seed = 1123) 
  # 
  # 
  # 
  # 
  # 
  # 

  }

