library(LaplacesDemon)
library(GeneralizedHyperbolic)
library(mvtnorm)
library(splines)
library(nimble)
library(Matrix)
library(pracma)
library(splines2)
library(truncnorm)
library(emulator)
library(mclust)
library(Rcpp)
library(compiler)
library(TruncatedNormal)
library(inline)
library(data.table)
library(refund)
library(mixtools)
library(KScorrect)
library(Rmpfr)
library(quantreg)
# library(QSOFRfunc) #-- This package will be available on the cluster
cat("\n **** Done loading all libraries ***--- > \n")

# Apparently, script should be run with parameters, so this recovers them
cmdArgs <- commandArgs()
cpchar <- cmdArgs[length(cmdArgs)]
cp <- as.numeric(cpchar)
Meth <- as.character(cmdArgs[length(cmdArgs) - 1])
tau0 <- as.numeric(as.character(cmdArgs[length(cmdArgs) - 2]))
case <- as.character(cmdArgs[length(cmdArgs) - 3])

# populates variables according to the case
switch(case,

    # case 1 - asymptotic verification
    "case1a" = {
        caseN <- "case1a"
        n <- 200
        sig.e <- 1
        sig.x <- 4
        sig.u <- 4
        sig.w <- 1
        rhox <- 0.
        rho.u <- 0.
        rho.m <- 0
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[2]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    }, # normal Error / beta(t) = met[1]
    "case2a" = {
        caseN <- "case2a"
        n <- 500
        sig.e <- 1
        sig.x <- 4
        sig.u <- 4
        sig.w <- 1
        rhox <- 0
        rho.u <- 0
        rho.m <- 0
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[2]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    }, # normal Error/ beta(t) = met[1]
    "case3a" = {
        caseN <- "case3a"
        n <- 1000
        sig.e <- 1
        sig.x <- 4
        sig.u <- 4
        sig.w <- 1
        rhox <- 0
        rho.u <- 0
        rho.m <- 0
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[2]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    },
    # case 2 - Correlated errors
    "case1b" = {
        caseN <- "case1b"
        n <- 200
        sig.e <- 1
        sig.x <- 4
        sig.u <- 4
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.5
        rho.m <- 0.5
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[2]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    }, # normal Error / beta(t) = met[1]
    "case2b" = {
        caseN <- "case2b"
        n <- 500
        sig.e <- 1
        sig.x <- 4
        sig.u <- 4
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.5
        rho.m <- 0.5
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[2]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    }, # normal Error/ beta(t) = met[1]
    "case3b" = {
        caseN <- "case3b"
        n <- 1000
        sig.e <- 1
        sig.x <- 4
        sig.u <- 4
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.5
        rho.m <- 0.5
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[2]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    },

    # case 3 - Various distribution for the responses
    "case1c" = {
        caseN <- "case1c"
        n <- 500
        sig.e <- 1
        sig.x <- 4
        sig.u <- 4
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.5
        rho.m <- 0.5
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[1]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    }, # normal Error / beta(t) = met[1]
    "case2c" = {
        caseN <- "case2c"
        n <- 500
        sig.e <- 1
        sig.x <- 4
        sig.u <- 4
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.5
        rho.m <- 0.5
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[4]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    }, # normal Error/ beta(t) = met[1]
    "case3c" = {
        caseN <- "case3c"
        n <- 500
        sig.e <- 1
        sig.x <- 4
        sig.u <- 4
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.5
        rho.m <- 0.5
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[3]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    },


    # case 4 - Varying error variance  sig.e
    "case1d" = {
        caseN <- "case1d"
        n <- 500
        sig.e <- 4
        sig.x <- 4
        sig.u <- 4
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.5
        rho.m <- 0.5
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[2]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    }, # normal Error / beta(t) = met[1]
    "case2d" = {
        caseN <- "case2d"
        n <- 500
        sig.e <- 16
        sig.x <- 4
        sig.u <- 4
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.5
        rho.m <- 0.5
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[2]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    }, # normal Error/ beta(t) = met[1]

    # case 5 - Varying error variance  sig.x
    "case1e" = {
        caseN <- "case1e"
        n <- 500
        sig.e <- 1
        sig.x <- 2
        sig.u <- 4
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.5
        rho.m <- 0.5
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[2]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    }, # normal Error/ beta(t) = met[1]
    "case2e" = {
        caseN <- "case2e"
        n <- 500
        sig.e <- 1
        sig.x <- 16
        sig.u <- 4
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.5
        rho.m <- 0.5
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distF <- "Norm"
    }, # normal Error/ beta(t) = met[1]
    # "case32" = {caseN ="case32"; n = 500; sig.e=1; sig.x = 4; sig.u = 4; sig.w=1; rhox = 0; rho.u = 0; rho.m = 0; Delta=4;t=50; nrep=5; pn = ceiling(n^{1/3.8})+1; idf = 1; distF <- "Norm"},

    # case 6 - Varying error variance  sig.u
    "case1f" = {
        caseN <- "case1f"
        n <- 500
        sig.e <- 1
        sig.x <- 4
        sig.u <- 1
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.5
        rho.m <- 0.5
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[2]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    }, # normal Error/ beta(t) = met[1]
    "case2f" = {
        caseN <- "case2f"
        n <- 500
        sig.e <- 1
        sig.x <- 4
        sig.u <- 16
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.5
        rho.m <- 0.5
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[2]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    }, # normal Error/ beta(t) = met[1]

    # case g - Varying error variance  rho.u
    "case1g" = {
        caseN <- "case1g"
        n <- 500
        sig.e <- 1
        sig.x <- 4
        sig.u <- 1
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.25
        rho.m <- 0.25
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[2]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    }, # normal Error/ beta(t) = met[1]
    "case2g" = {
        caseN <- "case2g"
        n <- 500
        sig.e <- 1
        sig.x <- 4
        sig.u <- 16
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.75
        rho.m <- 0.75
        Delta <- 1
        t <- 50
        nrep <- 5
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[2]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    }, # normal Error/ beta(t) = met[1]
    "case3g" = {
        caseN <- "case3g"
        n <- 500
        sig.e <- 1
        sig.x <- 4
        sig.u <- 1
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.25
        rho.m <- 0.25
        Delta <- 1
        t <- 50
        nrep <- 2
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[2]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    }, # normal Error/ beta(t) = met[1]
    "case4g" = {
        caseN <- "case4g"
        n <- 500
        sig.e <- 1
        sig.x <- 4
        sig.u <- 16
        sig.w <- 1
        rhox <- 0.5
        rho.u <- 0.75
        rho.m <- 0.75
        Delta <- 1
        t <- 50
        nrep <- 2
        pn <- ceiling(n^{
            1 / 3.8
        }) + 1
        idf <- 1
        distFep <- c("Rst", "Norm", "MixNorm", "Gam")[2]
        distFUe <- c("Rmst", "Mvnorm")[2]
        distFepWe <- c("Rmst", "Mvnorm")[2]
        CovMet <- c("CS", "Ar1")[2]
    }, # normal Error/ beta(t) = met[1]
)

# Source importan functions
MainPath <- paste0("/geode2/home/u070/", userid, "/Quartz/BSOFR/Scripts/ReducCore_Function.R")
source(MainPath)

# More variables
pn <- 20
Wdist <- "norm"
SeedVal <- runif(n = 1, min = 1000, max = 100000)
set.seed(20004)
distFep <- c("Rst", "Norm", "MixNorm", "Gam")[4]
j0 <- 3
pi.g <- .65
Prob0 <- rbinom(n = n, size = 1, prob = pi.g)
Z <- matrix(rnorm(n = n * (j0 - 1), sd = 1), ncol = c(j0 - 1))
Z <- scale(Z)
Z <- cbind(Z, rbinom(n = n, prob = pi.g, size = 1))

# Simulate data
DataSim <- DataSimReplicate(pn, n, t, a, j0 = 3, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep, distFUe, distFepWe, CovMet = c("CS", "Ar1")[2], sig.e, idf = c(1:4)[3], nrep, Wdist, Z)

# Assign data
Y <- DataSim$Y
W <- DataSim$Data$W
M <- DataSim$Data$M
X <- DataSim$Data$X
a <- DataSim$a
Delt <- DataSim$Delt
Z <- DataSim$Z
Xn <- DataSim$Xn # (X%*%bs2)/length(a)
if (Meth == "Wrep") {
    W <- DataSim$Data$Wrep
    M <- apply(DataSim$Data$Mall, c(1, 2), mean)
}

# Run MCMC
set.seed(round(tau0 * cp) + SeedVal)
tau0 <- tau0
tunprop <- 0.75 ## acceptance rate around 48%
tunprop <- 1.5 ## acceptance rate around 42%
tunprop <- 3.5 ## acceptance rate around 42%
g0 <- 15
Nsim <- 10000
MCMCSample01 <- FinalMeth(Meth, Y, W, M, Z, a, Nsim = Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0, X, g0, tunprop)

# Store results
OrgPath <- paste0("/N/scratch/", userid, "/BSOFR", Meth, "/quant_", tau0, "/")
DataPath <- paste0(OrgPath, "case_", caseN, "/")
ifelse(!dir.exists(DataPath), dir.create(DataPath, recursive = T), FALSE)
Path_file <- DataPath
Path_file <- paste(DataPath, "Gamma_", cp, ".csv", sep = "")
write.csv(x = MCMCSample01$Gamma, file = Path_file)
Path_file <- paste(DataPath, "Betaz_", cp, ".csv", sep = "")
write.csv(x = MCMCSample01$BetaZ, file = Path_file)
